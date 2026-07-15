"""Find stable grasp regions in ARCTIC or GRAB sequences.

Ported (minimally) from Get A Grip
(``code_arctic/stable_grasps_finder.py``), keeping only what is needed to *find*
stable grasp segments and emit them as a CSV. Video/render/mask/ego-camera code
from the original is intentionally dropped.

A *stable grasp segment* is a maximal frame interval ``[start, end)`` in which the
hand keeps a consistent contact area on the object: any two frames in the segment
share at least a threshold fraction of contacted object surface (fragment IoU).
Contact is defined by the signed hand-to-object mesh distance being below
``contact_thr`` (0 = object vertex has penetrated inside the hand surface).

Rather than re-porting the original ego-camera reader, this reuses the repo's own
sequence reader (``vis/o3d_visualizer.build_dataset`` + ``pose_sample``), which
already yields per-frame world-space hand/object vertices and faces for both
ARCTIC and GRAB (dispatched on ``cfg.name``). The IoU and object-to-hand
relative-rotation metrics are invariant to the choice of a consistent per-frame
rigid space, so world-space vertices give the same segments as the original
ego-space ones.

Output CSV schema (consumed by ``preprocess_{arctic,grab}.load_stable_grasps``):
    sid_seq_name, side, start, end, angle_diff
where ``sid_seq_name = f"{subject}/{seq_name}"`` (e.g. ``s01/box_grab_01`` for
ARCTIC, ``s1/airplane_lift`` for GRAB), ``side`` is ``left``/``right``,
``[start, end)`` are 0-indexed frame bounds, and ``angle_diff`` is the max
object-to-hand relative rotation (deg) over the segment. The subject is ARCTIC's
``subject`` field, or GRAB's ``sbj_id`` (e.g. ``s1``); the seq_name is ARCTIC's
``action_name`` (already the sequence filename) or GRAB's ``seq_name`` (the raw
filename without extension — unique, unlike GRAB's motion-intent action_name).

Run (in the ``hoi_common`` conda env, CUDA 12.4). The dataset is selected by the
hydra ``config_name`` — ``arctic`` (default) or ``grab``:
    python -m preprocess.stable_grasp_finder                       # ARCTIC
    python -m preprocess.stable_grasp_finder --config-name grab    # GRAB
    python -m preprocess.stable_grasp_finder stable_grasp_out_csv=/path/out.csv \
        stable_grasp_iou_thr=-0.5 stable_grasp_frag_ratio=0.01
Data locations (``data_path`` etc.) come from the selected ``configs/<name>.yaml``
and can be overridden on the command line via hydra.
"""

import os
import os.path as osp

import hydra
import numpy as np
import pandas as pd
import torch
import tqdm
from pytorch3d.ops import knn_gather, knn_points
from pytorch3d.transforms import so3_rotation_angle
import torch.nn.functional as F

from hoidata.utils.proc import proc_numpy, proc_torch_cuda
from hoidata.utils.rot import axis_angle_to_rotmat, rot6d_to_rotmat
from hoidata.vis.o3d_visualizer import build_dataset, pose_sample

LEFT = "left"
RIGHT = "right"
NUM_HAND_VERTS = 778  # MANO vertices per hand


# ---------------------------------------------------------------------------
# Fragmentation (from code_arctic/fragment.py, verbatim)
# ---------------------------------------------------------------------------
def fragmentation_fps(vertices, num_frags):
    """Fragmentation by the furthest point sampling algorithm.

    Args:
        vertices: [num_vertices, 3] ndarray with 3D vertices of the object model.
        num_frags: Number of fragments to define.

    Returns:
        [num_frags, 3] ndarray with fragment centers and [num_vertices] ndarray
        storing for each vertex the ID of the assigned fragment.
    """
    from scipy import spatial

    # Start with the centroid of the model coordinate system.
    frag_centers = [np.mean(vertices, 0)]

    nn_index = spatial.cKDTree(frag_centers)
    nn_dists, _ = nn_index.query(vertices, k=1)

    for _ in range(num_frags):
        # Select the furthest vertex as the next center.
        new_center_ind = np.argmax(nn_dists)
        new_center = vertices[new_center_ind]
        frag_centers.append(vertices[new_center_ind])

        nn_dists[new_center_ind] = -1
        nn_dists = np.minimum(
            nn_dists, np.linalg.norm(vertices - new_center, axis=1))

    # Remove the centroid seed and assign vertices to the fragments.
    frag_centers.pop(0)
    frag_centers = np.array(frag_centers)
    nn_index = spatial.cKDTree(frag_centers)
    _, vertex_frag_ids = nn_index.query(vertices, k=1)
    return frag_centers, vertex_frag_ids


# ---------------------------------------------------------------------------
# Mesh vertex normals + approximate mesh distance
# (from nnutils/mesh_utils_extra.py + code_arctic/mesh_interaction.py, verbatim)
# ---------------------------------------------------------------------------
def compute_face_angles(verts, faces):
    """Per-face interior angles.

    Args:
        verts: (B, V, 3) or (V, 3)
        faces: (F, 3) LongTensor
    Returns:
        face_angles: (B, F, 3) or (F, 3)
    """
    ndim_old = verts.ndim
    if verts.ndim == 2:
        verts = verts.unsqueeze(0)
    bs = verts.size(0)
    nf = faces.size(0)
    vfaces = verts[:, faces]  # (B, F, 3, 3)

    v1, v2, v3 = torch.split(vfaces, 1, dim=2)  # each (B, F, 1, 3)
    e12 = v2 - v1
    e23 = v3 - v2
    e31 = v1 - v3
    l12 = e12.norm(p=2, dim=-1)
    l23 = e23.norm(p=2, dim=-1)
    l31 = e31.norm(p=2, dim=-1)
    va1 = torch.sum(e12 * -e31, dim=-1) / (l12 * l31)
    va2 = torch.sum(e23 * -e12, dim=-1) / (l23 * l12)
    va3 = torch.sum(e31 * -e23, dim=-1) / (l31 * l23)
    face_cosines = torch.cat([va1, va2, va3], dim=2).view(bs, nf, 3)
    face_angles = face_cosines.acos_()
    if ndim_old == 2:
        face_angles = face_angles.squeeze_(0)
    return face_angles


def compute_face_normals(verts, faces):
    """Per-face unit normals (fn = e12 x e13).

    Args:
        verts: (B, V, 3) or (V, 3)
        faces: (F, 3) LongTensor
    Returns:
        face_normals: (B, F, 3) or (F, 3)
    """
    ndim_old = verts.ndim
    if verts.ndim == 2:
        verts = verts.unsqueeze(0)
    vfaces = verts[:, faces]  # (B, F, 3, 3)

    v1, v2, v3 = torch.split(vfaces, 1, dim=2)  # each (B, F, 1, 3)
    e12 = v2 - v1
    e31 = v1 - v3
    vn1 = torch.cross(e12, -e31, dim=-1)  # (B, F, 1, 3)
    vn1 = F.normalize(vn1, p=2, dim=-1)
    vn1 = vn1.squeeze_(2)
    if ndim_old == 2:
        vn1 = vn1.squeeze_(0)
    return vn1


def compute_vert_normals(verts, faces):
    """Per-vertex normals as the angle-weighted mean of adjacent face normals.

    Args:
        verts: (B, V, 3) or (V, 3)
        faces: (F, 3) LongTensor
    Returns:
        vert_normals: (B, V, 3) or (V, 3)
    """
    ndim_old = verts.ndim
    if verts.ndim == 2:
        verts = verts.unsqueeze(0)
    bs = verts.size(0)
    nf = faces.size(0)

    face_normals = compute_face_normals(verts, faces).unsqueeze_(2)  # (B, F, 1, 3)
    face_angles = compute_face_angles(verts, faces).unsqueeze_(3)  # (B, F, 3, 1)
    src = face_normals * face_angles  # (B, F, 3, 3)
    src = src.view(bs, nf * 3, 3)
    index = faces.view(1, nf * 3, 1).expand(bs, nf * 3, 3)
    vn = torch.zeros_like(verts).scatter_add_(dim=1, index=index, src=src)

    vn = F.normalize(vn, p=2, dim=2)
    if ndim_old == 2:
        vn = vn.squeeze_(0)
    return vn


def pcd_mesh_distance_approx(verts, faces, verts_query, along_normal=True):
    """Signed distance of query points to a mesh, approximated at its vertices.

    Args:
        verts: (N, V1, 3) mesh vertices
        faces: (F1, 3) LongTensor mesh faces
        verts_query: (N, V2, 3) query points
        along_normal: if True, distance is the projection onto the nearest
            vertex normal (negative = inside); otherwise the raw distance with the
            sign copied from that projection.
    Returns:
        signed_dist: (N, V2)
    """
    assert isinstance(verts, torch.Tensor)
    assert isinstance(faces, torch.Tensor)
    vn = compute_vert_normals(verts, faces)
    _dists, idx, nn = knn_points(verts_query, verts, K=1, return_nn=True)
    vn_ = knn_gather(vn, idx)  # (N, V2, 1, 3)
    vn_ = vn_[:, :, 0, :]
    nn = nn[:, :, 0, :]

    disp = verts_query - nn
    disp_norm = disp.norm(dim=2)  # (N, V2)
    disp_dir = disp / disp_norm.unsqueeze(-1)  # (N, V2, 3)
    cos = (disp_dir * vn_).sum(dim=2)  # (N, V2)
    if along_normal:
        signed_dist = disp_norm * cos
    else:
        signed_dist = torch.copysign(disp_norm, cos)
    return signed_dist


def mesh_distance_approx(verts1, verts2, faces1, faces2, along_normal=True):
    """Approximate signed vertex-to-mesh distances between two meshes.

    Args:
        verts1: (N, V1, 3), verts2: (N, V2, 3)
        faces1: (F1, 3), faces2: (F2, 3)
    Returns:
        dist1: (N, V1) signed distance of mesh-1 verts to mesh 2
        dist2: (N, V2) signed distance of mesh-2 verts to mesh 1
    """
    d1 = pcd_mesh_distance_approx(verts2, faces2, verts1, along_normal=along_normal)
    d2 = pcd_mesh_distance_approx(verts1, faces1, verts2, along_normal=along_normal)
    return d1, d2


# ---------------------------------------------------------------------------
# Relative object-to-hand pose metrics (from stable_grasps_finder.py, verbatim)
# ---------------------------------------------------------------------------
def relative_poses(T1, T2):
    """Return T12 where T2 = T12 @ T1, i.e. rel_poses[a, b] maps frame a -> b."""
    return torch.einsum("aij,bjk->abik", T2, torch.inverse(T1))


def relative_metrics(T1, T2):
    """Pairwise relative rotation angle (and translation) between two pose sets.

    Args:
        T1: (N, 4, 4), T2: (M, 4, 4)
    Returns:
        rel_angles: (N, M) in degrees
        rel_distance: (N, M) in metres
    """
    rel_poses = relative_poses(T1, T2)  # (N, M, 4, 4)
    rel_rot = rel_poses[..., :3, :3]
    rel_transl = rel_poses[..., :3, -1]
    _n, _m, _, _ = rel_rot.shape
    rel_ang = so3_rotation_angle(
        rel_rot.reshape(_n * _m, 3, 3)).reshape(_n, _m) / np.pi * 180
    rel_dis = rel_transl.norm(dim=-1)  # metres
    return rel_ang, rel_dis


# ---------------------------------------------------------------------------
# Greedy maximum-length segments under a pairwise constraint
# (from stable_grasps_finder.py, verbatim)
# ---------------------------------------------------------------------------
def max_length_under_constraint(err_list, thr_list, min_len):
    """Find non-overlapping segments I1, I2, ... maximizing length s.t. for any
    err in err_list (and matching thr), err[p, q] < thr for all p, q in a segment.

    Args:
        err_list: list of (N, N) error/correlation matrices
        thr_list: matching thresholds

    Returns:
        segments: list of [start, end)
    """
    def get_satisfied_segments(err_list, thr_list, min_len):
        num_criteria = len(thr_list)
        N = len(err_list[0])
        assert len(err_list) == len(thr_list)
        for k in range(num_criteria):
            assert err_list[k].shape == err_list[0].shape, \
                f"Got {err_list[k].shape} != {err_list[0].shape}"

        segments = []
        for i in range(N):
            e = 1e8
            for k in range(num_criteria):
                err = np.asarray(err_list[k])
                thr = thr_list[k]
                _, jj = np.nonzero(err[[i], i:] > thr)
                if len(jj) == 0:  # last segment
                    e = min(e, N)
                else:
                    e = min(e, i + jj[0])
            segment = [int(i), int(e)]  # longest possible
            segments.append(segment)

        segments = [v for v in segments if v[1] - v[0] >= min_len]
        return segments

    segments = get_satisfied_segments(err_list, thr_list, min_len)

    # Greedily pick the longest non-overlapping segments (brute force).
    results = []
    while len(segments) > 0:
        max_seg = [-1, -1, -1]
        for i, j in segments:
            if j - i > max_seg[0]:
                max_seg = [j - i, i, j]
        _, s, e = max_seg
        segments.pop(segments.index([s, e]))
        results.append([s, e])
        segments = [[i, j] for i, j in segments if j <= s or i >= e]
    return results


# ---------------------------------------------------------------------------
# Per-sequence stable-grasp finder (replaces GraspFinderContext)
# ---------------------------------------------------------------------------
class SequenceGraspFinder:
    """Finds stable grasp segments for a single posed ARCTIC sequence.

    Consumes the dict returned by ``vis/o3d_visualizer.pose_sample`` plus the raw
    per-frame object/hand global rotations (for the object-to-hand relative angle).
    Everything is kept in world space; only rigid, per-frame-consistent transforms
    are applied, so the resulting segments match the original ego-space finder.
    """

    def __init__(self, sample, obj_rotmat, hand_rotmat, device="cuda"):
        """
        Args:
            sample: pose_sample dict with keys hand_verts (T, 778*H, 3),
                hand_faces {'left','right'}, hand_sides, obj_verts (T, V, 3),
                obj_faces (F, 3), nframes.
            obj_rotmat: (T, 3, 3) object world rotation per frame.
            hand_rotmat: dict {side: (T, 3, 3)} hand world (global) rotation.
        """
        self.device = device
        self.num_frames = int(sample["nframes"])
        self.obj_verts = proc_torch_cuda(sample["obj_verts"]).to(device)  # (T, V, 3)
        self.fo = torch.as_tensor(
            np.asarray(sample["obj_faces"]), dtype=torch.long, device=device)

        # Split concatenated hand verts into per-side (T, 778, 3) blocks, in the
        # order given by hand_sides (matches action_dataset._getitem_full).
        hand_verts = proc_torch_cuda(sample["hand_verts"]).to(device)
        self.hand_verts = {}
        self.hand_faces = {}
        for blk, side in enumerate(sample["hand_sides"]):
            lo = blk * NUM_HAND_VERTS
            self.hand_verts[side] = hand_verts[:, lo:lo + NUM_HAND_VERTS]
            self.hand_faces[side] = torch.as_tensor(
                np.asarray(sample["hand_faces"][side]),
                dtype=torch.long, device=device)
        self.sides = list(sample["hand_sides"])

        self.obj_rotmat = obj_rotmat  # (T, 3, 3) cuda
        self.hand_rotmat = hand_rotmat  # {side: (T, 3, 3) cuda}

    def _obj2hand_transforms(self, side):
        """Object-to-hand rigid transforms per frame, rotation only.

        T_o2h = R_hand^{-1} @ R_obj, packed as (T, 4, 4) with zero translation
        (translation feeds only a dropped debug plot in the original).
        """
        R_o = self.obj_rotmat  # (T, 3, 3)
        R_h = self.hand_rotmat[side]  # (T, 3, 3)
        R_o2h = torch.matmul(R_h.transpose(1, 2), R_o)  # (T, 3, 3)
        T = R_o2h.shape[0]
        out = torch.eye(4, device=R_o2h.device).unsqueeze(0).repeat(T, 1, 1)
        out[:, :3, :3] = R_o2h
        return out

    def _vert_fragments(self, frag_ratio):
        """Map each object vertex to a fragment id (fps grouping of frame-0 verts)."""
        vo = self.obj_verts[0].cpu().numpy()
        _, frag_mapping = fragmentation_fps(vo, int(len(vo) * frag_ratio))
        return frag_mapping

    def calc_obj_contact_indices(self, side, along_normal=True, frag_ratio=1.0,
                                 contact_thr=0.0):
        """Per-frame set of object (fragment) indices in contact with the hand.

        Returns: list of length num_frames, each a set of object vertex/fragment
        indices whose signed hand-to-object distance is below contact_thr.
        """
        faces_h = self.hand_faces[side]
        faces_o = self.fo
        vh = self.hand_verts[side]  # (T, 778, 3)
        vo = self.obj_verts  # (T, V, 3)

        _, d2 = mesh_distance_approx(vh, vo, faces_h, faces_o, along_normal=along_normal)
        indices_list = []
        for f in range(self.num_frames):
            nz = torch.nonzero(d2[f] < contact_thr).view(-1)
            indices_list.append(set(nz.tolist()))
        if frag_ratio < 1.0:
            frag_mapping = self._vert_fragments(frag_ratio)
            indices_list = [set(map(lambda x: frag_mapping[x], v))
                            for v in indices_list]
        return indices_list

    def compute_indices_ious(self, inds_list):
        """Pairwise IoU (N, N) of the per-frame contacted-index sets."""
        N = self.num_frames
        ious = np.empty([N, N], dtype=np.float32)
        for i in range(N):
            for j in range(i, N):
                ind_i = inds_list[i]
                ind_j = inds_list[j]
                inter = ind_i.intersection(ind_j)
                union = ind_i.union(ind_j)
                ious[i, j] = len(inter) / (len(union) + 1e-5)
                ious[j, i] = ious[i, j]
        return ious

    def find_stable_segments(self, iou_thr, frag_ratio, along_normal=True,
                             contact_thr=0.0, min_len=10):
        """Find stable grasp segments (per active hand) by stable contact area.

        A segment is a maximal frame interval where any two frames share at least
        ``-iou_thr`` fraction of contacted object fragments. Segments shorter than
        ``min_len`` are dropped.

        Args:
            iou_thr: negative number; -iou_thr is the minimum pairwise IoU.
            frag_ratio: float in (0, 1]; <1 groups object verts into fragments.

        Returns:
            list of [side, start, end, angle_diff] with angle_diff the max
            object-to-hand relative rotation (deg) over the segment.
        """
        results = []
        for side in self.sides:
            ind_list = self.calc_obj_contact_indices(
                side, along_normal=along_normal, frag_ratio=frag_ratio,
                contact_thr=contact_thr)
            ious = self.compute_indices_ious(ind_list)
            segs = max_length_under_constraint([-ious], [iou_thr], min_len)

            T_o2h = self._obj2hand_transforms(side)
            angs, _ = relative_metrics(T_o2h, T_o2h)  # (T, T) degrees
            for s, e in sorted(segs):
                angle_diff = angs[s:e, s:e].max().item()
                results.append([side, s, e, angle_diff])
        return results


# ---------------------------------------------------------------------------
# Per-sequence global rotations (for the object-to-hand relative angle)
# ---------------------------------------------------------------------------
def _sequence_global_rotmats(ds, index, item, device="cuda"):
    """World-space object and hand global rotations for one sequence.

    Object rotation comes from ``item["obj_traj"]`` (trans + axis-angle rotvec).
    Hand global rotation comes from the stored x-data (trans + rot6d, first 6 pose
    dims after translation) for whichever hands are active.

    These rotations feed *only* the reported ``angle_diff`` metadata column (the
    max object-to-hand relative rotation over a segment); they do not affect
    segment boundaries, which come purely from contact-area IoU on world-space
    verts. Note GRAB's ``_obj_traj_from_xobj`` negates the object rotvec (returns
    R^{-1}) while the hand rot6d is not negated, so GRAB's ``angle_diff`` uses a
    different object-rotation sign convention than ARCTIC's — fine for a relative
    magnitude used only to tag/inspect segments.

    Returns:
        obj_rotmat: (T, 3, 3) cuda tensor
        hand_rotmat: {side: (T, 3, 3) cuda tensor} for active hands only
    """
    nframes = int(item["nframes"])

    obj_rotvec = proc_torch_cuda(item["obj_traj"][:nframes, 3:6])  # (T, 3)
    obj_rotmat = axis_angle_to_rotmat(obj_rotvec).to(device)  # (T, 3, 3)

    hand_rotmat = {}
    for side in item["hand_sides"]:
        x_hand = ds.x_lhand[index] if side == LEFT else ds.x_rhand[index]
        rot6d = proc_torch_cuda(x_hand[:nframes, 3:9])  # (T, 6) global rot6d
        hand_rotmat[side] = rot6d_to_rotmat(rot6d).to(device)  # (T, 3, 3)
    return obj_rotmat, hand_rotmat


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _sample_subject(sample, ds, index):
    """Subject id for the CSV: ARCTIC's ``subject`` or GRAB's ``sbj_id`` (e.g. s1)."""
    subject = sample.get("subject", "")
    if not subject and getattr(ds, "sbj_id", None) is not None:
        subject = str(ds.sbj_id[index])
    return subject


def _sample_seq_name(sample, ds, index):
    """Unique per-sequence name for the CSV.

    ARCTIC's ``action_name`` is already the sequence filename (e.g.
    ``box_grab_01``), so it is unique.  GRAB's ``action_name`` is the motion
    intent (e.g. ``lift``), shared across objects/subjects, so prefer GRAB's
    ``seq_name`` (the full raw filename without extension, e.g. ``airplane_lift``)
    when present.
    """
    if getattr(ds, "seq_name", None) is not None:
        return str(ds.seq_name[index])
    return sample["action_name"]


def find_all_stable_grasp(cfg, out_csv, iou_thr, frag_ratio,
                          contact_thr=0.0, min_len=10, along_normal=True,
                          dataset_name="arctic", splits=("train", "val", "test")):
    """Find stable grasp segments for every sequence of a dataset and write a CSV.

    Works for both ARCTIC and GRAB (selected by ``dataset_name``); the repo's
    reader poses each dataset's objects (articulated / rigid) transparently.

    The four criteria knobs below trade strictness for coverage. To loosen the
    criteria (find more / longer intervals): raise ``iou_thr`` toward 0, raise
    ``contact_thr`` to a small positive value, and/or lower ``min_len``.

    Args:
        cfg: hydra config (provides data_path, obj_root, mano_root, ...).
        out_csv: output CSV path (created; parent dirs made if needed).
        iou_thr: negative IoU threshold (e.g. -0.5 → min 50% pairwise IoU).
            Raising toward 0 requires less contact-area overlap → longer segments.
        frag_ratio: object-vertex fragmentation ratio in (0, 1] (e.g. 0.01).
            Lower = coarser fragments = more forgiving IoU.
        contact_thr: signed hand→object distance below which an object vertex
            counts as in contact (0.0 = must penetrate the hand surface). A small
            positive value (e.g. 0.005 m) also counts near-touching vertices,
            growing the contact set and yielding larger, more stable regions.
        min_len: segments shorter than this many frames are dropped.
        along_normal: measure the hand→object distance along the surface normal
            (vs. raw signed magnitude).
        dataset_name: "arctic" or "grab"; dispatches the object-posing path.
        splits: dataset splits to cover. Both ARCTIC and GRAB splits partition by
            object, so the default (all three) covers every sequence.
    """
    rows = []
    for split in splits:
        ds, lhand_layer, rhand_layer = build_dataset(cfg, dataset_name, split)
        for index in tqdm.tqdm(range(len(ds)), desc=f"{dataset_name}/{split}"):
            sample = pose_sample(cfg, dataset_name, ds, lhand_layer, rhand_layer, index)
            item = ds[index]
            obj_rotmat, hand_rotmat = _sequence_global_rotmats(ds, index, item)

            finder = SequenceGraspFinder(sample, obj_rotmat, hand_rotmat)
            segments = finder.find_stable_segments(
                iou_thr=iou_thr, frag_ratio=frag_ratio,
                contact_thr=contact_thr, min_len=min_len,
                along_normal=along_normal)

            subject = _sample_subject(sample, ds, index)
            seq_name = _sample_seq_name(sample, ds, index)
            sid_seq_name = f'{subject}/{seq_name}'
            for side, s, e, angle_diff in segments:
                rows.append([sid_seq_name, side, s, e, angle_diff])

    df = pd.DataFrame(
        rows, columns=["sid_seq_name", "side", "start", "end", "angle_diff"])
    os.makedirs(osp.dirname(osp.abspath(out_csv)), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} stable-grasp segments → {out_csv}")


@hydra.main(version_base=None, config_path="../../configs", config_name="arctic")
def main(cfg):
    # Dataset is selected by the hydra config_name (arctic | grab):
    #   python -m preprocess.stable_grasp_finder --config-name grab
    # All criteria knobs are hydra-overridable, e.g. to loosen the criteria:
    #   python -m preprocess.stable_grasp_finder \
    #       stable_grasp_iou_thr=-0.35 stable_grasp_contact_thr=0.005 \
    #       stable_grasp_min_len=5 stable_grasp_out_csv=/path/out.csv
    out_csv = cfg.get("stable_grasp_out_csv", cfg.get("stable_grasp_csv"))
    iou_thr = float(cfg.get("stable_grasp_iou_thr", -0.50))
    frag_ratio = float(cfg.get("stable_grasp_frag_ratio", 0.01))
    contact_thr = float(cfg.get("stable_grasp_contact_thr", 0.0))
    min_len = int(cfg.get("stable_grasp_min_len", 10))
    along_normal = bool(cfg.get("stable_grasp_along_normal", True))
    find_all_stable_grasp(
        cfg, out_csv=out_csv, iou_thr=iou_thr, frag_ratio=frag_ratio,
        contact_thr=contact_thr, min_len=min_len, along_normal=along_normal,
        dataset_name=cfg.name)


if __name__ == "__main__":
    main()
