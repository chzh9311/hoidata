"""
This file implements a visualizer based on Open3D.
It loads data from a dataloader in dataloaders/ and visualizes the hand-object interaction sequence in 3D.
The visualization window shows the action in loop, and the user can rotate the camera to view the scene from different angles.
The visualizer can also save the visualization as a video file.
"""

import argparse
import os.path as osp
import time

import numpy as np
import trimesh
import open3d as o3d
from omegaconf import OmegaConf

from hoidata.dataloaders.action_dataset import ObjectModel
from hoidata.dataloaders.grab_dataset import (
    MotionGRAB, TransitionMotionGRAB, BimanualTransitionMotionGRAB,
    StableGraspGRAB)
from hoidata.dataloaders.arctic_dataset import MotionARCTIC, StableGraspARCTIC
from hoidata.utils.hand_proc import ManoLayerWrapper
from hoidata.utils.proc import process_obj_result, proc_torch_cuda, proc_numpy
from hoidata.utils.proc_arctic import process_object
from hoidata.utils.object_tensors import ObjectTensors


HAND_SIDE_COLORS = {"left": (0.9, 0.75, 0.65), "right": (0.75, 0.85, 0.95)}
OBJECT_COLOR = (0.6, 0.7, 0.9)


def _pose_arctic_object(cfg, item):
    """Articulated ARCTIC object mesh, posed per frame.

    ``item["obj_traj"]`` is (T,6) trans+rotvec and ``item["obj_angle"]`` is (T,)
    radians; ``process_object`` takes both directly (no rot6d conversion needed,
    unlike the GRAB path).
    """
    object_model = ObjectTensors(cfg.obj_root)
    object_model.to("cuda")

    obj_trans = item["obj_traj"][:, :3]
    obj_rotvec = item["obj_traj"][:, 3:6]
    obj_angle = item["obj_angle"].reshape(-1, 1)
    out = process_object(obj_angle, obj_rotvec, obj_trans, item["obj_name"], object_model)

    v_len = int(out["v_len"][0].item())
    f_len = int(out["f_len"][0].item())
    obj_verts = proc_numpy(out["v"][:, :v_len])
    obj_faces = proc_numpy(out["f"][0, :f_len])
    return obj_verts, obj_faces


def _pose_grab_object(cfg, ds, index, item):
    """Rigid GRAB object mesh, posed per frame.

    Uses the dataset's raw ``x_obj`` (trans+rot6d), not ``item["obj_traj"]``
    (trans+rotvec) — ``process_obj_result`` expects the rot6d representation.
    The rotation-sign convention here is best-effort: GRAB has no preprocessed
    data on disk yet to verify against empirically (see plan notes).
    """
    _, _, _, obj_path = ObjectModel(cfg.data_obj_pc_path)(item["obj_name"])
    mesh = trimesh.load(osp.join(cfg.obj_root, obj_path), process=False)

    x_obj = ds.x_obj[index][: item["nframes"]]
    obj_verts_t = process_obj_result(
        proc_torch_cuda(np.asarray(mesh.vertices, dtype=np.float32)),
        proc_torch_cuda(x_obj),
        "grab",
    )
    return proc_numpy(obj_verts_t), np.asarray(mesh.faces)


def build_dataset(cfg, dataset_name: str, split: str = "test"):
    """Build the MANO hand layers and the full-sequence (``task="evaluate"``)
    dataloader once, so callers can pose individual sequences on demand without
    re-instantiating everything per index (see ``pose_sample``).

    Returns ``(ds, lhand_layer, rhand_layer)``.
    """
    # This repo's x-data convention stores a full 48-D axis-angle pose (3 global
    # + 45 per-joint, one 3-vector per finger joint), not manopth's default 6-D
    # PCA coefficient pose — use_pca must be off (ncomps=45) or ManoLayer slices
    # the finger pose down to 6 dims and treats them as PCA coefficients,
    # silently collapsing per-frame finger articulation into a near-constant
    # shape.
    use_pca = cfg.get("use_pca", False)
    ncomps = 6 if use_pca else 45
    lhand_layer = ManoLayerWrapper(
        side="left", flat_hand_mean=cfg.flat_hand, mano_root=cfg.mano_root,
        use_pca=use_pca, ncomps=ncomps,
    ).cuda()
    rhand_layer = ManoLayerWrapper(
        side="right", flat_hand_mean=cfg.flat_hand, mano_root=cfg.mano_root,
        use_pca=use_pca, ncomps=ncomps,
    ).cuda()

    dataset_cls = {"grab": MotionGRAB, "arctic": MotionARCTIC}[dataset_name]
    ds = dataset_cls(
        data_path=cfg.data_path,
        data_obj_pc_path=cfg.data_obj_pc_path,
        text_json=cfg.text_json,
        max_nframes=-1,
        task="evaluate",
        split=split,
        lhand_layer=lhand_layer,
        rhand_layer=rhand_layer,
    )
    return ds, lhand_layer, rhand_layer


def pose_sample(cfg, dataset_name, ds, lhand_layer, rhand_layer, index: int = 0) -> dict:
    """Pose one full sequence from an already-built dataset, returning hand and
    object meshes already posed per frame.

    Returns a dict:
        hand_verts : (T, 778 or 1556, 3) float32, world-space MANO vertices
        hand_faces : dict {'left': (1538,3), 'right': (1538,3)} int64 — MANO's
                     left/right topologies have opposite face winding (the two
                     hands are mirrored, not just re-translated), so they are
                     kept separate rather than shared
        hand_sides : list[str], 'left'/'right' per active hand, matching the
                     778-vert blocks in hand_verts
        obj_verts  : (T, V, 3) float32, world-space object mesh vertices
        obj_faces  : (F, 3) int, static object mesh topology
        nframes    : int
        obj_name, action_name : str
    """
    item = ds[index]

    if dataset_name == "arctic":
        obj_verts, obj_faces = _pose_arctic_object(cfg, item)
    else:
        obj_verts, obj_faces = _pose_grab_object(cfg, ds, index, item)

    # Left/right MANO models are mirrored (not just re-translated): their vertex
    # templates are X-axis reflections of each other, so their face index arrays
    # use opposite winding order to keep normals pointing outward. Reusing one
    # side's faces for the other renders that hand inside-out.
    hand_faces = {
        "left": proc_numpy(lhand_layer.th_faces),
        "right": proc_numpy(rhand_layer.th_faces),
    }

    return {
        "hand_verts": item["hand_verts"],
        "hand_faces": hand_faces,
        "hand_sides": item["hand_sides"],
        "obj_verts": obj_verts,
        "obj_faces": obj_faces,
        "nframes": item["nframes"],
        "obj_name": item["obj_name"],
        "action_name": item["action_name"],
        # Per-sequence annotation intervals (ARCTIC full_motion only); empty list
        # when unavailable. Each: {start, end, hand: left|right|both, action}.
        "labels": item.get("labels", []),
        "subject": item.get("subject", ""),
        # Stable-grasp intervals per side: {"left": [[s,e],...], "right": [...]}.
        "grasps": item.get("grasps", {"left": [], "right": []}),
    }


def _fit_fixed_axis_motion(R_wrist_frame, t_wrist_frame):
    """Fit a fixed-axis / fixed-pivot model to wrist-frame object motion.

    Takes the object pose **already expressed in the wrist frame** — ``R`` (T,3,3)
    and ``t`` (T,3) — so the fit and any drawing share one frame by construction.

    Mirrors the estimator in ``scripts/dof_analysis.py``: the dominant rotation
    axis comes from the scatter of ``sin(theta) * u`` over frames, the per-frame
    signed angle from ``atan2`` of the trace-cosine and the axis-projected sine,
    and the pivot from the least-squares system ``(I - R_i) t = d_i``.

    Returns ``(R_rec, t_rec, axis, angle, resid_deg)`` where ``R_rec`` (T,3,3)
    and ``t_rec`` (T,3) are the *reconstructed* wrist-frame object pose under the
    single-axis assumption.
    """
    import torch
    from hoidata.utils.rot import vee_vector, axis_angle_to_rotmat

    R = torch.from_numpy(np.asarray(R_wrist_frame, dtype=np.float32))    # (T,3,3)
    t = torch.from_numpy(np.asarray(t_wrist_frame, dtype=np.float32))    # (T,3)

    sinaa = vee_vector((R - R.transpose(1, 2)) / 2)                      # sin(th)*u
    axis = torch.linalg.eigh(sinaa.transpose(0, 1) @ sinaa)[1][:, -1]    # (3,)
    cos_a = (torch.diagonal(R, dim1=1, dim2=2).sum(-1) - 1) / 2
    angle = torch.atan2(sinaa @ axis, cos_a)                             # (T,)
    R_rec = axis_angle_to_rotmat(angle.unsqueeze(-1) * axis)             # (T,3,3)

    # Pivot: (I - R_i) t = t_i, rank-2 (axis component unobservable) → gelsd.
    # The system is homogeneous in t — the constant is already implied by the
    # model — so t must NOT be mean-centred here; centring destroys the very
    # signal the pivot is fitted from and biases the result.
    #
    # (I - R_i) maps onto the plane perpendicular to `axis`, so a fixed-pivot
    # model can ONLY produce translation perpendicular to the axis — it cannot
    # represent any motion along the axis.  Real grasps sit at a (roughly
    # constant) offset up the axis, which the pure model drops, leaving a
    # constant along-axis displacement between t_rec and t_true.  Split t into
    # its along-axis and perpendicular parts: fit the pivot on the perpendicular
    # part (which the model can represent) and add the mean along-axis offset
    # back so the reconstruction overlays the true object.
    I = torch.eye(3, dtype=R.dtype).expand_as(R)
    t_along = t @ axis                                                   # (T,) signed
    t_perp = t - t_along.unsqueeze(-1) * axis                            # (T,3)
    A = (I - R_rec).reshape(-1, 3)
    b = t_perp.reshape(-1)
    pivot = torch.linalg.lstsq(A, b, driver='gelsd').solution            # (3,)

    # Reconstructed translation: perpendicular part from the pivot model, plus
    # the constant along-axis offset the model cannot represent.
    t_rec = (torch.einsum('tij,j->ti', I - R_rec, pivot)
             + t_along.mean() * axis)                                    # (T,3)

    delta_R = R.transpose(1, 2) @ R_rec
    cos_d = (torch.diagonal(delta_R, dim1=1, dim2=2).sum(-1) - 1) / 2
    resid_deg = torch.rad2deg(torch.acos(cos_d.clamp(-1.0, 1.0)))        # (T,)

    return (proc_numpy(R_rec), proc_numpy(t_rec), proc_numpy(axis),
            proc_numpy(angle), proc_numpy(resid_deg))


def _wrist_origin(x_hand, layer):
    """Posed wrist (root) joint position per frame, (T, 3).

    ``x_hand[:, :3]`` is MANO's global translation, which is *not* the wrist
    position — the mesh is built around the root joint.  Posing the layer and
    reading ``joints_w_tip[:, 0]`` gives the actual wrist.
    """
    import torch
    from hoidata.utils.rot import rot6d_to_axis_angle

    xd = proc_torch_cuda(np.asarray(x_hand, dtype=np.float32))
    T = xd.shape[0]
    pose = rot6d_to_axis_angle(xd[:, 3:].reshape(-1, 6)).reshape(T, 48)
    betas = torch.zeros(T, 10, device=xd.device)
    out = layer(pose, betas, xd[:, :3])
    return proc_numpy(out.joints_w_tip[:, 0]).astype(np.float32)


def pose_wrist_aligned_sample(cfg, ds, lhand_layer, rhand_layer, index: int = 0) -> dict:
    """Wrist-aligned original vs. fixed-axis-reconstructed object motion.

    Everything is drawn in the **wrist frame** of the transition hand: the hand
    is de-rotated/de-translated by its own per-frame root pose (so it sits still
    at the origin), and the object is shown twice in that same space —

      - ``obj_verts``  : the true object motion relative to the wrist;
      - extra mesh ``obj_recovered`` : the same mesh under the fitted fixed-axis /
        fixed-pivot model.

    Divergence between the two is exactly the model error reported as
    ``resid_angle`` by ``scripts/dof_analysis.py``.
    """
    from hoidata.utils.proc import transform_xdata_to_verts
    from hoidata.utils.rot import rot6d_to_rotmat
    import torch

    item = ds[index]
    # Stable-grasp items expose is_lhand/is_rhand (one grasp = one hand); the
    # transition dataset exposes hand_sides instead. Accept either.
    if "hand_sides" in item:
        side = item["hand_sides"][0]
    else:
        side = "left" if item["is_lhand"] else "right"
    layer = lhand_layer if side == "left" else rhand_layer

    x_hand, x_object = item["x_hand"], item["x_object"]
    T = int(item["n_frames"])

    # World-space hand, then map into the wrist frame: v -> R_w^T (v - o_w).
    # The origin is the posed **wrist joint**, not x_hand[:, :3] — MANO applies
    # the latter as a global offset while the mesh is built around the root
    # joint, so subtracting it would leave the hand off-origin.
    hand_verts = transform_xdata_to_verts(x_hand, layer).astype(np.float32)  # (T,778,3)
    xh = torch.from_numpy(np.asarray(x_hand, dtype=np.float32))
    wrist_R = proc_numpy(rot6d_to_rotmat(xh[:, 3:9]))                        # (T,3,3)
    wrist_o = _wrist_origin(x_hand, layer)                                   # (T,3)
    hand_local = np.einsum('tji,tvj->tvi', wrist_R, hand_verts - wrist_o[:, None])

    # Canonical object geometry (GRAB applies R^T — see process_obj_result).
    # Prefer the full .ply mesh; fall back to the sampled point cloud from
    # obj.pkl when the raw GRAB meshes aren't present (renders as points, which
    # is enough to compare the two trajectories).
    _, obj_pc, _, obj_path = ObjectModel(cfg.data_obj_pc_path)(item["obj_name"])
    mesh_path = osp.join(cfg.obj_root, obj_path)
    if osp.isfile(mesh_path):
        mesh = trimesh.load(mesh_path, process=False)
        verts0 = np.asarray(mesh.vertices, dtype=np.float32)                 # (V,3)
        obj_faces = np.asarray(mesh.faces)
    else:
        verts0 = np.asarray(obj_pc, dtype=np.float32)                        # (P,3)
        obj_faces = np.zeros((0, 3), dtype=np.int64)                         # points only

    # True wrist-frame object pose, using the same wrist origin as the hand.
    #
    # GRAB stores the object rotation un-inverted (rot6d of R_o), and the
    # rotation actually *applied* to the mesh at display time is R_o^T — see
    # process_obj_result's grab branch (einsum 'tij,ki->tkj' == verts @ R_o ==
    # applies R_o^T).  In world space a vertex lands at  R_o^T v0 + t_o.  Mapping
    # that into the wrist frame,  v_wrist = R_w^T (v_world - o_w),  gives an
    # applied rotation  R_true = R_w^T R_o^T = (R_o R_w)^T  and translation
    # t_true = R_w^T (t_o - o_w).  R_true is the object's actual orientation in
    # the wrist frame, so both the fixed-axis fit and the drawing use it
    # directly (plain multiply, no extra transpose).
    xo = torch.from_numpy(np.asarray(x_object, dtype=np.float32))
    obj_R = proc_numpy(rot6d_to_rotmat(xo[:, 3:9]))                          # R_o (stored)
    obj_t = np.asarray(x_object[:, :3], dtype=np.float32)
    R_true = np.einsum('tji,tkj->tik', wrist_R, obj_R)                       # R_w^T R_o^T
    t_true = np.einsum('tji,tj->ti', wrist_R, obj_t - wrist_o)

    R_rec, t_rec, axis, angle, resid_deg = _fit_fixed_axis_motion(R_true, t_true)

    # R_true/R_rec are already the applied wrist-frame rotation → direct multiply.
    obj_true = np.einsum('tij,kj->tki', R_true, verts0) + t_true[:, None]
    obj_rec = np.einsum('tij,kj->tki', R_rec, verts0) + t_rec[:, None]

    hand_faces = {
        "left": proc_numpy(lhand_layer.th_faces),
        "right": proc_numpy(rhand_layer.th_faces),
    }

    if hasattr(ds, "transition_label"):
        label = ds.transition_label(index)
    else:
        label = f"{item['obj_name']} {side} grasp #{index}"
    return {
        "hand_verts": hand_local.astype(np.float32),
        "hand_faces": hand_faces,
        "hand_sides": [side],
        "obj_verts": obj_true.astype(np.float32),
        "obj_faces": obj_faces,
        # Reconstructed object, same mesh/topology, drawn in the same space.
        "extra_meshes": [{
            "name": "obj_recovered",
            "verts": obj_rec.astype(np.float32),
            "faces": obj_faces,
            "color": [0.95, 0.55, 0.20],
            "opacity": 0.5,
        }],
        "nframes": T,
        "obj_name": item["obj_name"],
        "action_name": f"{label} | resid={resid_deg.mean():.1f}deg",
        "labels": [],
        "subject": "",
        "grasps": {"left": [], "right": []},
    }


def build_transition_dataset(cfg, split: str = "test", aug_p=None):
    """Build the MANO layers and a :class:`TransitionMotionGRAB` dataset.

    Mirrors :func:`build_dataset` but constructs the transition dataset (indexed
    per transition, not per sequence) with the config's transition window bounds.
    ``aug_p`` overrides the config's ``aug_p`` (the random-trim augmentation
    probability on "01"/"10" windows); pass ``0.0`` to view raw windows, or leave
    ``None`` to use the config value so the viewer shows augmented windows.
    Returns ``(ds, lhand_layer, rhand_layer)``.
    """
    use_pca = cfg.get("use_pca", False)
    ncomps = 6 if use_pca else 45
    lhand_layer = ManoLayerWrapper(
        side="left", flat_hand_mean=cfg.flat_hand, mano_root=cfg.mano_root,
        use_pca=use_pca, ncomps=ncomps,
    ).cuda()
    rhand_layer = ManoLayerWrapper(
        side="right", flat_hand_mean=cfg.flat_hand, mano_root=cfg.mano_root,
        use_pca=use_pca, ncomps=ncomps,
    ).cuda()

    if aug_p is None:
        aug_p = cfg.get("aug_p", 0.0)

    ds = TransitionMotionGRAB(
        data_path=cfg.data_path,
        data_obj_pc_path=cfg.data_obj_pc_path,
        text_json=cfg.text_json,
        max_nframes=-1,
        task="evaluate",
        split=split,
        lhand_layer=lhand_layer,
        rhand_layer=rhand_layer,
        min_len=cfg.get("transition_min_len", 10),
        max_len=cfg.get("transition_max_len", 100),
        aug_p=aug_p,
    )
    return ds, lhand_layer, rhand_layer


def build_bimanual_transition_dataset(cfg, split: str = "test", aug_p=None):
    """Build the MANO layers and a :class:`BimanualTransitionMotionGRAB` dataset.

    Like :func:`build_transition_dataset` but the two hands are merged per sequence
    into a grasp-count timeline, so windows are 0↔1↔2 count transitions
    (types ``01/02/10/12/20/21``) rather than per-hand ``01/10/11``.
    Returns ``(ds, lhand_layer, rhand_layer)``.
    """
    use_pca = cfg.get("use_pca", False)
    ncomps = 6 if use_pca else 45
    lhand_layer = ManoLayerWrapper(
        side="left", flat_hand_mean=cfg.flat_hand, mano_root=cfg.mano_root,
        use_pca=use_pca, ncomps=ncomps,
    ).cuda()
    rhand_layer = ManoLayerWrapper(
        side="right", flat_hand_mean=cfg.flat_hand, mano_root=cfg.mano_root,
        use_pca=use_pca, ncomps=ncomps,
    ).cuda()

    if aug_p is None:
        aug_p = cfg.get("aug_p", 0.0)

    ds = BimanualTransitionMotionGRAB(
        data_path=cfg.data_path,
        data_obj_pc_path=cfg.data_obj_pc_path,
        text_json=cfg.text_json,
        max_nframes=-1,
        task="evaluate",
        split=split,
        lhand_layer=lhand_layer,
        rhand_layer=rhand_layer,
        min_len=cfg.get("transition_min_len", 10),
        max_len=cfg.get("transition_max_len", 100),
        aug_p=aug_p,
    )
    return ds, lhand_layer, rhand_layer


def build_stable_grasp_dataset(cfg, dataset_name: str, split: str = "test"):
    """Build the MANO layers and a stable-grasp dataset (one item per grasp).

    Mirrors :func:`build_dataset` but constructs :class:`StableGraspGRAB` /
    :class:`StableGraspARCTIC`, indexed per stable-grasp CSV row rather than per
    sequence.  Returns ``(ds, lhand_layer, rhand_layer)``.
    """
    use_pca = cfg.get("use_pca", False)
    ncomps = 6 if use_pca else 45
    lhand_layer = ManoLayerWrapper(
        side="left", flat_hand_mean=cfg.flat_hand, mano_root=cfg.mano_root,
        use_pca=use_pca, ncomps=ncomps,
    ).cuda()
    rhand_layer = ManoLayerWrapper(
        side="right", flat_hand_mean=cfg.flat_hand, mano_root=cfg.mano_root,
        use_pca=use_pca, ncomps=ncomps,
    ).cuda()

    dataset_cls = {"grab": StableGraspGRAB, "arctic": StableGraspARCTIC}[dataset_name]
    ds = dataset_cls(
        data_path=cfg.data_path,
        data_obj_pc_path=cfg.data_obj_pc_path,
        text_json=cfg.text_json,
        max_nframes=-1,
        task="evaluate",
        split=split,
        lhand_layer=lhand_layer,
        rhand_layer=rhand_layer,
        stable_grasp_csv=cfg.get("stable_grasp_csv"),
    )
    return ds, lhand_layer, rhand_layer


def pose_transition_sample(cfg, ds, lhand_layer, rhand_layer, index: int = 0) -> dict:
    """Pose one GRAB transition window from a :class:`TransitionMotionGRAB`.

    Unlike :func:`pose_sample` (which re-poses the whole sequence's object off
    ``ds.x_obj[index]``), a transition item already carries its own sliced
    ``x_hand``/``x_object`` windows, so both hand and object are posed directly
    from the item.  Returns the same dict shape the viewer expects (note the
    viewer's own key is ``nframes``, while the dataset item uses ``n_frames``).
    """
    from hoidata.utils.proc import transform_xdata_to_verts

    item = ds[index]
    side = item["hand_sides"][0]
    layer = lhand_layer if side == "left" else rhand_layer

    # Hand: raw x-data window → world-space MANO vertices for the single hand.
    hand_verts = transform_xdata_to_verts(item["x_hand"], layer).astype(np.float32)  # (Tt,778,3)

    # Object: rigid GRAB mesh posed per frame from the sliced x_object window.
    _, _, _, obj_path = ObjectModel(cfg.data_obj_pc_path)(item["obj_name"])
    mesh = trimesh.load(osp.join(cfg.obj_root, obj_path), process=False)
    obj_verts_t = process_obj_result(
        proc_torch_cuda(np.asarray(mesh.vertices, dtype=np.float32)),
        proc_torch_cuda(item["x_object"]),
        "grab",
    )
    obj_verts = proc_numpy(obj_verts_t)
    obj_faces = np.asarray(mesh.faces)

    hand_faces = {
        "left": proc_numpy(lhand_layer.th_faces),
        "right": proc_numpy(rhand_layer.th_faces),
    }

    return {
        "hand_verts": hand_verts,
        "hand_faces": hand_faces,
        "hand_sides": item["hand_sides"],
        "obj_verts": obj_verts,
        "obj_faces": obj_faces,
        "nframes": item["n_frames"],
        "obj_name": item["obj_name"],
        # Label carries object / hand / transition-type (see transition_label).
        "action_name": ds.transition_label(index),
        "labels": [],
        "subject": "",
        "grasps": {"left": [], "right": []},
    }


def pose_bimanual_transition_sample(
        cfg, ds, lhand_layer, rhand_layer, index: int = 0) -> dict:
    """Pose one bimanual transition window from a :class:`BimanualTransitionMotionGRAB`.

    The item exposes each posed hand separately as ``x_lhand`` / ``x_rhand``
    (each ``(Tt, D)`` or ``None``); ``item["hand_sides"]`` lists which are present.
    Each hand is posed independently and the world-space vertex blocks are
    concatenated along the vertex axis into ``(Tt, 778*H, 3)`` — the layout the
    viewer's ``build_hand_mesh`` expects (it slices 778-vertex blocks per side).
    The object is posed directly from the item's sliced ``x_object`` window,
    exactly like the unimanual poser.
    """
    from hoidata.utils.proc import transform_xdata_to_verts

    item = ds[index]
    sides = list(item["hand_sides"])

    hand_blocks = []
    for side in sides:
        layer = lhand_layer if side == "left" else rhand_layer
        xh = np.asarray(item["x_lhand" if side == "left" else "x_rhand"],
                        dtype=np.float32)                                  # (Tt, D)
        verts = transform_xdata_to_verts(xh, layer).astype(np.float32)     # (Tt,778,3)
        hand_blocks.append(verts)
    hand_verts = np.concatenate(hand_blocks, axis=1)  # (Tt, 778*H, 3)

    # Object: rigid GRAB mesh posed per frame from the sliced x_object window.
    obj_model = ObjectModel(cfg.data_obj_pc_path)
    _, obj_pc, _obj_pc_normal, obj_path = obj_model(item["obj_name"])
    mesh = trimesh.load(osp.join(cfg.obj_root, obj_path), process=False)
    mesh_verts_canon = np.asarray(mesh.vertices, dtype=np.float32)          # (Vo,3)
    obj_verts_t = process_obj_result(
        proc_torch_cuda(mesh_verts_canon),
        proc_torch_cuda(item["x_object"]),
        "grab",
    )
    obj_verts = proc_numpy(obj_verts_t)
    obj_faces = np.asarray(mesh.faces)

    # Per-frame, per-hand contact map over the window, colored on the full-res
    # mesh: each mesh vertex takes the contact of its nearest sampled point.  The
    # dataset returns per-hand maps (Tt, 2, O) = [left, right]; both are kept so
    # the viewer can color the two hands' contact distinctly (and pick the larger
    # where a vertex touches both).  The canonical-space nearest-sampled-point map
    # is frame-invariant (mesh and pc move rigidly together), so compute it once.
    contact_lr = ds._window_contact(
        item["seq_idx"], item["start_frame"], item["end_frame"], sides)     # (Tt,2,O)
    from scipy.spatial import cKDTree
    nn_idx = cKDTree(obj_pc.astype(np.float32)).query(mesh_verts_canon)[1]   # (Vo,)
    obj_contact = contact_lr[:, :, nn_idx].astype(np.float32)               # (Tt,2,Vo)

    hand_faces = {
        "left": proc_numpy(lhand_layer.th_faces),
        "right": proc_numpy(rhand_layer.th_faces),
    }

    return {
        "hand_verts": hand_verts,
        "hand_faces": hand_faces,
        "hand_sides": sides,
        "obj_verts": obj_verts,
        "obj_faces": obj_faces,
        "obj_contact": obj_contact,          # (Tt, 2, Vo) per-frame per-hand [L,R] contact
        "nframes": item["n_frames"],
        "obj_name": item["obj_name"],
        # Label carries object / driving hand(s) / transition-type.
        "action_name": ds.transition_label(index),
        "labels": [],
        "subject": "",
        "grasps": {"left": [], "right": []},
    }


def load_sample(cfg, dataset_name: str, split: str = "test", index: int = 0) -> dict:
    """Convenience wrapper: build the dataset and pose one sequence in one call.

    Equivalent to ``build_dataset(...)`` followed by ``pose_sample(...)``.
    Kept for the standalone o3d CLI and existing callers; a long-lived server
    should instead call ``build_dataset`` once and ``pose_sample`` per index.
    """
    ds, lhand_layer, rhand_layer = build_dataset(cfg, dataset_name, split)
    return pose_sample(cfg, dataset_name, ds, lhand_layer, rhand_layer, index)


def build_hand_mesh(hand_verts_frame, hand_sides, hand_faces) -> o3d.geometry.TriangleMesh:
    """One frame of hand_verts (778|1556, 3), sliced per active side and summed
    into a single mesh so it stays correct for one or two active hands.

    ``hand_faces`` is a dict {'left': ..., 'right': ...} — each side's own face
    winding must be used, since MANO's left/right topologies are mirrored.
    """
    combined = o3d.geometry.TriangleMesh()
    for i, side in enumerate(hand_sides):
        v = hand_verts_frame[i * 778 : (i + 1) * 778]
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(hand_faces[side])
        )
        mesh.paint_uniform_color(HAND_SIDE_COLORS[side])
        combined += mesh
    combined.compute_vertex_normals()
    return combined


def build_object_mesh(obj_verts_frame, obj_faces) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(obj_verts_frame), o3d.utility.Vector3iVector(obj_faces)
    )
    mesh.paint_uniform_color(OBJECT_COLOR)
    mesh.compute_vertex_normals()
    return mesh


class HOIVisualizer:
    """Loops a hand-object interaction sequence in an Open3D window (free orbit
    camera via mouse drag), and can export the same sequence to a video file."""

    def __init__(self, sample: dict, fps: int = 30):
        self.sample = sample
        self.fps = fps
        self.nframes = sample["nframes"]
        self._frame_idx = 0
        self._last_tick = 0.0

        self.hand_mesh = build_hand_mesh(
            sample["hand_verts"][0], sample["hand_sides"], sample["hand_faces"]
        )
        self.obj_mesh = build_object_mesh(sample["obj_verts"][0], sample["obj_faces"])

    def _set_frame(self, mesh_hand: o3d.geometry.TriangleMesh, mesh_obj: o3d.geometry.TriangleMesh, frame_idx: int):
        hv = self.sample["hand_verts"][frame_idx]
        ov = self.sample["obj_verts"][frame_idx]

        combined_hv = np.concatenate(
            [hv[i * 778 : (i + 1) * 778] for i in range(len(self.sample["hand_sides"]))], axis=0
        )
        mesh_hand.vertices = o3d.utility.Vector3dVector(combined_hv)
        mesh_obj.vertices = o3d.utility.Vector3dVector(ov)
        mesh_hand.compute_vertex_normals()
        mesh_obj.compute_vertex_normals()

    def _advance_frame(self):
        self._frame_idx = (self._frame_idx + 1) % self.nframes
        self._set_frame(self.hand_mesh, self.obj_mesh, self._frame_idx)

    def _animation_callback(self, vis):
        now = time.time()
        if now - self._last_tick < 1.0 / self.fps:
            return False
        self._last_tick = now
        self._advance_frame()
        vis.update_geometry(self.hand_mesh)
        vis.update_geometry(self.obj_mesh)
        return False

    def run_interactive(self):
        """Blocking. Opens a window, loops the sequence, and lets the user
        freely orbit the camera by dragging with the mouse. Requires an active
        display (a monitor with a running X/Wayland session)."""
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"HOI Visualizer — {self.sample['action_name']}")
        vis.add_geometry(self.hand_mesh)
        vis.add_geometry(self.obj_mesh)
        self._last_tick = time.time()
        vis.register_animation_callback(self._animation_callback)
        vis.run()
        vis.destroy_window()

    def render_video(self, out_path: str, fps: int = 30, orbit: bool = True,
                      width: int = 960, height: int = 720):
        """Deterministic, headless-safe export: renders every frame once via
        OffscreenRenderer (independent of wall-clock time and of any live
        interactive window) and writes them to a video file."""
        import imageio

        renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
        renderer.scene.set_background([1, 1, 1, 1])
        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultLit"

        hand_mesh = build_hand_mesh(
            self.sample["hand_verts"][0], self.sample["hand_sides"], self.sample["hand_faces"]
        )
        obj_mesh = build_object_mesh(self.sample["obj_verts"][0], self.sample["obj_faces"])
        renderer.scene.add_geometry("hand", hand_mesh, material)
        renderer.scene.add_geometry("obj", obj_mesh, material)

        # Frame the camera to the full scene (object + hand trajectories across
        # all frames), not just the object's initial pose — hands can start far
        # from the object before a grasp.
        all_pts = np.concatenate(
            [self.sample["obj_verts"].reshape(-1, 3), self.sample["hand_verts"].reshape(-1, 3)],
            axis=0,
        )
        bbox_min, bbox_max = all_pts.min(axis=0), all_pts.max(axis=0)
        center = (bbox_min + bbox_max) / 2.0
        radius = max(0.3, float(np.linalg.norm(bbox_max - bbox_min)))

        writer = imageio.get_writer(out_path, fps=fps)
        try:
            for t in range(self.nframes):
                self._set_frame(hand_mesh, obj_mesh, t)
                renderer.scene.remove_geometry("hand")
                renderer.scene.remove_geometry("obj")
                renderer.scene.add_geometry("hand", hand_mesh, material)
                renderer.scene.add_geometry("obj", obj_mesh, material)

                angle = 2 * np.pi * t / max(self.nframes, 1) if orbit else 0.0
                eye = center + radius * np.array([np.sin(angle), 0.4, np.cos(angle)])
                renderer.scene.camera.look_at(center, eye, [0, 1, 0])

                img = renderer.render_to_image()
                writer.append_data(np.asarray(img))
        finally:
            writer.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize a hand-object interaction sequence.")
    parser.add_argument("--config", required=True, help="Path to a dataset config yaml, e.g. configs/arctic.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--save", default=None, help="Optional output video path, e.g. out.mp4")
    parser.add_argument("--no-interactive", action="store_true", help="Skip the interactive window")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    sample = load_sample(cfg, cfg.name, split=args.split, index=args.index)
    viz = HOIVisualizer(sample, fps=args.fps)

    if args.save:
        viz.render_video(args.save, fps=args.fps)
    if not args.no_interactive:
        viz.run_interactive()


if __name__ == "__main__":
    main()
