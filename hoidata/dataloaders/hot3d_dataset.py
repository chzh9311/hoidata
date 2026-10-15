"""HOT3D loaders in this repo's ``x_hand`` / ``x_obj`` format.

Two datasets:

* ``MotionHOT3DMultiObj`` (raw loader, no preprocessing): every HOT3D recording
  is cut into fixed-length clips, and each clip returns the two hand motions
  (``x_lhand`` / ``x_rhand``) and the trajectories of **all** tracked dynamic
  objects (``x_obj``) in the same trans + 6D-rotation layout used by the GRAB /
  ARCTIC / GigaHands loaders, plus per-frame tracking masks, the MANO shape
  (beta) and the recording metadata.  Used by the web viewer.
* ``MotionHOT3D`` (+ ``MotionHOT3DDataModule``): short **single-object** motion
  segments with ``MotionGRAB``'s interface (``MotionDatasetBase``), read from
  the ``data.npz`` built by ``hoidata/preprocess/preprocess_hot3d.py``; the
  segmentation is ``segment_single_object_motions`` below.  No text.

Only the ground-truth pose files are read (no VRS / images); the parsing
mirrors the official toolkit (``hot3d/data_loaders``: ``loader_hand_poses``,
``ObjectPose3dProvider``, ``mano_layer``) without its projectaria_tools / smplx
dependencies.

* ``x_lhand`` / ``x_rhand`` : (T, 99) = trans(3) + rot6d(16*6).  HOT3D stores
  MANO hands as ``wrist_xform`` (``t_xyz`` + ``q_wxyz``) and 15 **PCA**
  coefficients, posed by smplx with ``flat_hand_mean=False``.  The full 45-D
  finger pose is expanded here as ``pca @ hands_components[:15] + hands_mean``,
  so the x-data holds *absolute* joint rotations — pose it with a
  ``flat_hand_mean=True`` layer.  smplx and manopth both rotate the hand about
  the root joint and add the translation afterwards, so ``t_xyz`` is used as the
  translation directly (no pivot correction, unlike GigaHands).  Posing needs
  the recording's ``beta`` and a left layer with the MANO_LEFT shapedirs fix
  (``hoidata.utils.hand_proc.fix_mano_left_shapedirs``).
* ``x_obj`` : (T, K, 9) = trans(3) + rot6d(6) per object, ``R`` / ``t`` of
  ``T_world_object`` — a mesh vertex maps to the world as ``R @ v + t``.

Timeline: the object-pose timestamps of ``dynamic_objects.csv``.  Aria
recordings log poses for two camera streams ~0.1 ms apart; timestamps closer
than ``merge_tol_ms`` are merged into one frame (≈30 Hz for both headsets).
Hand poses are matched to the nearest frame within the same tolerance.

Missing poses: frames where a hand / object has no tracked pose are **not**
interpolated.  They are flagged ``False`` in the ``lhand_valid`` /
``rhand_valid`` / ``obj_valid`` masks and hold a placeholder x-data row — zero
translation and identity rotations (``PLACEHOLDER_HAND_XDATA`` /
``PLACEHOLDER_OBJ_XDATA``) — so the arrays stay finite and valid rotations;
consumers must use the masks to ignore those frames.

World frame: raw Aria recordings are **Z-up**, raw Quest3 recordings are
**Y-up** (the official viewer's ``RIGHT_HAND_Z_UP`` / ``RIGHT_HAND_Y_UP``).
With ``canonical_z_up`` (default) Quest3 recordings are rotated +90° about x
(``Y_UP_TO_Z_UP``: (x, y, z) → (x, −z, y)) so every recording is +Z up like
GRAB; hands and objects are transformed rigidly (see ``_hand_xdata`` for the
MANO root-joint correction).  Heights are not shifted: the Quest3 floor stays
at z ≈ 0 while the Aria origin stays near the headset.  Each item reports its
world up axis as ``up_axis``.

Expected dataset root (``data/hot3d_raw`` by default)::

    <root>/
      assets/instance.json, assets/<object_uid>.glb
      <P00xx>_<hash>/metadata.json
      <P00xx>_<hash>/dynamic_objects.csv
      <P00xx>_<hash>/mano_hand_pose_trajectory.jsonl
"""

import os
import json
import glob
from collections import OrderedDict
from os.path import join as pjoin

import numpy as np
import pandas as pd
import torch
from torch.utils import data
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_axis_angle

from hoidata.utils.proc import (
    transform_hand_to_xdata, transform_obj_to_xdata, pc_normalize, proc_numpy,
    proc_torch_cuda)
from .action_dataset import MotionDatasetBase, MotionDataModuleBase


#: HOT3D hand index -> side (``loader_hand_poses.LEFT_HAND_INDEX`` = 0).
_HAND_KEYS = {"left": "0", "right": "1"}
#: Headset -> raw world up axis (matches the official Hot3DVisualizer).
_UP_AXIS = {"Aria": "z", "Quest3": "y"}
#: World rotation taking a +Y-up recording to +Z-up: +90° about x,
#: (x, y, z) -> (x, -z, y).  Applied to Quest3 recordings by ``canonical_z_up``.
Y_UP_TO_Z_UP = np.array([[1.0, 0.0, 0.0],
                         [0.0, 0.0, -1.0],
                         [0.0, 1.0, 0.0]])
#: Number of MANO PCA pose coefficients HOT3D stores per hand.
_N_PCA = 15

#: pytorch3d rot6d of the identity rotation (first two rows of I).
_ROT6D_IDENTITY = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
#: x-data written on frames without a tracked pose (see ``*_valid`` masks):
#: zero translation + identity rotations.
PLACEHOLDER_HAND_XDATA = np.concatenate(
    [np.zeros(3, np.float32), np.tile(_ROT6D_IDENTITY, 16)])      # (99,)
PLACEHOLDER_OBJ_XDATA = np.concatenate(
    [np.zeros(3, np.float32), _ROT6D_IDENTITY])                   # (9,)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
_MANO_PCA_CACHE = {}


def _mano_pca(mano_root, side):
    """``(hands_components (45, 45), hands_mean (45,))`` from the MANO pkl."""
    key = (mano_root, side)
    if key not in _MANO_PCA_CACHE:
        from manopth.manolayer import ManoLayer
        layer = ManoLayer(side=side, mano_root=mano_root, use_pca=False,
                          ncomps=45, flat_hand_mean=False)
        comps = np.asarray(layer.smpl_data["hands_components"], np.float32)
        mean = np.asarray(layer.smpl_data["hands_mean"], np.float32)
        _MANO_PCA_CACHE[key] = (comps, mean)
    return _MANO_PCA_CACHE[key]


_MANO_J0_CACHE = {}


def _mano_root_joint(mano_root, side, beta):
    """MANO root joint ``J0`` (3,) in metres for shape ``beta``.

    Uses the HOT3D-corrected MANO_LEFT shape blend shapes
    (``fix_mano_left_shapedirs``).  ``J0`` is the pivot manopth rotates the
    global orientation about, so it is needed to rotate a posed hand rigidly.
    """
    beta = np.asarray(beta, dtype=np.float32).reshape(-1)[:10]
    key = (mano_root, side, tuple(np.round(beta, 6)))
    if key not in _MANO_J0_CACHE:
        from hoidata.utils.hand_proc import ManoLayerWrapper, fix_mano_left_shapedirs
        kw = dict(mano_root=mano_root, use_pca=False, ncomps=45, flat_hand_mean=True)
        layer = ManoLayerWrapper(side=side, **kw)
        if side == "left":
            fix_mano_left_shapedirs(layer, ManoLayerWrapper(side="right", **kw))
        with torch.no_grad():
            out = layer(torch.zeros(1, 48), torch.from_numpy(beta)[None], torch.zeros(1, 3))
        _MANO_J0_CACHE[key] = out.joints_w_tip[0, 0].numpy().astype(np.float64)
    return _MANO_J0_CACHE[key]


def _merge_timestamps(ts, tol_ns):
    """Collapse sorted unique timestamps closer than ``tol_ns`` into frames.

    Returns ``(frame_ts, frame_of)`` — the first timestamp of every frame, and
    for each input timestamp the index of the frame it belongs to.
    """
    ts = np.asarray(ts, dtype=np.int64)
    if len(ts) == 0:
        return ts, np.zeros(0, dtype=np.int64)
    new_frame = np.ones(len(ts), dtype=bool)
    new_frame[1:] = np.diff(ts) > tol_ns
    frame_of = np.cumsum(new_frame) - 1
    return ts[new_frame], frame_of


def _match_frames(frame_ts, ts, tol_ns):
    """Frame index of each timestamp in ``ts`` (nearest earlier frame within
    ``tol_ns``), or -1 when no frame is close enough."""
    idx = np.searchsorted(frame_ts, ts, side="right") - 1
    ok = idx >= 0
    ok[ok] &= (ts[ok] - frame_ts[idx[ok]]) <= tol_ns
    return np.where(ok, idx, -1)


def _quat_to_rotmat(quat_wxyz):
    """(..., 4) wxyz quaternion -> (..., 3, 3) rotation matrix (float32 torch)."""
    q = torch.as_tensor(np.asarray(quat_wxyz), dtype=torch.float64)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return quaternion_to_matrix(q).float()


# --------------------------------------------------------------------------- #
# Single-object motion segmentation
# --------------------------------------------------------------------------- #
def _runs(labels):
    """Run-length encode a 1-D label array → list of ``(label, start, end)``."""
    labels = np.asarray(labels)
    if len(labels) == 0:
        return []
    cut = np.nonzero(labels[1:] != labels[:-1])[0] + 1
    starts = np.concatenate([[0], cut])
    ends = np.concatenate([cut, [len(labels)]])
    return [(labels[s], int(s), int(e)) for s, e in zip(starts, ends)]


def object_moving_mask(x_obj, valid, window=5, trans_thr=0.005, rot_thr=0.05,
                       merge_gap=10):
    """Per-frame "is moving" mask of one object's trajectory.

    Slides a ``window``-frame window over the object's **tracked** frames only
    (untracked frames are dropped and the remaining frames concatenated, so a
    tracking gap neither reads as motion nor hides it).  A window is *moving*
    when the max − min of any translation dim exceeds ``trans_thr`` (metres) or
    of any rot6d dim exceeds ``rot_thr``; every original frame it spans
    (including untracked frames in between) is then marked moving.  Finally,
    moving runs separated by fewer than ``merge_gap`` frames are joined into one
    long moving interval.

    Args:
        x_obj: (N, 9) trans + rot6d (untracked rows may hold placeholders).
        valid: (N,) bool, the pose is tracked on that frame.

    Returns:
        (N,) bool.
    """
    N = len(valid)
    moving = np.zeros(N, dtype=bool)
    idx = np.nonzero(valid)[0]
    if len(idx) < window:
        return moving
    w = np.lib.stride_tricks.sliding_window_view(
        np.asarray(x_obj)[idx], window, axis=0)                 # (n-w+1, 9, w)
    rng = w.max(-1) - w.min(-1)                                 # (n-w+1, 9)
    win_moving = (rng[:, :3] > trans_thr).any(1) | (rng[:, 3:] > rot_thr).any(1)
    for j in np.nonzero(win_moving)[0]:
        moving[idx[j]:idx[j + window - 1] + 1] = True

    # Join moving runs whose static gap is shorter than merge_gap frames.
    return merge_short_gaps(moving, merge_gap)


def merge_short_gaps(mask, max_gap):
    """Fill interior ``False`` runs shorter than ``max_gap`` frames in a 1-D mask
    (joins active runs separated by a short gap).  Returns a new array."""
    mask = np.asarray(mask, dtype=bool).copy()
    runs = _runs(mask)
    for k, (lab, s, e) in enumerate(runs):
        if not lab and 0 < k < len(runs) - 1 and e - s < max_gap:
            mask[s:e] = True
    return mask


def drop_short_runs(mask, min_len):
    """Clear ``True`` runs shorter than ``min_len`` frames in a 1-D mask (e.g.
    a hand brushing a neighbouring object for a frame or two).  Returns a new
    array."""
    mask = np.asarray(mask, dtype=bool).copy()
    for lab, s, e in _runs(mask):
        if lab and e - s < min_len:
            mask[s:e] = False
    return mask


def _hand_verts_torch(x_hand, layer, chunk=2048):
    """(N, 99) x-data → (N, 778, 3) MANO vertices on the layer's device (zero
    betas — pass a ``shaped_mano_layer`` for a subject's hand shape)."""
    from pytorch3d.transforms import rotation_6d_to_matrix
    device = layer.th_faces.device
    out = []
    for a in range(0, len(x_hand), chunk):
        xd = torch.as_tensor(np.asarray(x_hand[a:a + chunk], np.float32), device=device)
        T = xd.shape[0]
        pose = matrix_to_axis_angle(
            rotation_6d_to_matrix(xd[:, 3:].reshape(-1, 6))).reshape(T, 48)
        out.append(layer(pose, torch.zeros(T, 10, device=device), xd[:, :3]).vertices)
    return torch.cat(out) if out else torch.zeros(0, 778, 3, device=device)


@torch.no_grad()
def hand_object_contact(x_lhand, x_rhand, lhand_valid, rhand_valid, x_obj,
                        obj_valid, obj_points, lhand_layer, rhand_layer,
                        thr=0.005, chunk=256):
    """Per-frame hand–object contact for every (hand, object) pair.

    A hand is in contact with an object on a frame when the nearest distance
    between the hand's MANO vertices and the object's surface (``obj_points``,
    dense samples in the object frame) is below ``thr`` (metres).  Hand
    vertices are moved into each object's frame (``R^T (p - t)``, as the object
    is posed ``R v + t``); frames where no hand vertex falls inside the
    object's bounding box grown by ``thr`` are skipped before the exact
    nearest-neighbour test.  Frames where the hand or the object is untracked
    are never in contact.

    Args:
        x_lhand / x_rhand: (N, 99) hand x-data; *_valid: (N,) bool.
        x_obj: (N, K, 9) object x-data; obj_valid: (N, K) bool.
        obj_points: list of K (P_k, 3) arrays, object-frame surface samples.
        lhand_layer / rhand_layer: MANO layers with the subject's shape baked
            in (posed with zero betas), e.g. via ``shaped_mano_layer``.

    Returns:
        (contact (N, K, 2) bool [left, right], dist (N, K, 2) float32 nearest
        distance in metres, ``inf`` where skipped / untracked).
    """
    from pytorch3d.ops import knn_points
    from pytorch3d.transforms import rotation_6d_to_matrix

    N, K = np.asarray(obj_valid).shape
    device = lhand_layer.th_faces.device
    dist = torch.full((N, K, 2), float("inf"), device=device)
    hands = [(x_lhand, lhand_valid, lhand_layer), (x_rhand, rhand_valid, rhand_layer)]
    for h, (x_hand, h_valid, layer) in enumerate(hands):
        if not np.any(h_valid):
            continue
        hv = _hand_verts_torch(x_hand, layer)                         # (N, 778, 3)
        for k in range(K):
            pts = torch.as_tensor(np.asarray(obj_points[k], np.float32), device=device)
            lo, hi = pts.min(0).values - thr, pts.max(0).values + thr
            xo = torch.as_tensor(np.asarray(x_obj[:, k], np.float32), device=device)
            R = rotation_6d_to_matrix(xo[:, 3:])                       # (N, 3, 3)
            both = torch.as_tensor(np.asarray(h_valid) & np.asarray(obj_valid[:, k]),
                                   device=device)
            for a in range(0, N, chunk):
                sl = slice(a, min(a + chunk, N))
                local = torch.einsum("nji,nvj->nvi", R[sl], hv[sl] - xo[sl, None, :3])
                near = ((local >= lo) & (local <= hi)).all(-1).any(-1) & both[sl]
                idx = torch.nonzero(near).squeeze(1)
                if len(idx) == 0:
                    continue
                q = local[idx]                                         # (B, 778, 3)
                d2 = knn_points(q, pts[None].expand(len(idx), -1, -1), K=1).dists
                dist[a + idx, k, h] = d2.squeeze(-1).min(-1).values.sqrt()
    dist = dist.cpu().numpy().astype(np.float32)
    return dist < thr, dist


def segment_by_activity(active, merge_same_obj=True, same_obj_max_gap=None,
                        max_transition_len=None):
    """Split a recording into single-object segments from per-object activity.

    ``active`` (N, K) bool marks the frames where object ``k`` is being
    interacted with (``hand_object_contact`` → in contact with a hand, or
    ``object_moving_mask`` → moving), short gaps already joined and very short
    runs already removed (``merge_short_gaps`` / ``drop_short_runs``).

    1. Every frame is labelled *static* (no object active), *solo k* (only
       object ``k`` active) or *multi* (>1 objects active).  Multi frames are
       dropped: an object's interaction ends when a second object becomes
       active and restarts once every other object is inactive again.
    2. With ``merge_same_obj``, two solo-``k`` runs of the **same** object
       separated only by a static run are merged into one core, static run
       included (e.g. a steady hold read as static).  ``same_obj_max_gap``
       (frames) optionally limits which static runs are absorbed; ``None``
       absorbs any length.  A multi run or another object's run in between
       always keeps them apart.
    3. Each solo-``k`` run is extended over the static run(s) directly before
       and after it (the transition frames — approach before / release after);
       it never extends across a multi or another object's solo run.  So two
       consecutive segments may share the static frames between them.
       ``max_transition_len`` optionally caps each extension (frames).

    Returns:
        list of dicts ``{"obj": k, "start", "end", "move_start", "move_end"}``
        — the segment ``[start, end)`` and its active core
        ``[move_start, move_end)`` (frame indices of the recording; the
        ``move_*`` names are kept from the motion-based segmentation).
    """
    active = np.asarray(active, dtype=bool)
    count = active.sum(1)
    STATIC, MULTI = -1, -2
    labels = np.where(count == 0, STATIC,
                      np.where(count > 1, MULTI, active.argmax(1)))

    if merge_same_obj:
        runs = _runs(labels)
        for i in range(1, len(runs) - 1):
            lab, s, e = runs[i]
            if (lab == STATIC and runs[i - 1][0] >= 0
                    and runs[i - 1][0] == runs[i + 1][0]
                    and (same_obj_max_gap is None or e - s <= same_obj_max_gap)):
                labels[s:e] = runs[i - 1][0]

    runs = _runs(labels)
    segments = []
    for i, (lab, s, e) in enumerate(runs):
        if lab < 0:
            continue
        start, end = s, e
        if i > 0 and runs[i - 1][0] == STATIC:
            start = runs[i - 1][1]
            if max_transition_len is not None:
                start = max(start, s - int(max_transition_len))
        if i + 1 < len(runs) and runs[i + 1][0] == STATIC:
            end = runs[i + 1][2]
            if max_transition_len is not None:
                end = min(end, e + int(max_transition_len))
        segments.append({"obj": int(lab), "start": start, "end": end,
                         "move_start": s, "move_end": e})
    return segments


def segment_single_object_motions(x_obj, obj_valid, window=5, trans_thr=0.005,
                                  rot_thr=0.05, merge_gap=10,
                                  max_transition_len=None, merge_same_obj=True,
                                  same_obj_max_gap=None):
    """Motion-based segmentation: per-object moving masks
    (``object_moving_mask``) fed to ``segment_by_activity``.

    Args:
        x_obj: (N, K, 9) trans + rot6d of the K objects.
        obj_valid: (N, K) bool tracking mask.
    """
    N, K = np.asarray(obj_valid).shape
    moving = np.stack([
        object_moving_mask(x_obj[:, k], obj_valid[:, k], window, trans_thr,
                           rot_thr, merge_gap)
        for k in range(K)], axis=1) if K else np.zeros((N, 0), bool)
    return segment_by_activity(moving, merge_same_obj=merge_same_obj,
                               same_obj_max_gap=same_obj_max_gap,
                               max_transition_len=max_transition_len)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class MotionHOT3DMultiObj(data.Dataset):
    """HOT3D clips in this repo's ``x_hand`` / ``x_obj`` format.

    Args:
        dataset_root: folder with ``assets/`` and the ``P*`` recording folders
            (defaults to ``data/hot3d_raw``).
        split: ``'train' | 'val' | 'test' | 'all'``.  Split by participant:
            ``val_participants`` / ``test_participants`` are held out, every
            other participant is train.
        clip_len: frames per clip; ``-1`` = one clip per whole recording.
        clip_stride: frame step between clip starts (defaults to ``clip_len``,
            i.e. non-overlapping clips).
        min_clip_len: drop a trailing clip shorter than this many frames.
        headsets: keep only recordings from these headsets (``"Aria"`` /
            ``"Quest3"``); ``None`` keeps both.
        merge_tol_ms: timestamps closer than this are one frame (Aria logs two
            streams ~0.1 ms apart); also the hand-to-frame matching tolerance.
        mano_root: MANO model folder, for the PCA basis / mean pose (and the
            root joint used by ``canonical_z_up``).
        cache_size: number of parsed recordings kept in memory (consecutive
            clips share a recording).
        canonical_z_up: rotate Quest3 (+Y-up) recordings to +Z-up
            (``Y_UP_TO_Z_UP``) so every recording shares GRAB's up axis.

    Each item is a dict with:
        ``seq_name``  : ``"<recording>:<start>-<end>"`` (unique per clip).
        ``recording`` / ``participant`` / ``headset`` : str.
        ``up_axis``   : world up axis — always ``"z"`` with ``canonical_z_up``,
            else ``"z"`` (Aria) or ``"y"`` (Quest3).
        ``text``      : ``""`` (HOT3D has no captions).
        ``nframes``, ``start_frame``, ``end_frame`` : int (frames of the recording).
        ``timestamps_ns`` : (T,) int64 frame timestamps.
        ``is_lhand`` / ``is_rhand`` : bool, the hand has a pose on any clip frame.
        ``lhand_valid`` / ``rhand_valid`` : (T,) bool, the hand's pose is tracked
            on that frame.
        ``x_lhand`` / ``x_rhand`` : (T, 99) float32 trans + rot6d; untracked
            frames hold ``PLACEHOLDER_HAND_XDATA``.
        ``beta`` (and aliases ``lhand_beta`` / ``rhand_beta``) : (10,) float32.
        ``x_obj``     : (T, K, 9) float32 trans + rot6d for the K objects;
            untracked frames hold ``PLACEHOLDER_OBJ_XDATA``.
        ``obj_valid`` : (T, K) bool, the object's pose is tracked on that frame.
        ``obj_uids`` / ``obj_names`` / ``obj_meshes`` : K-lists (uid, name, .glb path).
    """

    def __init__(
        self,
        dataset_root="data/hot3d_raw",
        split="all",
        clip_len=300,
        clip_stride=None,
        min_clip_len=30,
        headsets=None,
        val_participants=("P0017",),
        test_participants=("P0021",),
        merge_tol_ms=5.0,
        mano_root="/home/zxc417/data/mano_v1_2/models",
        cache_size=4,
        canonical_z_up=True,
    ):
        if split not in ("train", "val", "test", "all"):
            raise ValueError(f"split must be train/val/test/all, got {split!r}")
        self.dataset_root = dataset_root
        self.split = split
        self.clip_len = int(clip_len)
        self.clip_stride = int(clip_stride) if clip_stride else None
        self.min_clip_len = int(min_clip_len)
        self.headsets = set(headsets) if headsets else None
        self.val_participants = set(val_participants or ())
        self.test_participants = set(test_participants or ())
        self.tol_ns = int(round(float(merge_tol_ms) * 1e6))
        self.mano_root = mano_root
        self.cache_size = int(cache_size)
        self.canonical_z_up = bool(canonical_z_up)
        #: recording name -> parsed arrays (LRU, see ``_recording``).
        self._cache = OrderedDict()

        self.asset_root = pjoin(dataset_root, "assets")
        with open(pjoin(self.asset_root, "instance.json")) as f:
            instances = json.load(f)
        self.uid_to_name = {k: v["instance_name"] for k, v in instances.items()}

        # ---- recordings in this split ----------------------------------- #
        self.recordings = []   # list of metadata dicts (+ "nframes")
        for meta_path in sorted(glob.glob(pjoin(dataset_root, "P*", "metadata.json"))):
            with open(meta_path) as f:
                meta = json.load(f)
            if not meta.get("have_hand_object_pose_gt", False):
                continue
            if self.headsets is not None and meta["headset"] not in self.headsets:
                continue
            if self._split_of(meta["participant_id"]) != split and split != "all":
                continue
            rec_dir = os.path.dirname(meta_path)
            if not (os.path.exists(pjoin(rec_dir, "dynamic_objects.csv"))
                    and os.path.exists(pjoin(rec_dir, "mano_hand_pose_trajectory.jsonl"))):
                continue
            ts = pd.read_csv(pjoin(rec_dir, "dynamic_objects.csv"),
                             usecols=["timestamp[ns]"])["timestamp[ns]"].to_numpy()
            frame_ts, _ = _merge_timestamps(np.unique(ts), self.tol_ns)
            meta = dict(meta, rec_dir=rec_dir, nframes=int(len(frame_ts)))
            self.recordings.append(meta)

        # ---- clip index: (recording idx, start, end) -------------------- #
        self.clips = []
        for ri, rec in enumerate(self.recordings):
            for s, e in self._windows(rec["nframes"]):
                self.clips.append((ri, s, e))

        hs = "/" + "+".join(sorted(self.headsets)) if self.headsets else ""
        print(f"MotionHOT3DMultiObj [{split}{hs}] {len(self.recordings)} recordings, "
              f"{len(self.clips)} clips (clip_len={self.clip_len})")

    # ---- index construction ------------------------------------------------ #
    def _split_of(self, participant):
        if participant in self.test_participants:
            return "test"
        if participant in self.val_participants:
            return "val"
        return "train"

    def _world_rot(self, meta):
        """(3, 3) world rotation applied to this recording, or ``None``."""
        if self.canonical_z_up and _UP_AXIS.get(meta["headset"], "z") == "y":
            return Y_UP_TO_Z_UP
        return None

    def up_axis_of(self, meta):
        """World up axis of a recording as returned by this dataset."""
        return "z" if self.canonical_z_up else _UP_AXIS.get(meta["headset"], "z")

    def _windows(self, n):
        """``[start, end)`` clip windows over a recording of ``n`` frames."""
        if n <= 0:
            return []
        if self.clip_len <= 0:
            return [(0, n)]
        stride = self.clip_stride or self.clip_len
        out = []
        for s in range(0, n, stride):
            e = min(s + self.clip_len, n)
            if e - s >= self.min_clip_len:
                out.append((s, e))
            if e == n:
                break
        return out

    # ---- recording parsing -------------------------------------------------- #
    def _recording(self, ri):
        """Parsed full-recording arrays for recording ``ri`` (LRU-cached)."""
        name = self.recordings[ri]["recording_name"]
        if name in self._cache:
            self._cache.move_to_end(name)
            return self._cache[name]
        rec = self._parse_recording(self.recordings[ri])
        self._cache[name] = rec
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return rec

    def _parse_recording(self, meta):
        """Read one recording's object + hand poses onto a merged timeline.

        No gap filling: untracked frames keep placeholder x-data and a False
        mask.  Returns a dict of full-length arrays (N frames): ``timestamps_ns``,
        ``x_lhand`` / ``x_rhand`` (N, 99), ``lhand_valid`` / ``rhand_valid``
        (N,), ``beta`` (10,), ``x_obj`` (N, K, 9), ``obj_valid`` (N, K),
        ``obj_uids`` (K,).
        """
        rec_dir = meta["rec_dir"]

        # ---- objects: canonical timeline -------------------------------- #
        df = pd.read_csv(pjoin(rec_dir, "dynamic_objects.csv"))
        all_ts = np.unique(df["timestamp[ns]"].to_numpy(np.int64))
        frame_ts, frame_of = _merge_timestamps(all_ts, self.tol_ns)
        N = len(frame_ts)
        row_frame = frame_of[np.searchsorted(all_ts, df["timestamp[ns]"].to_numpy(np.int64))]

        present = set(df["object_uid"].astype(str))
        uids = [u for u in meta.get("object_uids", []) if u in present]
        uids += sorted(present - set(uids))
        K = len(uids)
        obj_t = np.zeros((N, K, 3))
        obj_q = np.zeros((N, K, 4))
        obj_q[..., 0] = 1.0
        obj_valid = np.zeros((N, K), dtype=bool)
        df_uid = df["object_uid"].astype(str).to_numpy()
        t_cols = df[["t_wo_x[m]", "t_wo_y[m]", "t_wo_z[m]"]].to_numpy(np.float64)
        q_cols = df[["q_wo_w", "q_wo_x", "q_wo_y", "q_wo_z"]].to_numpy(np.float64)
        for k, uid in enumerate(uids):
            rows = np.nonzero(df_uid == uid)[0]
            # First pose per frame (Aria: the earlier of the two stream copies).
            frames, first = np.unique(row_frame[rows], return_index=True)
            rows = rows[first]
            obj_t[frames, k] = t_cols[rows]
            obj_q[frames, k] = q_cols[rows]
            obj_valid[frames, k] = True

        C = self._world_rot(meta)
        R_obj = _quat_to_rotmat(obj_q.reshape(-1, 4))                 # (N*K, 3, 3)
        t_obj = torch.from_numpy(obj_t.reshape(-1, 3)).float()
        if C is not None:   # v' = C (R v + t) = (C R) v + C t
            Ct = torch.from_numpy(C).float()
            R_obj, t_obj = Ct @ R_obj, t_obj @ Ct.T
        extmat = torch.zeros(N * K, 3, 4)
        extmat[:, :, :3] = R_obj
        extmat[:, :, 3] = t_obj
        x_obj = transform_obj_to_xdata(extmat).reshape(N, K, 9).astype(np.float32)
        x_obj[~obj_valid] = PLACEHOLDER_OBJ_XDATA       # untracked: placeholder

        # ---- hands: MANO wrist xform + PCA pose ------------------------- #
        hands = {s: {"t": np.zeros((N, 3)), "q": np.tile([1.0, 0, 0, 0], (N, 1)),
                     "pca": np.zeros((N, _N_PCA)), "valid": np.zeros(N, bool)}
                 for s in _HAND_KEYS}
        beta = None
        with open(pjoin(rec_dir, "mano_hand_pose_trajectory.jsonl")) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        hand_ts = np.asarray([l["timestamp_ns"] for l in lines], dtype=np.int64)
        hand_frame = _match_frames(frame_ts, hand_ts, self.tol_ns)
        for line, fi in zip(lines, hand_frame):
            if fi < 0:
                continue
            for side, key in _HAND_KEYS.items():
                hp = line["hand_poses"].get(key)
                h = hands[side]
                if hp is None or h["valid"][fi]:   # keep the first pose per frame
                    continue
                h["t"][fi] = hp["wrist_xform"]["t_xyz"]
                h["q"][fi] = hp["wrist_xform"]["q_wxyz"]
                h["pca"][fi] = hp["pose"][:_N_PCA]
                h["valid"][fi] = True
                if beta is None and "betas" in hp:
                    beta = np.asarray(hp["betas"], np.float32)[:10]
        if beta is None:
            beta = np.zeros(10, np.float32)

        out = {
            "timestamps_ns": frame_ts,
            "beta": beta,
            "x_obj": x_obj,
            "obj_valid": obj_valid,
            "obj_uids": uids,
        }
        for side, h in hands.items():
            out[f"x_{side[0]}hand"] = self._hand_xdata(h, side, C=C, beta=beta)
            out[f"{side[0]}hand_valid"] = h["valid"]
        return out

    def _hand_xdata(self, h, side, C=None, beta=None):
        """One hand's wrist xform + PCA pose -> (N, 99) trans + rot6d.

        Expands the 15 PCA coefficients to the full 45-D axis-angle finger pose
        exactly as smplx does for ``flat_hand_mean=False``
        (``pca @ hands_components[:15] + hands_mean``) and takes the wrist
        rotation as the MANO global orientation.  Untracked frames
        (``h["valid"] == False``) get ``PLACEHOLDER_HAND_XDATA``.

        ``C`` (3, 3) optionally rotates the world.  MANO poses a hand as
        ``V = R (V0 - J0) + J0 + t`` (rotation about the root joint ``J0`` of
        the shape ``beta``), so a rigid world rotation ``C V`` is the pose
        ``R' = C R``, ``t' = C (J0 + t) - J0``.
        """
        comps, mean = _mano_pca(self.mano_root, side)
        finger = h["pca"].astype(np.float32) @ comps[:_N_PCA] + mean   # (N, 45)
        R_root = _quat_to_rotmat(h["q"])                               # (N, 3, 3)
        t = np.asarray(h["t"], dtype=np.float64)
        if C is not None:
            J0 = _mano_root_joint(self.mano_root, side, beta)
            R_root = torch.from_numpy(C).float() @ R_root
            t = (J0 + t) @ C.T - J0
        root = matrix_to_axis_angle(R_root)                            # (N, 3)
        pose = torch.cat([root, torch.from_numpy(finger)], dim=1)      # (N, 48)
        trans = torch.from_numpy(t.astype(np.float32))
        x_hand = transform_hand_to_xdata(trans, pose).astype(np.float32)
        x_hand[~h["valid"]] = PLACEHOLDER_HAND_XDATA                   # untracked
        return x_hand

    # ---- API --------------------------------------------------------------- #
    def __len__(self):
        return len(self.clips)

    def clip_meta(self, item):
        """Cheap per-clip metadata (no pose parsing) for listings.

        Returns ``{"seq_name", "recording", "headset", "obj_names", "nframes"}``.
        ``obj_names`` come from ``metadata.json`` (objects in the recording).
        """
        ri, s, e = self.clips[int(item)]
        rec = self.recordings[ri]
        return {
            "seq_name": f"{rec['recording_name']}:{s}-{e}",
            "recording": rec["recording_name"],
            "headset": rec["headset"],
            "obj_names": list(rec.get("object_names", [])),
            "nframes": int(e - s),
        }

    def __getitem__(self, item):
        ri, s, e = self.clips[int(item)]
        meta = self.recordings[ri]
        rec = self._recording(ri)

        lvalid = rec["lhand_valid"][s:e]
        rvalid = rec["rhand_valid"][s:e]
        uids = rec["obj_uids"]
        beta = rec["beta"]
        return {
            "seq_name": f"{meta['recording_name']}:{s}-{e}",
            "recording": meta["recording_name"],
            "participant": meta["participant_id"],
            "headset": meta["headset"],
            "up_axis": self.up_axis_of(meta),
            "text": "",
            "nframes": int(e - s),
            "start_frame": int(s),
            "end_frame": int(e),
            "timestamps_ns": rec["timestamps_ns"][s:e],
            "is_lhand": bool(lvalid.any()),
            "is_rhand": bool(rvalid.any()),
            "lhand_valid": lvalid,                    # (T,) bool
            "rhand_valid": rvalid,                    # (T,) bool
            "x_lhand": rec["x_lhand"][s:e],           # (T, 99)
            "x_rhand": rec["x_rhand"][s:e],           # (T, 99)
            "beta": beta,                             # (10,)
            "lhand_beta": beta,
            "rhand_beta": beta,
            "x_obj": rec["x_obj"][s:e],               # (T, K, 9)
            "obj_valid": rec["obj_valid"][s:e],       # (T, K) bool
            "obj_uids": list(uids),
            "obj_names": [self.uid_to_name.get(u, u) for u in uids],
            "obj_meshes": [pjoin(self.asset_root, f"{u}.glb") for u in uids],
        }


# --------------------------------------------------------------------------- #
# Single-object segments (MotionGRAB interface)
# --------------------------------------------------------------------------- #
class Hot3DSubjectLayers:
    """Per-participant left/right MANO layers with the HOT3D hand shape baked in.

    HOT3D fits one MANO ``beta`` per participant.  Each participant's layers
    are deep copies of the shared default layers with the MANO_LEFT shapedirs
    fix applied and ``beta`` baked into the template (``shaped_mano_layer``), so
    posing them with zero betas — as ``MotionDatasetBase._getitem_full`` does —
    gives the participant's hand.  Built lazily; ``get(sbj_id, side)`` matches
    GRAB's ``SubjectManoLayers`` interface.
    """

    def __init__(self, sbj_betas, lhand_layer, rhand_layer):
        self._betas = {str(k): np.asarray(v, np.float32) for k, v in sbj_betas.items()}
        self._default = {"left": lhand_layer, "right": rhand_layer}
        self._layers = {}

    def get(self, sbj_id, side):
        import copy
        from hoidata.utils.hand_proc import fix_mano_left_shapedirs, shaped_mano_layer
        sbj = str(sbj_id)
        if sbj not in self._betas:
            return self._default[side]
        if sbj not in self._layers:
            left = copy.deepcopy(self._default["left"])
            fix_mano_left_shapedirs(left, self._default["right"])
            beta = self._betas[sbj]
            self._layers[sbj] = {
                "left": shaped_mano_layer(left, beta),
                "right": shaped_mano_layer(self._default["right"], beta),
            }
        return self._layers[sbj][side]


class MotionHOT3D(MotionDatasetBase):
    """HOT3D single-object motion segments with ``MotionGRAB``'s interface.

    Built from the preprocessed ``data.npz`` of
    ``hoidata/preprocess/preprocess_hot3d.py``, which splits every long
    multi-object recording into short single-object segments
    (``segment_by_activity``): one object's interaction interval — by default
    the frames where a hand is in contact with it (``seg_activity``) — plus
    the inactive transition frames directly before / after it, with frames
    where more than one object is active dropped.  Each segment is one sample
    with a
    rigid object x-data ``x_obj`` (T, 9) (``R @ v + t``), so the shared
    ``MotionDatasetBase`` windowing applies unchanged: ``max_nframes`` random
    windows (generation) or the full segment (``max_nframes == -1``).

    Differences from MotionGRAB:
      * no text (``text_json=None``) and no contact / distance maps;
      * train / val / test split by participant (``val_participants`` /
        ``test_participants``; ``"all"`` keeps everything);
      * untracked frames keep placeholder x-data; items carry
        ``track_mask_{lhand,rhand,obj}`` (True = tracked) and the matching
        deltas are zeroed wherever frame t or t+1 is untracked;
      * items carry per-frame hand–object contact with the segment's object,
        ``contact_lhand`` / ``contact_rhand`` (raw, before gap joining);
      * ``sbj_id`` is the participant; hands are posed with per-participant
        shaped layers (``Hot3DSubjectLayers``), built lazily from the default
        layers unless ``subject_layers`` is passed.
    ``action_name`` holds the segment id (HOT3D has no action labels);
    ``move_start`` / ``move_end`` bound the segment's active core and
    ``seg_activity`` says which signal defined it (``"contact"`` / ``"moving"``).
    """

    def __init__(self, *args, val_participants=("P0017",),
                 test_participants=("P0021",), **kwargs):
        # Stored before super().__init__(): _load_data runs inside it.
        self.val_participants = set(val_participants or ())
        self.test_participants = set(test_participants or ())
        super().__init__(*args, **kwargs)

    def _split_of(self, participant):
        if participant in self.test_participants:
            return "test"
        if participant in self.val_participants:
            return "val"
        return "train"

    def _load_data(self, data, task, split):
        participant = data["participant"]
        if split == "all":
            mask = np.ones(len(participant), dtype=bool)
        else:
            mask = np.array([self._split_of(str(p)) == split for p in participant])

        self.obj_name = data["obj_name"][mask]
        self.x_lhand = data["x_lhand"][mask]
        self.x_rhand = data["x_rhand"][mask]
        self.x_obj = data["x_obj"][mask]
        self.lhand_valid = data["lhand_valid"][mask]
        self.rhand_valid = data["rhand_valid"][mask]
        self.obj_valid = data["obj_valid"][mask]
        self.lhand_contact = data["lhand_contact"][mask]
        self.rhand_contact = data["rhand_contact"][mask]
        params = json.loads(str(data["seg_params"])) if "seg_params" in data.files else {}
        self.seg_activity = params.get("activity", "moving")
        self.is_lhand = data["is_lhand"][mask]
        self.is_rhand = data["is_rhand"][mask]
        self.nframes = data["nframes"][mask]
        self.seq_name = data["seq_name"][mask]
        self.action_name = self.seq_name          # no action labels in HOT3D
        self.recording = data["recording"][mask]
        self.headset = data["headset"][mask]
        self.up_axis = data["up_axis"][mask]
        self.start_frame = data["start_frame"][mask]
        self.end_frame = data["end_frame"][mask]
        self.move_start = data["move_start"][mask]
        self.move_end = data["move_end"][mask]
        self.beta = data["beta"][mask]
        self.sbj_id = participant[mask]
        self.subject = self.sbj_id                # exposed in _getitem_full items
        self.sbj_betas = {str(s): b for s, b in zip(self.sbj_id, self.beta)}
        # No contact data for HOT3D.
        self.lcf_idx = self.lcov_idx = self.lchj_idx = self.ldist_value = None
        self.rcf_idx = self.rcov_idx = self.rchj_idx = self.rdist_value = None

    # ------------------------------------------------------------------ hooks
    def _get_layers(self, index):
        if (self.subject_layers is None and self.lhand_layer is not None
                and self.rhand_layer is not None):
            self.subject_layers = Hot3DSubjectLayers(
                self.sbj_betas, self.lhand_layer, self.rhand_layer)
        return super()._get_layers(index)

    def _object_name(self, index):
        return self.obj_name[index]

    def _text(self, index, is_lhand, is_rhand):
        return None

    def _slice_obj(self, index, init_frame, nframes):
        # Remember the window start for the tracking masks (_getitem_windowed).
        self._init_frame = int(init_frame)
        return self.x_obj[index][init_frame:init_frame + self._slice_len]

    def _obj_delta(self, x_obj, nframes, max_nframes):
        return self._to_delta(x_obj)

    def _object_pc(self, object_name, item):
        _, obj_pc, obj_pc_normal, _ = self.object_model(object_name)
        normalized_obj_pc, obj_norm_cent, obj_norm_scale = pc_normalize(
            obj_pc, return_params=True)
        item["obj_pc"] = obj_pc
        item["normalized_obj_pc"] = normalized_obj_pc
        item["obj_pc_normal"] = obj_pc_normal
        item["obj_cent"] = obj_norm_cent
        item["obj_scale"] = obj_norm_scale

    def _obj_traj_from_xobj(self, x_obj, nframes):
        # x_obj is (T, 9) trans + rot6d of R with v_world = R @ v + t, so the
        # rotvec is R's own (no inversion, unlike GRAB's v @ R convention).
        from pytorch3d.transforms import rotation_6d_to_matrix
        obj_rotvec = proc_numpy(
            matrix_to_axis_angle(rotation_6d_to_matrix(proc_torch_cuda(x_obj[:, 3:]))))
        obj_traj = np.concatenate([x_obj[:, :3], obj_rotvec], axis=1)    # (T, 6)
        return {"obj_traj": obj_traj.astype(np.float32)}

    # ----------------------------------------------------------------- items
    def _contact_fields(self, index, start, nframes, length):
        """Per-hand contact masks over ``[start, start+nframes)`` padded to
        ``length``: ``{"contact_lhand", "contact_rhand"}``."""
        out = {}
        for key, src in (("lhand", self.lhand_contact), ("rhand", self.rhand_contact)):
            m = np.zeros(length, dtype=bool)
            m[:nframes] = np.asarray(src[index], dtype=bool)[start:start + nframes]
            out[f"contact_{key}"] = m
        return out

    def _track_fields(self, index, start, nframes, length, slice_len=None):
        """Tracking masks over ``[start, start+nframes)`` padded to ``length``,
        plus per-stream delta-valid masks (frame t and t+1 both tracked)."""
        masks, dvalid = {}, {}
        for key, src in (("lhand", self.lhand_valid), ("rhand", self.rhand_valid),
                         ("obj", self.obj_valid)):
            full = np.asarray(src[index], dtype=bool)
            m = np.zeros(length, dtype=bool)
            m[:nframes] = full[start:start + nframes]
            masks[key] = m
            if slice_len is not None:
                ext = full[start:start + slice_len]
                dv = np.zeros(length, dtype=bool)
                pair = (ext[:-1] & ext[1:])[:nframes]
                dv[:len(pair)] = pair
                dvalid[key] = dv
        return masks, dvalid

    def _getitem_windowed(self, index):
        item = super()._getitem_windowed(index)
        masks, dvalid = self._track_fields(
            index, self._init_frame, int(item["nframes"]), self.max_nframes,
            slice_len=self._slice_len)
        for key in ("lhand", "rhand", "obj"):
            item[f"track_mask_{key}"] = masks[key]
            # A delta touching an untracked (placeholder) frame is meaningless.
            item[f"x_{key}_delta"][~dvalid[key]] = 0.0
        item.update(self._contact_fields(
            index, self._init_frame, int(item["nframes"]), self.max_nframes))
        item["seq_name"] = str(self.seq_name[index])
        item["beta"] = np.asarray(self.beta[index], dtype=np.float32)
        return item

    def _getitem_full(self, index):
        item = super()._getitem_full(index)
        n = int(item["nframes"])
        masks, _ = self._track_fields(index, 0, n, n)
        for key in ("lhand", "rhand", "obj"):
            item[f"track_mask_{key}"] = masks[key]
        item.update(self._contact_fields(index, 0, n, n))
        item["seg_activity"] = self.seg_activity
        item["x_obj"] = np.asarray(self.x_obj[index][:n], dtype=np.float32)
        item["seq_name"] = str(self.seq_name[index])
        item["up_axis"] = str(self.up_axis[index])
        item["move_start"] = int(self.move_start[index] - self.start_frame[index])
        item["move_end"] = int(self.move_end[index] - self.start_frame[index])
        return item


class MotionHOT3DDataModule(MotionDataModuleBase):
    """LightningDataModule wrapping :class:`MotionHOT3D`.

    ``prepare_data`` builds the object point clouds and the single-object
    segment ``data.npz`` when missing (or when the segmentation parameters in
    the config changed); splits are by participant.  The ``evaluate`` task
    always yields full-length segments (``max_nframes=-1``).
    """

    dataset_cls = MotionHOT3D

    def prepare_data(self):
        from hoidata.preprocess import preprocess_hot3d
        preprocess_hot3d.run_missing_preprocessing(self.cfg)

    def _make_dataset(self, split: str) -> MotionHOT3D:
        dc = self.cfg
        return self.dataset_cls(
            data_path=dc.data_path,
            data_obj_pc_path=dc.data_obj_pc_path,
            text_json=None,
            max_nframes=-1 if self.task == "evaluate" else dc.max_nframes,
            data_ratio=dc.get("data_ratio", 1.0),
            task=self.task,
            split=split,
            lhand_layer=self.lhand_layer,
            rhand_layer=self.rhand_layer,
            subject_layers=self.subject_layers,
            val_participants=dc.get("val_participants", ["P0017"]),
            test_participants=dc.get("test_participants", ["P0021"]),
        )
