"""HOT3D loader in this repo's ``x_hand`` / ``x_obj`` format.

Basic ``MotionHOT3D`` dataset: every HOT3D recording is cut into fixed-length
clips, and each clip returns the two hand motions (``x_lhand`` / ``x_rhand``)
and the trajectories of **all** tracked dynamic objects (``x_obj``) in the same
trans + 6D-rotation layout used by the GRAB / ARCTIC / GigaHands loaders, plus
per-frame tracking masks, the MANO shape (beta) and the recording metadata.

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

World frame: Aria recordings are **Z-up**, Quest3 recordings are **Y-up**
(the official viewer's ``RIGHT_HAND_Z_UP`` / ``RIGHT_HAND_Y_UP``); each item
reports it as ``up_axis``.

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

from hoidata.utils.proc import transform_hand_to_xdata, transform_obj_to_xdata


#: HOT3D hand index -> side (``loader_hand_poses.LEFT_HAND_INDEX`` = 0).
_HAND_KEYS = {"left": "0", "right": "1"}
#: Headset -> world up axis (matches the official Hot3DVisualizer).
_UP_AXIS = {"Aria": "z", "Quest3": "y"}
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
# Dataset
# --------------------------------------------------------------------------- #
class MotionHOT3D(data.Dataset):
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
        mano_root: MANO model folder, for the PCA basis / mean pose.
        cache_size: number of parsed recordings kept in memory (consecutive
            clips share a recording).

    Each item is a dict with:
        ``seq_name``  : ``"<recording>:<start>-<end>"`` (unique per clip).
        ``recording`` / ``participant`` / ``headset`` : str.
        ``up_axis``   : ``"z"`` (Aria) or ``"y"`` (Quest3) — world up.
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
        print(f"MotionHOT3D [{split}{hs}] {len(self.recordings)} recordings, "
              f"{len(self.clips)} clips (clip_len={self.clip_len})")

    # ---- index construction ------------------------------------------------ #
    def _split_of(self, participant):
        if participant in self.test_participants:
            return "test"
        if participant in self.val_participants:
            return "val"
        return "train"

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

        extmat = torch.zeros(N * K, 3, 4)
        extmat[:, :, :3] = _quat_to_rotmat(obj_q.reshape(-1, 4))
        extmat[:, :, 3] = torch.from_numpy(obj_t.reshape(-1, 3)).float()
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
            out[f"x_{side[0]}hand"] = self._hand_xdata(h, side)
            out[f"{side[0]}hand_valid"] = h["valid"]
        return out

    def _hand_xdata(self, h, side):
        """One hand's wrist xform + PCA pose -> (N, 99) trans + rot6d.

        Expands the 15 PCA coefficients to the full 45-D axis-angle finger pose
        exactly as smplx does for ``flat_hand_mean=False``
        (``pca @ hands_components[:15] + hands_mean``) and takes the wrist
        rotation as the MANO global orientation.  Untracked frames
        (``h["valid"] == False``) get ``PLACEHOLDER_HAND_XDATA``.
        """
        comps, mean = _mano_pca(self.mano_root, side)
        finger = h["pca"].astype(np.float32) @ comps[:_N_PCA] + mean   # (N, 45)
        root = matrix_to_axis_angle(_quat_to_rotmat(h["q"]))           # (N, 3)
        pose = torch.cat([root, torch.from_numpy(finger)], dim=1)      # (N, 48)
        trans = torch.from_numpy(h["t"].astype(np.float32))
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
            "up_axis": _UP_AXIS.get(meta["headset"], "z"),
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
