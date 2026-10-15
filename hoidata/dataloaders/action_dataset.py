"""Dataset interfaces for hand-object motion.

This module holds the *common* dataset machinery shared by the GRAB and ARCTIC
motion datasets:

- ``compute_xdata_delta`` — per-frame x-data delta (world-frame rotation delta).
- ``ActionDataset``       — the action-classifier dataset contract (placeholder).
- ``MotionDatasetBase``   — base ``Dataset`` with the shared generation / eval /
  action-classify item logic, parameterised by a handful of overridable hooks.
- ``MotionDataModuleBase``— base LightningDataModule (split selection, dataloaders).

Dataset-specific subclasses live in:
- ``common/data/grab_dataset.py``   (rigid objects, per-subject MANO vtemplates)
- ``common/data/arctic_dataset.py`` (articulated objects, +1 articulation DoF)
- ``hoidata/dataloaders/hot3d_dataset.py`` (rigid single-object HOT3D segments;
  no text, no contact maps)

``MotionGRAB`` / ``MotionGRABDataModule`` are re-exported from this module (see the
bottom of the file) so existing ``from common.data.action_dataset import
MotionGRABDataModule`` imports keep working after the refactor.
"""

import time
import json
import os.path as osp

import numpy as np
from torch.utils.data import Dataset, DataLoader
import lightning as L

import torch
import pickle

from hoidata.utils.frame import get_valid_mask
from hoidata.utils.proc import get_contact_map, process_dist_map, transform_xdata_to_joints
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix


#: rot6d convention stamped into every preprocessed data.npz.  Caches written
#: before the pytorch3d migration carry no ``rot_repr`` key and store rot6d as
#: the first two *columns* of R, which ``rotation_6d_to_matrix`` would silently
#: misread — hence the hard failure in ``_check_rot_repr``.
ROT_REPR = "pytorch3d_6d"


#: Half-width (frames) of a transition-action window, centred on a grasp-count
#: change: the TA spans ``[p - TA_HALF, p + TA_HALF)`` (2*TA_HALF frames total).
TA_HALF = 5

#: Minimum length (frames) of a stable-grasp (or null) segment left between two
#: transition windows.  When two change points are closer than
#: ``2*TA_HALF + MIN_SEG`` their windows would leave less than this between them;
#: rather than merge into one long transition, the two transitions are split at
#: the midpoint of the change points, back to back.
MIN_SEG = 5

#: Backwards-compatible alias: the old "short null gap" absorb threshold now maps
#: onto the minimum-segment split threshold above.
MIN_NULL = MIN_SEG


def _coalesce_intervals(segs, N):
    """Merge touching / overlapping ``[s, e)`` intervals, clamped to ``[0, N)``.

    Abutting grasps (``[a, m)`` then ``[m, b)``) describe one continuous grasp;
    without merging the shared frame reads as a spurious release+regrasp.
    """
    iv = sorted((max(0, min(int(s), N)), max(0, min(int(e), N)))
                for s, e in (segs or []))
    iv = [(s, e) for s, e in iv if e > s]
    out = []
    for s, e in iv:
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def grasp_count_and_masks(grasps, N):
    """Per-frame grasp count and per-hand boolean masks over ``[0, N)``.

    Returns ``(count, lc, rc)`` where ``count`` is int8 (0/1/2) and ``lc``/``rc``
    are bool arrays of whether the left / right hand is in stable grasp at each
    frame, from the coalesced per-side intervals.
    """
    N = int(N)
    lc = np.zeros(N, dtype=bool)
    rc = np.zeros(N, dtype=bool)
    if grasps:
        for s, e in _coalesce_intervals(grasps.get("left", []), N):
            lc[s:e] = True
        for s, e in _coalesce_intervals(grasps.get("right", []), N):
            rc[s:e] = True
    count = lc.astype(np.int8) + rc.astype(np.int8)
    return count, lc, rc


def segment_grasp_timeline(grasps, N, ta_half=TA_HALF, min_seg=MIN_SEG,
                           min_null=None):
    """Split a sequence into null / transition / stable-grasp action segments.

    From the per-side stable-grasp intervals, build the grasp-*count* timeline
    (0 = no hand in stable grasp, 1 = one hand, 2 = both).  A **transition
    action** (TA) is the ``ta_half`` frames either side of every frame where the
    count changes among {null, one-hand, both-hand} — i.e. ``[p-ta_half,
    p+ta_half)`` around each count-change point ``p``.  Every non-TA frame is
    **null** (count 0) or **stable_grasp** (count >= 1).

    When two consecutive change points ``p_i < p_{i+1}`` are **close** — closer
    than ``2*ta_half + min_seg``, so their windows would leave fewer than
    ``min_seg`` frames of stable-grasp / null between them — the two transitions
    are **split** rather than merged: window ``i`` ends and window ``i+1`` begins
    at the midpoint ``(p_i + p_{i+1}) // 2``, giving two back-to-back transition
    segments (no null/stable_grasp between them).  When they are far enough apart,
    the windows keep their full ``±ta_half`` extent and the middle is filled by
    the grasp count (null or stable_grasp).

    Touching / overlapping same-hand intervals are coalesced first so an abutting
    boundary is not read as a spurious release+regrasp count flicker.

    ``min_null`` is a deprecated alias for ``min_seg`` (kept for old callers).

    Returns a list of contiguous ``{"start", "end", "type"}`` segments covering
    ``[0, N)``, ``type`` in ``{"null", "transition", "stable_grasp"}``.
    """
    if min_null is not None:
        min_seg = min_null
    N = int(N)
    if N <= 0:
        return []

    count, _lc, _rc = grasp_count_and_masks(grasps, N)

    # Change points, then per-window [start, end) around each (clamped to [0, N)).
    changes = list(np.nonzero(count[1:] != count[:-1])[0] + 1)  # frames p that step
    starts = [max(0, p - ta_half) for p in changes]
    ends = [min(N, p + ta_half) for p in changes]

    # For each adjacent pair, decide split-vs-separate.  "Close" pairs (change
    # points nearer than 2*ta_half + min_seg) abut at the midpoint of the two
    # change points; well-separated pairs keep their full windows and leave the
    # count-driven segment between them.  A split midpoint is recorded as a forced
    # segment boundary so the two abutting transitions stay two segments (they
    # would otherwise run-length-encode into one indistinguishable band).
    cuts = set()
    for i in range(len(changes) - 1):
        if changes[i + 1] - changes[i] < 2 * ta_half + min_seg:
            mid = (changes[i] + changes[i + 1]) // 2
            ends[i] = mid
            starts[i + 1] = mid
            cuts.add(int(mid))

    # Paint the TA mask from the (possibly split) windows.
    is_ta = np.zeros(N, dtype=bool)
    for s, e in zip(starts, ends):
        if e > s:
            is_ta[s:e] = True

    # Per-frame type, then run-length encode into contiguous segments, forcing a
    # break at every split midpoint so back-to-back transitions remain separate.
    seg_type = np.where(
        is_ta, "transition",
        np.where(count == 0, "null", "stable_grasp")).astype(object)

    segments = []
    a = 0
    for f in range(1, N + 1):
        if f == N or seg_type[f] != seg_type[a] or f in cuts:
            segments.append({"start": int(a), "end": int(f),
                             "type": str(seg_type[a])})
            a = f
    return segments


def _check_rot_repr(data, data_path):
    """Reject a preprocessed .npz whose rot6d convention is not ``ROT_REPR``."""
    repr_name = str(data["rot_repr"]) if "rot_repr" in data.files else None
    if repr_name != ROT_REPR:
        raise RuntimeError(
            f"{data_path} has rot_repr={repr_name!r}, expected {ROT_REPR!r}.\n"
            "This cache predates the pytorch3d 6D-rotation migration and its "
            "x_lhand / x_rhand / x_obj would be silently misinterpreted.\n"
            f"Convert it in place with:  python -m scripts.migrate_rot6d {data_path}"
        )


def _seed_worker(worker_id):
    """Per-worker RNG seeding for DataLoader workers.

    The generation getitem draws a random window start via ``np.random.randint``
    (NumPy's *global* RNG).  PyTorch reseeds Python's ``random`` and torch per
    worker, but NOT NumPy — so without this, every worker forks the same NumPy
    state and draws identical window starts, collapsing window diversity.  We
    derive a unique per-worker seed from ``torch.initial_seed()`` (which PyTorch
    sets uniquely per worker, and which varies each epoch), so each worker — and
    each epoch — samples different sub-clips while staying reproducible given the
    global seed.
    """
    import random
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def compute_xdata_delta(xdata):
    """Per-frame delta of an x-data sequence (trans + rot6d).

    x-data layout: ``[:, :3]`` translation, ``[:, 3:]`` rot6d for one or more
    rotations (16 joints for hands → 96, one rotation for the object → 6).
    rot6d is pytorch3d's convention (first two *rows* of R); see
    ``hoidata.utils.proc.transform_hand_to_xdata``.

    - translation delta: ``trans[t+1] - trans[t]``.
    - rotation delta: computed in rotation space, NOT by subtracting 6D.  For
      each rotation R_t (recovered from its 6D), the world-frame relative
      rotation to the next frame is ``R_delta = R_{t+1} @ R_t^T`` (so
      ``R_{t+1} = R_delta @ R_t``), converted back to 6D.

    Args:
        xdata: (T, 3 + n_rot*6) numpy array.
    Returns:
        (T-1, 3 + n_rot*6) numpy array of deltas.  Empty (0, D) if T < 2.
    """
    xdata = np.asarray(xdata, dtype=np.float32)
    T, D = xdata.shape
    if T < 2:
        return np.zeros((0, D), dtype=np.float32)

    trans = xdata[:, :3]
    rot6d = xdata[:, 3:]
    n_rot = rot6d.shape[1] // 6

    trans_delta = trans[1:] - trans[:-1]                       # (T-1, 3)

    rot6d_t = torch.from_numpy(rot6d).reshape(T, n_rot, 6)
    rotmat = rotation_6d_to_matrix(rot6d_t)                    # (T, n_rot, 3, 3)
    R_cur = rotmat[:-1]                                        # R_t
    R_nxt = rotmat[1:]                                         # R_{t+1}
    R_delta = R_nxt @ R_cur.transpose(-1, -2)                  # world-frame: R_{t+1} @ R_t^T
    rot6d_delta = matrix_to_rotation_6d(R_delta).reshape(T - 1, n_rot * 6)
    rot6d_delta = rot6d_delta.cpu().numpy().astype(np.float32)

    return np.concatenate([trans_delta, rot6d_delta], axis=1)  # (T-1, D)


class ActionDataset(Dataset):
    """Placeholder dataset for hand-action classification.

    Subclass this and implement ``__init__`` and ``__getitem__``.

    -------------------------------------------------------------------------
    Required ``__getitem__`` return value
    -------------------------------------------------------------------------
    ``__getitem__`` must return a dict with **at minimum** the following keys:

    ``joints``  : np.ndarray, shape (T, 42, 3), dtype float32
        3-D world-space positions of 42 hand joints per frame.
        Layout: first 21 entries are left-hand joints, last 21 are right-hand.

        Per-hand MANO joint ordering (indices 0-20 within each hand block):
            0         — wrist
            1, 2, 3   — index  (proximal → distal phalanx)
            4, 5, 6   — middle
            7, 8, 9   — pinky
            10, 11,12 — ring
            13, 14,15 — thumb
            16        — index  fingertip
            17        — middle fingertip
            18        — pinky  fingertip
            19        — ring   fingertip
            20        — thumb  fingertip

        Sequences must be padded (or truncated) to a fixed ``max_frames``
        length so that DataLoader can form batches.  Padding frames should
        repeat the last valid frame or be set to zeros — the trainer masks
        them out via ``seq_len``.

        Coordinate frame: consistent across all samples (e.g. world space or
        wrist-relative).  Using a wrist-relative frame for each hand tends to
        help the classifier focus on hand pose rather than global position.

    ``action_name``  : str
        Human-readable action label for this sample (e.g. ``"drink"``).

    ``seq_len``  : int
        Number of valid (non-padded) frames in ``joints``.  Must satisfy
        1 <= seq_len <= max_frames.  The trainer slices ``joints`` to a
        window of ``args.window_size`` frames, so very short sequences
        (seq_len < window_size) will produce a window that is partially padded.

    -------------------------------------------------------------------------
    Optional keys (ignored by the trainer but useful for debugging)
    -------------------------------------------------------------------------
    ``sample_id``   : str  — unique identifier for this sequence
    ``action_names``: list[str] — human-readable action labels
    ``hand_type``   : str  — 'left', 'right', or 'both'
    ``obj_name``    : str  — name of the interacted object

    -------------------------------------------------------------------------
    DataLoader configuration used by the trainer
    -------------------------------------------------------------------------
    The trainer constructs DataLoaders as::

        DataLoader(dataset, batch_size=1, shuffle=True,
                   num_workers=0, drop_last=True)

    ``batch_size=1`` is fixed because the LSTM hidden state is re-initialised
    per forward pass and the loss accumulation assumes single-sample batches.
    """

    def __init__(self, data_dir: str, split: str = 'train', max_frames: int = 20):
        """
        Args:
            data_dir:   Path to the root directory containing processed data files.
            split:      One of 'train', 'val', 'test'.
            max_frames: All sequences will be padded / truncated to this length.
        """
        raise NotImplementedError(
            "ActionDataset.__init__ must be implemented.  "
            "Load your data files here and store a list of sample metadata "
            "so that __getitem__ can retrieve individual samples."
        )

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict:
        """Return a single sample dict.  See class docstring for the required
        keys and shapes."""
        raise NotImplementedError


class MotionDatasetBase(Dataset):
    """Base motion dataset shared by GRAB and ARCTIC.

    The shared per-item logic (windowing, hand x-data + deltas, valid masks,
    distance / coverage maps, text, object point cloud) lives here.  Subclasses
    fill in dataset-specific behaviour through these hooks:

    - ``_load_data(data, task, split)`` : populate the per-sample arrays from the
      opened ``np.load`` handle (``x_lhand``, ``x_rhand``, ``is_lhand``,
      ``is_rhand``, ``action_name``, ``nframes`` and whatever else the dataset
      needs; object arrays handled by ``_slice_obj``).
    - ``_object_name(index)``          : raw object name for ``object_model`` lookup.
    - ``_text(index, is_lhand, is_rhand)`` : natural-language prompt for the sample.
    - ``_slice_obj(index, init_frame, nframes)`` : object x-data window
      (GRAB → (T,9); ARCTIC → (T,10) incl. articulation angle).
    - ``_obj_delta(x_obj, nframes, max_nframes)`` : per-frame object delta.
    - ``_object_pc(object_name, item)`` : unpack the object-model tuple, fill
      obj_pc / normalized / normals / cent / scale (and ARCTIC ``obj_pc_top_idx``).
    - ``_obj_traj_from_xobj(x_obj, nframes)`` : evaluation object trajectory
      (returns a dict merged into the eval item, e.g. ``obj_traj`` and, for
      ARCTIC, ``obj_angle``).

    Subclasses also set the class attributes ``val_obj_name`` / ``test_obj_name``
    / ``all_obj_name`` (object-name split lists) used by ``_split_objs``.
    """

    # Subclasses override these split lists.
    all_obj_name = []
    val_obj_name = []
    test_obj_name = []

    def __init__(
        self,
        data_path,
        data_obj_pc_path,
        text_json,
        max_nframes,
        data_ratio=1.0,
        task="generation",
        split="train",
        lhand_layer=None,
        rhand_layer=None,
        subject_layers=None,
        **kwargs,
    ):
        super().__init__()

        self.data_path = data_path
        self.data_obj_pc_path = data_obj_pc_path
        self.max_nframes = max_nframes
        self.data_ratio = data_ratio
        self.task = task
        # Default (mean-template) layers, used as a fallback when no per-subject
        # template is available for a sample.
        self.lhand_layer = lhand_layer
        self.rhand_layer = rhand_layer
        # Optional per-subject MANO layers (subject-specific hand vtemplates).
        self.subject_layers = subject_layers
        # Subclasses may set ``self.sbj_id`` in _load_data; default to none.
        self.sbj_id = None

        start_time = time.time()
        print(f"Start to read data ({self.__class__.__name__})!!!")
        with np.load(data_path, allow_pickle=True) as data:
            _check_rot_repr(data, data_path)
            self._load_data(data, task, split)

        if task != "action_classify":
            # text_json=None: dataset without text descriptions (e.g. HOT3D).
            if task != "vae_training" and text_json is not None:
                with open(text_json, "r") as f:
                    self.text_description = json.load(f)
            else:
                self.text_description = None
            self.object_model = ObjectModel(data_obj_pc_path)
        else:
            self.text_description = None
            self.object_model = None

        print(f"Finish to read data ({self.__class__.__name__})!!!",
              f"{time.time()-start_time:.2f}s")

    # ------------------------------------------------------------------ hooks
    def _load_data(self, data, task, split):
        raise NotImplementedError

    def _object_name(self, index):
        raise NotImplementedError

    def _text(self, index, is_lhand, is_rhand):
        raise NotImplementedError

    def _slice_obj(self, index, init_frame, nframes):
        raise NotImplementedError

    def _obj_delta(self, x_obj, nframes, max_nframes):
        raise NotImplementedError

    def _object_pc(self, object_name, item):
        raise NotImplementedError

    def _obj_traj_from_xobj(self, x_obj, nframes):
        raise NotImplementedError

    # ----------------------------------------------------------- split helper
    @classmethod
    def _split_objs(cls, split):
        if split == "val":
            return set(cls.val_obj_name)
        elif split == "test":
            return set(cls.test_obj_name)
        # train
        return set(cls.all_obj_name) - set(cls.val_obj_name) - set(cls.test_obj_name)

    # -------------------------------------------------------------- mechanics
    def __len__(self):
        return int(len(self.action_name) * self.data_ratio)

    def _get_layers(self, index):
        """Per-subject (lhand_layer, rhand_layer) for this sample.

        Uses the subject-specific hand vtemplate when both a SubjectManoLayers
        cache and a recorded subject id are available; otherwise falls back to
        the shared default (mean-template) layers.
        """
        if self.subject_layers is not None and self.sbj_id is not None:
            sbj = str(self.sbj_id[index])
            return (self.subject_layers.get(sbj, 'left'),
                    self.subject_layers.get(sbj, 'right'))
        return self.lhand_layer, self.rhand_layer

    # ----------------------------------------------------------------- items
    def __getitem__(self, index):
        """Return one motion clip.

        ``max_nframes == -1`` means no frame-length limit: the entire sequence
        (all valid frames) is treated as a single clip, and full-resolution hand
        vertices/joints are returned (physics-metric evaluation).  Otherwise the
        clip is a random ``max_nframes`` window with the standard generation
        item layout (x-data + deltas + valid masks + object point cloud).
        """
        if self.max_nframes == -1:
            return self._getitem_full(index)
        return self._getitem_windowed(index)

    def _getitem_full(self, index):
        """Return one full-length motion clip for physics-metric evaluation.

        No frame-length limit: the entire sequence (all valid frames) is treated
        as a single clip.  Hand vertices come from the active hand(s); the object
        trajectory is delegated to ``_obj_traj_from_xobj`` (rigid for GRAB,
        articulated for ARCTIC).
        """
        from hoidata.utils.proc import transform_xdata_to_verts, transform_xdata_to_joints

        nframes = int(self.nframes[index])
        is_lhand = self.is_lhand[index]
        is_rhand = self.is_rhand[index]
        if is_lhand is None:
            is_lhand, is_rhand = True, True
        is_lhand, is_rhand = bool(is_lhand), bool(is_rhand)
        lhand_layer, rhand_layer = self._get_layers(index)

        # Full sequence — no max_nframes slicing.  Capsule-based contact detection
        # (ManipNet) needs per-hand joints; we also keep verts for vertex-level
        # penetration rate.  Both are collected per active hand, in [left, right]
        # order, matching the concatenation used downstream.
        hand_verts_list = []   # each (T, 778, 3)
        hand_joints_list = []  # each (T, 21, 3)
        hand_sides = []        # 'left' / 'right' per active hand
        if is_lhand:
            x_lhand = self.x_lhand[index][:nframes]
            hand_verts_list.append(transform_xdata_to_verts(x_lhand, lhand_layer))
            hand_joints_list.append(transform_xdata_to_joints(x_lhand, lhand_layer))
            hand_sides.append("left")
        if is_rhand:
            x_rhand = self.x_rhand[index][:nframes]
            hand_verts_list.append(transform_xdata_to_verts(x_rhand, rhand_layer))
            hand_joints_list.append(transform_xdata_to_joints(x_rhand, rhand_layer))
            hand_sides.append("right")
        # Active hand(s) only; concatenate verts when both hands are present.
        hand_verts = np.concatenate(hand_verts_list, axis=1).astype(np.float32)  # (T, 778[*2], 3)
        hand_joints = np.stack(hand_joints_list, axis=0).astype(np.float32)      # (H, T, 21, 3)

        item = {
            "hand_verts": hand_verts,                  # (T, 778 or 1556, 3)
            "hand_joints": hand_joints,                # (H, T, 21, 3) per active hand
            "hand_sides": hand_sides,                  # ['left'|'right', ...]
            "obj_name":   str(self._object_name(index)),
            "action_name": str(self.action_name[index]),
            "nframes":    nframes,
            "is_lhand":   is_lhand,
            "is_rhand":   is_rhand,
        }
        # Object trajectory (+ articulation for ARCTIC) is dataset-specific.
        item.update(self._obj_traj_from_xobj(self.x_obj[index][:nframes], nframes))
        # Optional per-sequence annotation intervals + subject (ARCTIC full_motion
        # data only); used by the web viewer to colour labelled frame ranges.
        if getattr(self, "labels", None) is not None:
            item["labels"] = list(self.labels[index])
        if getattr(self, "subject", None) is not None:
            item["subject"] = str(self.subject[index])
        if getattr(self, "grasps", None) is not None:
            item["grasps"] = self.grasps[index]
        return item

    def _getitem_windowed(self, index):
        item = {}

        total_frames = self.nframes[index]

        max_nframes = self.max_nframes

        # The window holds `nframes` absolute frames; to compute the LAST frame's
        # delta from a real successor (not zero-pad) we additionally read ONE extra
        # frame after the window when the sequence is long enough.  So the slice
        # length is `nframes + 1` whenever `has_next`, and `init_frame` is chosen so
        # that extra frame exists.
        if total_frames > max_nframes:
            # Room for max_nframes + 1: reserve the successor for the last delta.
            init_frame = np.random.randint(0, total_frames - max_nframes)
            nframes = max_nframes
            has_next = (init_frame + nframes) < total_frames  # True unless flush at end
        else:
            # Whole (short) sequence; a successor exists for every frame but the
            # genuine last frame of the sequence.
            init_frame = 0
            nframes = total_frames
            has_next = False

        item["nframes"] = nframes
        # Number of absolute frames to read: window + 1 successor when available.
        slice_len = nframes + 1 if has_next else nframes

        def _to_delta(x_ext):
            """Per-frame delta over the window, padded to max_nframes.

            ``x_ext`` is the absolute window of length ``slice_len`` = nframes (+1
            successor when ``has_next``).  delta[t] = x_ext[t+1] - x_ext[t] for
            t in [0, nframes).  With the successor present every one of the nframes
            frames — including the last — gets a real delta; only when the window
            is flush against the end of the sequence (no successor) does the final
            frame's delta stay zero.
            """
            delta = np.zeros((max_nframes, x_ext.shape[1]), dtype=np.float32)
            if x_ext.shape[0] >= 2:
                d = compute_xdata_delta(x_ext)          # (slice_len-1, D)
                delta[:d.shape[0]] = d[:nframes]
            return delta

        # Expose state the subclass object hooks need.
        self._to_delta = _to_delta
        self._slice_len = slice_len
        self._has_next = has_next

        # ---- Object (dataset-specific slice + delta) ----
        x_obj = self._slice_obj(index, init_frame, nframes)
        x_obj_abs = x_obj[:max_nframes]
        if x_obj_abs.shape[0] < max_nframes:
            pad = np.zeros((max_nframes - x_obj_abs.shape[0], x_obj_abs.shape[1]), dtype=np.float32)
            x_obj_abs = np.concatenate([x_obj_abs, pad], axis=0)
        item["x_obj"] = x_obj_abs
        item["x_obj_delta"] = self._obj_delta(x_obj, nframes, max_nframes)

        # ---- Hands ----
        # Read slice_len = nframes (+1 successor) frames so the last delta is real;
        # keep max_nframes absolute frames and compute the delta from the extended
        # window.
        hand_dim = self.x_lhand[index].shape[-1]

        def _hand(x_src, active):
            if not active:
                z = np.zeros((max_nframes, hand_dim), dtype=np.float32)
                return z, z
            x_ext = x_src[index][init_frame:init_frame + slice_len]  # (slice_len, D)
            x_abs = x_ext[:max_nframes]
            if x_abs.shape[0] < max_nframes:
                pad = np.zeros((max_nframes - x_abs.shape[0], x_abs.shape[1]), dtype=np.float32)
                x_abs = np.concatenate([x_abs, pad], axis=0)
            return x_abs, _to_delta(x_ext)

        ## Set to always available if taking longer sequence as input.
        is_lhand = self.is_lhand[index]
        is_rhand = self.is_rhand[index]

        if is_lhand is not None and getattr(self, "lcf_idx", None) is not None:
            item["is_lhand"] = is_lhand
            item["is_rhand"] = is_rhand
            # ---- Distance / coverage maps (shared) ----
            lcf_idx = self.lcf_idx[index]
            lcov_idx = self.lcov_idx[index]
            lchj_idx = self.lchj_idx[index]
            ldist_value = self.ldist_value[index]
            item["ldist_map"] = process_dist_map(
                self.max_nframes, init_frame, lcf_idx, lcov_idx, lchj_idx,
                ldist_value, is_lhand)

            rcf_idx = self.rcf_idx[index]
            rcov_idx = self.rcov_idx[index]
            rchj_idx = self.rchj_idx[index]
            rdist_value = self.rdist_value[index]
            item["rdist_map"] = process_dist_map(
                self.max_nframes, init_frame, rcf_idx, rcov_idx, rchj_idx,
                rdist_value, is_rhand)

            lcov_map = get_contact_map(lcov_idx, 1024, is_lhand)
            rcov_map = get_contact_map(rcov_idx, 1024, is_rhand)
            cov_map = (lcov_map + rcov_map) > 0
            item["cov_map"] = cov_map.astype(np.float32)
        elif is_lhand is not None:
            # Per-sequence hand flags but no contact data (e.g. HOT3D): no
            # distance / coverage maps.
            item["is_lhand"] = is_lhand
            item["is_rhand"] = is_rhand
        else:
            is_lhand = True
            is_rhand = True
            item["is_lhand"] = True
            item["is_rhand"] = True

        item["x_lhand"], item["x_lhand_delta"] = _hand(self.x_lhand, is_lhand)
        item["x_rhand"], item["x_rhand_delta"] = _hand(self.x_rhand, is_rhand)

        valid_mask_lhand, valid_mask_rhand, valid_mask_obj = get_valid_mask(
            is_lhand, is_rhand, max_nframes, nframes)  # max_nframes: 2x frames
        item["valid_mask_lhand"] = valid_mask_lhand
        item["valid_mask_rhand"] = valid_mask_rhand
        item["valid_mask_obj"] = valid_mask_obj

        object_name = self._object_name(index)
        item["object_name"] = object_name
        if self.text_description is not None:
            item["text"] = self._text(index, is_lhand, is_rhand)

        # ---- Object point cloud (dataset-specific unpack) ----
        self._object_pc(object_name, item)

        return item


class MotionDataModuleBase(L.LightningDataModule):
    """Base LightningDataModule for the motion datasets.

    The ``task`` parameter selects the behaviour:

    - ``"action_classify"``: object-name based train/val/test splits, so val and
      test dataloaders are provided.
    - ``"generation"``: no held-out splits — trains on the full dataset, so only
      a single training set / dataloader is created.
    - ``"evaluate"``: physics-metric sanity check — exposes the test split (same
      obj-name selection as action_classify) via ``test_dataloader`` with full
      length, unbatched (batch_size=1) sequences.

    Subclasses set ``dataset_cls`` (the concrete ``MotionDatasetBase`` subclass)
    and may override ``_make_dataset`` to inject dataset-specific construction
    arguments (e.g. GRAB's per-subject MANO layers).

    Args:
        cfg: Hydra dataset config node with keys:
            data_path, data_obj_pc_path, text_json, max_nframes, mano_root
            (optionally data_ratio for the generation task)
        lhand_layer: ManoLayer_C for the left hand (created externally, shared)
        rhand_layer: ManoLayer_C for the right hand
        task: "action_classify", "generation", or "evaluate"
        batch_size, num_workers: passed to DataLoader
    """

    dataset_cls = None  # subclass sets this

    def __init__(self, cfg, lhand_layer, rhand_layer,
                 task="action_classify", batch_size=1, num_workers=0):
        super().__init__()
        self.cfg = cfg
        self.lhand_layer = lhand_layer
        self.rhand_layer = rhand_layer
        self.task = task
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.subject_layers = None  # GRAB subclass may populate this

    def _run_missing_preprocessing(self, pp):
        """Run only the preprocessing steps whose output files are missing.

        ``pp`` is a preprocessing module (``hoidata.preprocess.preprocess_arctic``
        or ``...preprocess_grab``), which must expose ``preprocessing_object``,
        ``preprocessing_data``, ``preprocessing_text``,
        ``preprocessing_balance_weights``, ``preprocessing_text2length`` and
        ``print_text_data_num`` — all taking ``cfg``.

        Each step is checked independently against the exact config output path(s)
        it writes, so a partially-preprocessed dataset only fills in what is
        missing.  The text / balance-weight / text-length steps are gated on
        ``full_motion`` (matching each pipeline's ``main()``): in full-motion mode
        the data has ``is_lhand=None`` and those steps do not apply.

        Called from each subclass's ``prepare_data`` (Lightning runs it once,
        rank-0 only, before ``setup``).
        """
        cfg = self.cfg
        full_motion = bool(cfg.get("full_motion", False))

        # (label, function, [required output paths]) — a step runs if ANY of its
        # outputs is absent.  Order matters: later steps read earlier outputs.
        steps = [
            ("object mesh point clouds", pp.preprocessing_object,
             [cfg.data_obj_pc_path]),
            ("motion + contact data", pp.preprocessing_data,
             [cfg.data_path]),
        ]
        if not full_motion:
            steps += [
                ("text descriptions", pp.preprocessing_text,
                 [cfg.text_json]),
                ("balance weights", pp.preprocessing_balance_weights,
                 [cfg.balance_weights_path, cfg.t2c_json]),
                ("text-to-length stats", pp.preprocessing_text2length,
                 [cfg.t2l_json]),
            ]

        ran_any = False
        for label, fn, out_paths in steps:
            missing = [p for p in out_paths if not osp.exists(p)]
            if missing:
                print(f"[prepare_data] {label}: missing {missing} — running "
                      f"{fn.__name__}()")
                fn(cfg)
                ran_any = True
            else:
                print(f"[prepare_data] {label}: found {out_paths} — skipping.")

        if not full_motion and ran_any:
            # Sanity-count print (cheap; only when something was (re)built).
            pp.print_text_data_num(cfg)

    def _make_dataset(self, split: str) -> MotionDatasetBase:
        dc = self.cfg
        return self.dataset_cls(
            data_path=dc.data_path,
            data_obj_pc_path=dc.data_obj_pc_path,
            text_json=dc.text_json,
            max_nframes=dc.max_nframes,
            data_ratio=dc.get('data_ratio', 1.0),
            task=self.task,
            split=split,
            lhand_layer=self.lhand_layer,
            rhand_layer=self.rhand_layer,
            subject_layers=self.subject_layers,
        )

    def setup(self, stage: str = None):
        # The evaluate task only exposes the test split.
        if self.task == 'evaluate':
            if stage in ('test', None):
                self.test_set = self._make_dataset('test')
            return

        if stage in ('fit', None):
            self.train_set = self._make_dataset('train')
            # Generation trains on the full dataset; no held-out splits.
            if self.task == 'action_classify':
                self.val_set = self._make_dataset('val')
        if stage in ('test', None):
            # Both action_classify and generation expose the held-out test split
            # (test-object clips) so reconstruction can be inspected on unseen objects.
            self.test_set = self._make_dataset('test')

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, drop_last=True,
                          worker_init_fn=_seed_worker)

    def val_dataloader(self):
        if self.task != 'action_classify':
            return None
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, drop_last=True,
                          worker_init_fn=_seed_worker)

    def test_dataloader(self):
        # Evaluate uses full-length, variable-length sequences → batch_size=1.
        bs = 1 if self.task == 'evaluate' else self.batch_size
        return DataLoader(self.test_set, batch_size=bs, shuffle=False,
                          num_workers=self.num_workers, drop_last=False,
                          worker_init_fn=_seed_worker)


class ObjectModel:
    def __init__(self, pkl_file):
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
            self.object_name = data["object_name"]
            self.obj_pcs = data["obj_pcs"]
            self.obj_pc_normals = data["obj_pc_normals"]
            self.point_sets = data["point_sets"]
            self.obj_path = data["obj_path"]
            self.obj_pc_top = data.get("obj_pc_top", None)

    def __call__(self, object_name):
        if isinstance(object_name, int):
            object_name = self.object_name[object_name]
        point_set = self.point_sets[object_name].copy()
        obj_pc = self.obj_pcs[object_name].copy()
        obj_pc_normal = self.obj_pc_normals[object_name].copy()
        obj_path = self.obj_path[object_name]
        if self.obj_pc_top is not None:
            return point_set, obj_pc, obj_pc_normal, obj_path, self.obj_pc_top[object_name].copy()
        return point_set, obj_pc, obj_pc_normal, obj_path


class StableGraspMotionDataset(MotionDatasetBase):
    """Stable-grasp subset of a motion dataset (dataset-agnostic base).

    This dataset is a filtered version of the full motion dataset, containing only
    sequences where the hand maintains a stable grasp on the object throughout
    the interaction. The stable grasps are selected based on the preprocessed csv file.
    Each stable grasp motion clip is one item.

    Every **row** of the stable-grasp CSV (``sid_seq_name, side, start, end,
    angle_diff``) becomes one independent item — left and right hands count
    separately even when their intervals overlap in time.  Rows are matched to the
    preprocessed sequences by ``sid_seq_name``; rows whose sequence is absent from
    the current split (or whose interval is empty after clamping to ``nframes``)
    are dropped.

    ``__getitem__`` returns the raw x-data window for the grasping hand and the
    object — no MANO posing (the model poses them):

        x_hand   : (T, 99) float32 — trans(3) + rot6d(16*6) for the grasp's hand
        x_object : (T, D)  float32 — object x-data (GRAB D=9; ARCTIC D=10 incl. angle)
        is_lhand / is_rhand : bool — which hand this grasp belongs to
        obj_name : str
        n_frames : int  (== end - start)

    Dataset-specific behaviour lives in these subclass hooks:
      - ``_sg_load_arrays(data, mask)`` : populate per-sequence arrays.
      - ``_sg_split_mask(data, split)`` : boolean mask selecting the split.
      - ``_sg_seq_key(seq_idx)``        : ``(subject, seq_name)`` CSV key.
      - ``_sg_obj_name(seq_idx)``       : object name for a sequence.
      - ``_sg_x_object(seq_idx, a, b)`` : object x-data window.
    """

    def __init__(self, *args, stable_grasp_csv=None, bimanual=False, min_len=0,
                 **kwargs):
        # Stored before super().__init__(): _load_data (which reads the CSV and
        # builds the per-grasp index) runs *inside* the base __init__.
        self.stable_grasp_csv = stable_grasp_csv
        # Bimanual mode: emit one clip per constant-grasp-count segment (>=1 hand
        # grasping) between consecutive left/right grasp transition points, rather
        # than one clip per raw CSV grasp row.  ``min_len`` drops segments shorter
        # than this (in frames); 0 keeps every non-empty segment.
        self.bimanual = bimanual
        self.min_len = int(min_len)
        super().__init__(*args, **kwargs)

    # --------------------------------------------------------- subclass hooks
    def _sg_split_mask(self, data, split):
        raise NotImplementedError

    def _sg_load_arrays(self, data, mask):
        raise NotImplementedError

    def _sg_seq_key(self, seq_idx):
        raise NotImplementedError

    def _sg_obj_name(self, seq_idx):
        raise NotImplementedError

    def _sg_x_object(self, seq_idx, a, b):
        """Object x-data window ``[a, b)`` for sequence ``seq_idx``."""
        return np.asarray(self.x_obj[seq_idx][a:b], dtype=np.float32)

    def _sg_text(self, seq_idx, is_lhand, is_rhand):
        """Natural-language description for a grasp, or ``None``.

        ``is_lhand`` / ``is_rhand`` are the **grasping** hand's flags (exactly one
        is True for a stable grasp).  The default returns ``None``; subclasses
        override to build a prompt (e.g. via ``process_text``).
        """
        return None

    # ------------------------------------------------------------- data load
    def _load_data(self, data, task, split):
        mask = self._sg_split_mask(data, split)
        self._sg_load_arrays(data, mask)

        # Base expects these to exist; this dataset's item derives the hand from
        # the CSV `side` column instead of the per-sequence flags.
        self.lcf_idx = self.lcov_idx = self.lchj_idx = self.ldist_value = None
        self.rcf_idx = self.rcov_idx = self.rchj_idx = self.rdist_value = None

        # Flat per-grasp index: one entry per clip that lands in this split.
        # ``hand_side`` is "left"/"right" (single-hand grasp) or "both" (bimanual
        # count-2 segment); ``is_lhand``/``is_rhand`` flag which hands are active.
        self.stable_grasps = {
            "seq_idx": [], "start_frame": [], "end_frame": [],
            "hand_side": [], "obj_name": [], "is_lhand": [], "is_rhand": [],
        }
        self._build_index()

    def _build_index(self):
        """Populate ``self.stable_grasps`` (the flat clip index).

        Default: bimanual segments when ``self.bimanual`` else one clip per CSV
        row.  Subclasses (e.g. segmented-motion) override to build a different
        set of clips into the same flat index.
        """
        if self.bimanual:
            self._build_bimanual_stable_grasps()
        else:
            self._build_stable_grasps()

    def _build_stable_grasps(self):
        """Read the stable-grasp CSV and flatten its rows into a per-grasp index."""
        from hoidata.preprocess.preprocess_arctic import load_stable_grasps

        if not self.stable_grasp_csv:
            raise ValueError(
                f"{type(self).__name__} requires `stable_grasp_csv` "
                "(path to the stable-grasp CSV).")

        grasps = load_stable_grasps(self.stable_grasp_csv)

        sg = self.stable_grasps
        for seq_idx in range(len(self.nframes)):
            key = self._sg_seq_key(seq_idx)
            if key not in grasps:
                continue
            N = int(self.nframes[seq_idx])
            obj_name = self._sg_obj_name(seq_idx)
            for side in ("left", "right"):
                for s, e in grasps[key][side]:
                    s = max(0, min(int(s), N))
                    e = max(0, min(int(e), N))
                    if e <= s:
                        continue
                    sg["seq_idx"].append(seq_idx)
                    sg["start_frame"].append(s)
                    sg["end_frame"].append(e)
                    sg["hand_side"].append(side)
                    sg["obj_name"].append(obj_name)
                    sg["is_lhand"].append(side == "left")
                    sg["is_rhand"].append(side == "right")

    def _build_bimanual_stable_grasps(self):
        """Flatten the CSV into per-segment bimanual stable-grasp clips.

        A *bimanual stable grasp* considers both hands' grasp intervals together.
        Every left/right grasp start/end is a **transition point**; between two
        consecutive transition points the set of grasping hands is constant, so
        each such segment has a fixed grasp count (0, 1 or 2).  One clip is
        emitted per segment where **at least one hand** is in stable grasp
        (count >= 1) and the segment is at least ``min_len`` frames long.  A
        count-2 segment is tagged ``hand_side="both"``; a count-1 segment carries
        the single grasping hand.

        Touching / overlapping same-hand intervals are coalesced first so an
        abutting boundary (``[a,m)+[m,b)``) does not create a spurious zero-width
        release+regrasp transition point.
        """
        from hoidata.preprocess.preprocess_arctic import load_stable_grasps

        if not self.stable_grasp_csv:
            raise ValueError(
                f"{type(self).__name__} requires `stable_grasp_csv` "
                "(path to the stable-grasp CSV).")

        grasps = load_stable_grasps(self.stable_grasp_csv)

        def coalesce(segs, N):
            iv = sorted((max(0, min(int(s), N)), max(0, min(int(e), N)))
                        for s, e in segs)
            iv = [(s, e) for s, e in iv if e > s]
            out = []
            for s, e in iv:
                if out and s <= out[-1][1]:
                    out[-1] = (out[-1][0], max(out[-1][1], e))
                else:
                    out.append((s, e))
            return out

        sg = self.stable_grasps
        for seq_idx in range(len(self.nframes)):
            key = self._sg_seq_key(seq_idx)
            if key not in grasps:
                continue
            N = int(self.nframes[seq_idx])
            obj_name = self._sg_obj_name(seq_idx)
            left = coalesce(grasps[key]["left"], N)
            right = coalesce(grasps[key]["right"], N)
            if not left and not right:
                continue

            # Transition points: every grasp start/end of either hand, sorted &
            # deduped.  Segments live between consecutive points.
            points = sorted({p for s, e in left + right for p in (s, e)})

            for a, b in zip(points[:-1], points[1:]):
                if b - a < self.min_len:
                    continue
                mid = (a + b) / 2.0  # any interior frame: coverage is constant here
                is_l = any(s <= mid < e for s, e in left)
                is_r = any(s <= mid < e for s, e in right)
                count = int(is_l) + int(is_r)
                if count < 1:            # keep only segments with >=1 hand grasping
                    continue
                side = "both" if count == 2 else ("left" if is_l else "right")
                sg["seq_idx"].append(seq_idx)
                sg["start_frame"].append(a)
                sg["end_frame"].append(b)
                sg["hand_side"].append(side)
                sg["obj_name"].append(obj_name)
                sg["is_lhand"].append(is_l)
                sg["is_rhand"].append(is_r)

    # ------------------------------------------------------------------ mechanics
    def __len__(self):
        return len(self.stable_grasps["seq_idx"])

    def __getitem__(self, index):
        """Return one stable-grasp clip (raw hand(s) + object x-data).

        Unimanual rows have a single grasping hand: ``x_hand`` is that hand's
        window.  Bimanual "both" rows have two: ``x_hand`` is None and each hand
        is exposed separately.  ``x_lhand`` / ``x_rhand`` always carry the active
        hand's window (or None if that hand is not grasping in this clip).
        """
        sg = self.stable_grasps
        seq_idx = sg["seq_idx"][index]
        a = sg["start_frame"][index]
        b = sg["end_frame"][index]
        is_lhand = bool(sg["is_lhand"][index])
        is_rhand = bool(sg["is_rhand"][index])

        x_lhand = (np.asarray(self.x_lhand[seq_idx][a:b], dtype=np.float32)
                   if is_lhand else None)
        x_rhand = (np.asarray(self.x_rhand[seq_idx][a:b], dtype=np.float32)
                   if is_rhand else None)
        # x_hand: the single grasping hand's window for a one-hand clip (backward
        # compatible); None when both hands grasp (use x_lhand / x_rhand).
        x_hand = x_lhand if (is_lhand and not is_rhand) else (
            x_rhand if (is_rhand and not is_lhand) else None)

        item = {
            "x_hand":   x_hand,                                 # (T, 99) or None
            "x_lhand":  x_lhand,                                # (T, 99) or None
            "x_rhand":  x_rhand,                                # (T, 99) or None
            "x_object": self._sg_x_object(seq_idx, a, b),       # (T, 9|10)
            "hand_side": str(sg["hand_side"][index]),           # left|right|both
            "is_lhand": is_lhand,
            "is_rhand": is_rhand,
            "obj_name": str(sg["obj_name"][index]),
            "n_frames": b - a,
        }
        # Natural-language description, built for the grasping hand(s)
        # (is_lhand/is_rhand above).  None when no descriptions are loaded or the
        # subclass doesn't provide text.
        text = self._sg_text(seq_idx, is_lhand, is_rhand)
        if text is not None:
            item["text"] = text
        return item

    # ------------------------------------------------------------------- hooks
    # Indexed per grasp, not per sequence (mirrors the per-grasp __getitem__).
    def _object_name(self, index):
        return self.stable_grasps["obj_name"][index]

    def _text(self, index, is_lhand, is_rhand):
        raise NotImplementedError("StableGraspMotionDataset has no text prompts.")

    def _slice_obj(self, index, init_frame, nframes):
        raise NotImplementedError("StableGraspMotionDataset uses its own __getitem__.")

    def _obj_delta(self, x_obj, nframes, max_nframes):
        raise NotImplementedError("StableGraspMotionDataset uses its own __getitem__.")

    def _object_pc(self, object_name, item):
        raise NotImplementedError("StableGraspMotionDataset uses its own __getitem__.")

    def _obj_traj_from_xobj(self, x_obj, nframes):
        raise NotImplementedError("StableGraspMotionDataset uses its own __getitem__.")