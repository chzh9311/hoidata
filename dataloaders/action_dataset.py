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

``MotionGRAB`` / ``MotionGRABDataModule`` are re-exported from this module (see the
bottom of the file) so existing ``from common.data.action_dataset import
MotionGRABDataModule`` imports keep working after the refactor.
"""

import time
import json

import numpy as np
from torch.utils.data import Dataset, DataLoader
import lightning as L

import torch
import pickle

from utils.frame import get_valid_mask
from utils.proc import get_contact_map, process_dist_map, transform_xdata_to_joints
from utils.rot import rot6d_to_rotmat, rotmat_to_rot6d


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

    rot6d_t = torch.from_numpy(rot6d).reshape(T * n_rot, 6)
    rotmat = rot6d_to_rotmat(rot6d_t).reshape(T, n_rot, 3, 3)  # (T, n_rot, 3, 3)
    R_cur = rotmat[:-1]                                        # R_t
    R_nxt = rotmat[1:]                                         # R_{t+1}
    R_delta = R_nxt @ R_cur.transpose(-1, -2)                  # world-frame: R_{t+1} @ R_t^T
    rot6d_delta = rotmat_to_rot6d(R_delta.reshape(-1, 3, 3)).reshape(T - 1, n_rot * 6)
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
            self._load_data(data, task, split)

        if task != "action_classify":
            if task != "vae_training":
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
        from utils.proc import transform_xdata_to_verts, transform_xdata_to_joints

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

        if is_lhand is not None:
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