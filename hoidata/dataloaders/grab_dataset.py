"""GRAB motion dataset (rigid objects).

Subclass of the shared ``MotionDatasetBase`` / ``MotionDataModuleBase`` in
``common/data/action_dataset.py``; only the GRAB-specific data loading and the
object hooks live here.
"""

import numpy as np

from .action_dataset import MotionDatasetBase, MotionDataModuleBase
from hoidata.utils.proc import pc_normalize, proc_numpy, proc_torch_cuda, process_text
from hoidata.constants.grab_constants import grab_val_obj_name, grab_test_obj_name, grab_obj_name


class MotionGRAB(MotionDatasetBase):
    """Forked from Text2HOI.  Rigid GRAB objects; object x-data is (T, 9)."""

    all_obj_name = grab_obj_name
    val_obj_name = grab_val_obj_name
    test_obj_name = grab_test_obj_name

    def _load_data(self, data, task, split):
        if task == "action_classify":
            split_objs = self._split_objs(split)
            all_obj_names = data["obj_name"]
            mask = np.array([n in split_objs for n in all_obj_names])

            self.obj_name    = all_obj_names[mask]
            self.action_name = data["action_name"][mask]
            self.is_lhand    = data["is_lhand"][mask]
            self.is_rhand    = data["is_rhand"][mask]
            self.nframes     = data["nframes"][mask]
            self.x_lhand     = data["x_lhand"][mask]
            self.x_rhand     = data["x_rhand"][mask]
            self.sbj_id      = data["sbj_id"][mask] if "sbj_id" in data else None
            # remaining attributes unused in action_classify task
            self.proc_obj_name = self.x_obj = None
            self.lcf_idx = self.lcov_idx = self.lchj_idx = self.ldist_value = None
            self.rcf_idx = self.rcov_idx = self.rchj_idx = self.rdist_value = None
        elif task == "evaluate":
            # Physics-metric sanity check: reuse the same obj-name based split
            # selection as action_classify, but load the full generation
            # attribute set (x_obj, proc_obj_name, ...).
            split_objs = self._split_objs(split)
            all_obj_names = data["obj_name"]
            mask = np.array([n in split_objs for n in all_obj_names])

            self.obj_name      = all_obj_names[mask]
            self.proc_obj_name = data["proc_obj_name"][mask]
            self.x_lhand       = data["x_lhand"][mask]
            self.x_rhand       = data["x_rhand"][mask]
            self.x_obj         = data["x_obj"][mask]
            self.is_lhand      = data["is_lhand"][mask]
            self.is_rhand      = data["is_rhand"][mask]
            self.action_name   = data["action_name"][mask]
            self.nframes       = data["nframes"][mask]
            self.sbj_id        = data["sbj_id"][mask] if "sbj_id" in data else None
            # Full raw filename (no ext) per sequence — unique, unlike action_name
            # (the motion intent).  Used to build a distinguishable stable-grasp key.
            self.seq_name      = data["seq_name"][mask] if "seq_name" in data else None
            # Optional stable-grasp intervals (present only when the CSV was wired
            # into preprocessing); consumed by the web viewer's grasp tracks.
            self.grasps        = data["grasps"][mask] if "grasps" in data else None
            # contact attributes not needed for the physics metrics
            self.lcf_idx = self.lcov_idx = self.lchj_idx = self.ldist_value = None
            self.rcf_idx = self.rcov_idx = self.rchj_idx = self.rdist_value = None
        else:
            # Train uses every clip; val/test restrict to their held-out objects
            # so reconstruction can be inspected on unseen objects.
            all_obj_names = data["obj_name"]
            if split in ("val", "test"):
                split_objs = self._split_objs(split)
                mask = np.array([n in split_objs for n in all_obj_names])
            else:
                mask = np.ones(len(all_obj_names), dtype=bool)
            self.obj_name = all_obj_names[mask]
            self.proc_obj_name = data["proc_obj_name"][mask]
            self.x_lhand = data["x_lhand"][mask]
            self.x_rhand = data["x_rhand"][mask]
            self.x_obj = data["x_obj"][mask]
            self.lcf_idx = data["lcf_idx"][mask]
            self.lcov_idx = data["lcov_idx"][mask]
            self.lchj_idx = data["lchj_idx"][mask]
            self.ldist_value = data["ldist_value"][mask]
            self.rcf_idx = data["rcf_idx"][mask]
            self.rcov_idx = data["rcov_idx"][mask]
            self.rchj_idx = data["rchj_idx"][mask]
            self.rdist_value = data["rdist_value"][mask]
            self.is_lhand = data["is_lhand"][mask]
            self.is_rhand = data["is_rhand"][mask]
            self.action_name = data["action_name"][mask]
            self.nframes = data["nframes"][mask]
            self.sbj_id = data["sbj_id"][mask] if "sbj_id" in data else None

    # ------------------------------------------------------------------ hooks
    def _object_name(self, index):
        # object_model lookup uses the raw obj name (also what the original
        # generation item exposed as item["object_name"]).
        return self.obj_name[index]

    def _text(self, index, is_lhand, is_rhand):
        return process_text(
            self.action_name[index], self.proc_obj_name[index],
            is_lhand, is_rhand, self.text_description,
        )

    def _slice_obj(self, index, init_frame, nframes):
        # Read slice_len = nframes (+1 successor) frames so the last delta is real;
        # the caller stores only the first max_nframes as the absolute stream.
        return self.x_obj[index][init_frame:init_frame + self._slice_len]

    def _obj_delta(self, x_obj, nframes, max_nframes):
        # Rigid object: trans(3) + rot6d(6) → standard x-data delta over the
        # extended window (x_obj has slice_len rows; _to_delta keeps max_nframes).
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
        # x_obj is (T, 9) = trans(3) + rot6d(6) → trans + rotvec.
        from hoidata.utils.rot import rot6d_to_axis_angle
        obj_trans = x_obj[:, :3]
        obj_rotvec = proc_numpy(
            rot6d_to_axis_angle(proc_torch_cuda(x_obj[:, 3:]))
        )
        # Use the inverse object rotation R^{-1}: for an axis-angle (rotvec)
        # representation the inverse is simply the negated vector.
        obj_rotvec = -obj_rotvec
        obj_traj = np.concatenate([obj_trans, obj_rotvec], axis=1)  # (T, 6)
        return {"obj_traj": obj_traj.astype(np.float32)}


class TransitionMotionGRAB(MotionGRAB):
    """GRAB transition-motion dataset (rigid objects).

    A *transition motion* is the motion **between** stable grasps — the
    reach / re-grasp / retreat segments.  Stable-grasp intervals are already
    stored per sequence in ``data.npz`` (``grasps``, produced by the stable-grasp
    finder): ``{"left": [[s,e),...], "right": [[s,e),...]}``, 0-indexed, sorted by
    start, non-overlapping.  From those, per sequence and **per hand
    independently**, we derive contiguous transition windows and flatten them into
    ``self.transition_motions`` so the dataset is indexed **per transition**, not
    per sequence.  Each ``__getitem__`` poses the single active hand over one
    window ``[start, end)`` (like ``_getitem_full`` but sliced).

    Transition types (which stable-grasp boundary the window touches):
      - ``"01"``: window **ends** at a grasp *start* (preceding frames non-stable).
      - ``"10"``: window **starts** at a grasp *end* (following frames non-stable).
      - ``"11"``: a short between-grasps gap kept whole (touches a grasp end AND
        the next grasp start; only when the gap length ``<= max_len``).

    Window derivation, per side (grasps ``[s_k, e_k)`` sorted, ``N`` = nframes):
      - leading gap  ``[0, s_0)``            → ``"01"`` (anchor = end ``s_0``)
      - trailing gap ``[e_last, N)``          → ``"10"`` (anchor = start ``e_last``)
      - between ``[e_k, s_{k+1})`` (len ``L``):
          * ``L <= max_len`` → one ``"11"`` window ``[e_k, s_{k+1})``
          * ``L >  max_len`` → two windows (overlap allowed): ``"10"`` at
            ``[e_k, e_k+max_len)`` and ``"01"`` at ``[s_{k+1}-max_len, s_{k+1})``
      - leading/trailing gaps longer than ``max_len`` are clipped to ``max_len``
        keeping the anchored end.
    Windows shorter than ``min_len`` (after clipping) are discarded.  ``min_len``
    and ``max_len`` are config hyperparameters (``transition_{min,max}_len``).
    """

    def __init__(self, *args, min_len=10, max_len=100, aug_p=0.0, **kwargs):
        # Stored before super().__init__() because _load_data (which builds the
        # transition index) runs *inside* the base __init__.
        self.min_len = int(min_len)
        self.max_len = int(max_len)
        # Probability of applying the random-trim augmentation on "01"/"10"
        # windows (trims the non-anchored end, keeping the stable-grasp anchor).
        # 0.0 → deterministic (no augmentation), e.g. for visualization/eval.
        self.aug_p = float(aug_p)
        super().__init__(*args, **kwargs)
        # This class always poses full-resolution sub-windows; never the random
        # windowed path.  Force it defensively (our own __getitem__ ignores it,
        # but keep the invariant explicit).
        self.max_nframes = -1

    def _load_data(self, data, task, split):
        all_obj_names = data["obj_name"]
        if split in ("val", "test"):
            split_objs = self._split_objs(split)
            mask = np.array([n in split_objs for n in all_obj_names])
        else:
            mask = np.ones(len(all_obj_names), dtype=bool)
        self.obj_name = all_obj_names[mask]
        self.proc_obj_name = data["proc_obj_name"][mask]
        self.x_lhand = data["x_lhand"][mask]
        self.x_rhand = data["x_rhand"][mask]
        self.x_obj = data["x_obj"][mask]
        self.lcf_idx = data["lcf_idx"][mask]
        self.lcov_idx = data["lcov_idx"][mask]
        self.lchj_idx = data["lchj_idx"][mask]
        self.ldist_value = data["ldist_value"][mask]
        self.rcf_idx = data["rcf_idx"][mask]
        self.rcov_idx = data["rcov_idx"][mask]
        self.rchj_idx = data["rchj_idx"][mask]
        self.rdist_value = data["rdist_value"][mask]
        self.is_lhand = data["is_lhand"][mask]
        self.is_rhand = data["is_rhand"][mask]
        self.action_name = data["action_name"][mask]
        self.nframes = data["nframes"][mask]
        self.sbj_id = data["sbj_id"][mask] if "sbj_id" in data else None
        self.stable_grasps = data["grasps"][mask]

        self.transition_motions = {
            "seq_idx": [],
            "start_frame": [],
            "end_frame": [],
            "hand_side": [],
            "obj_name": [],
            "transition_type": [],  # "01", "10", "11"
        }
        self._build_transitions()

    # ---------------------------------------------------------- transition index
    def _build_transitions(self):
        """Populate ``self.transition_motions`` from ``self.stable_grasps``."""
        tm = self.transition_motions
        for seq_idx in range(len(self.stable_grasps)):
            N = int(self.nframes[seq_idx])
            obj_name = str(self.obj_name[seq_idx])
            grasps = self.stable_grasps[seq_idx]
            for side in ("left", "right"):
                segs = grasps.get(side, []) if grasps is not None else []
                for a, b, ttype in self._side_transitions(segs, N):
                    tm["seq_idx"].append(seq_idx)
                    tm["start_frame"].append(a)
                    tm["end_frame"].append(b)
                    tm["hand_side"].append(side)
                    tm["obj_name"].append(obj_name)
                    tm["transition_type"].append(ttype)

    def _side_transitions(self, segs, N):
        """Yield ``(start, end, type)`` transition windows for one side.

        ``segs`` are this side's stable grasps ``[[s,e), ...]`` (sorted, clamped);
        ``N`` is the sequence length.  Applies the leading/between/trailing rules
        plus the ``min_len``/``max_len`` clip/split/discard logic.
        """
        min_len, max_len = self.min_len, self.max_len
        if not segs:
            return

        def clamp(x):
            return max(0, min(int(x), N))

        # Leading gap (before first grasp): anchored at the grasp start → "01".
        s0 = clamp(segs[0][0])
        a = 0 if (s0 - 0) <= max_len else s0 - max_len
        if s0 - a >= min_len:
            yield a, s0, "01"

        # Between adjacent grasps.
        for k in range(len(segs) - 1):
            e_k = clamp(segs[k][1])
            s_next = clamp(segs[k + 1][0])
            L = s_next - e_k
            if L < min_len:
                continue
            if L <= max_len:
                # Short gap: one window spanning both anchors → "11".
                yield e_k, s_next, "11"
            else:
                # Long gap: split into two max_len windows (overlap allowed).
                yield e_k, e_k + max_len, "10"            # anchored at former end
                yield s_next - max_len, s_next, "01"      # anchored at next start

        # Trailing gap (after last grasp): anchored at the grasp end → "10".
        e_last = clamp(segs[-1][1])
        b = N if (N - e_last) <= max_len else e_last + max_len
        if b - e_last >= min_len:
            yield e_last, b, "10"

    # ------------------------------------------------------------------ mechanics
    def __len__(self):
        return len(self.transition_motions["seq_idx"])

    def __getitem__(self, index):
        """Return one transition window: raw hand + object x-data.

        Neither hand nor object is posed here — the model consumes the raw
        ``x_hand`` (trans + rot6d) and ``x_obj`` (trans + rot6d) windows and
        poses them itself.
        """
        tm = self.transition_motions
        seq_idx = tm["seq_idx"][index]
        ttype = tm["transition_type"][index]
        a = tm["start_frame"][index]
        b = tm["end_frame"][index]
        if ttype in ("01", "10"):
            # Random-trim augmentation: shrink the non-anchored end while keeping
            # the stable-grasp anchor and at least min_len frames.  slack is the
            # max number of frames we may trim; skip when there's none (window
            # already at min_len) so np.random.randint gets a positive range.
            slack = (b - a) - self.min_len
            if slack > 0 and np.random.rand() < self.aug_p:
                trim = np.random.randint(0, slack + 1)  # 0..slack inclusive
                if ttype == "01":
                    a = a + trim   # keep the end anchor (grasp start)
                else:
                    b = b - trim   # keep the start anchor (grasp end)

        side = tm["hand_side"][index]
        Tt = b - a

        x_hand = (self.x_lhand if side == "left" else self.x_rhand)[seq_idx][a:b]
        x_hand = np.asarray(x_hand, dtype=np.float32)   # (Tt, D) trans + rot6d
        x_obj = np.asarray(self.x_obj[seq_idx][a:b], dtype=np.float32)  # (Tt, 9) trans + rot6d

        item = {
            "x_hand":      x_hand,                        # (Tt, D) raw hand x-data
            "x_obj":       x_obj,                         # (Tt, 9) raw object x-data
            "hand_sides":  [side],
            "obj_name":    str(self.obj_name[seq_idx]),
            "nframes":     Tt,
            "transition_type": ttype,
            "seq_idx":     int(seq_idx),
            "start_frame": int(a),
            "end_frame":   int(b),
            "is_lhand":    side == "left",
            "is_rhand":    side == "right",
        }
        return item

    # -------------------------------------------------- per-transition metadata
    # These make the dataset indexable per transition by tools that expect the
    # per-sequence attributes (e.g. the web viewer's sequence list): each returns
    # info for transition ``index`` rather than a whole sequence.
    def _object_name(self, index):
        return self.transition_motions["obj_name"][index]

    def transition_label(self, index):
        """Human-readable label for transition ``index`` (viewer dropdown / title).

        e.g. ``airplane_lift/left/01`` — object of the parent sequence, the hand,
        and the transition type.
        """
        tm = self.transition_motions
        seq_idx = tm["seq_idx"][index]
        return (f'{tm["obj_name"][index]}/{tm["hand_side"][index]}/'
                f'{tm["transition_type"][index]}@{seq_idx}')

    def transition_nframes(self, index):
        tm = self.transition_motions
        return tm["end_frame"][index] - tm["start_frame"][index]


class MotionGRABDataModule(MotionDataModuleBase):
    """LightningDataModule wrapping MotionGRAB."""

    dataset_cls = MotionGRAB

    def prepare_data(self):
        """Run any missing GRAB preprocessing steps before setup.

        Detects each preprocessed file independently and only regenerates the
        missing ones (see ``_run_missing_preprocessing``).
        """
        from hoidata.preprocess import preprocess_grab
        self._run_missing_preprocessing(preprocess_grab)


class TransitionMotionGRABDataModule(MotionGRABDataModule):
    """LightningDataModule wrapping :class:`TransitionMotionGRAB`.

    Same preprocessing / split machinery as :class:`MotionGRABDataModule`, but the
    dataset is indexed **per transition** (see ``TransitionMotionGRAB``) and the
    transition min/max window lengths come from the config
    (``transition_min_len`` / ``transition_max_len``).
    """

    dataset_cls = TransitionMotionGRAB

    def _make_dataset(self, split: str):
        dc = self.cfg
        # Augmentation is a train-time only knob: keep val/test deterministic.
        aug_p = dc.get("aug_p", 0.0) if split == "train" else 0.0
        return self.dataset_cls(
            data_path=dc.data_path,
            data_obj_pc_path=dc.data_obj_pc_path,
            text_json=dc.text_json,
            max_nframes=-1,                 # this dataset always poses full windows
            data_ratio=dc.get("data_ratio", 1.0),
            task=self.task,
            split=split,
            lhand_layer=self.lhand_layer,
            rhand_layer=self.rhand_layer,
            subject_layers=self.subject_layers,
            min_len=dc.get("transition_min_len", 10),
            max_len=dc.get("transition_max_len", 100),
            aug_p=aug_p,
        )

