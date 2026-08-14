"""GRAB motion dataset (rigid objects).

Subclass of the shared ``MotionDatasetBase`` / ``MotionDataModuleBase`` in
``common/data/action_dataset.py``; only the GRAB-specific data loading and the
object hooks live here.
"""

import numpy as np

from .action_dataset import (
    MotionDatasetBase, MotionDataModuleBase, StableGraspMotionDataset)
from hoidata.utils.proc import (
    pc_normalize, proc_numpy, proc_torch_cuda, process_text, process_obj_result)
from hoidata.constants.grab_constants import grab_val_obj_name, grab_test_obj_name, grab_obj_name
from hoidata.utils.contact_utils import calculate_contact_capsule


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
        from pytorch3d.transforms import matrix_to_axis_angle, rotation_6d_to_matrix
        obj_trans = x_obj[:, :3]
        obj_rotvec = proc_numpy(
            matrix_to_axis_angle(rotation_6d_to_matrix(proc_torch_cuda(x_obj[:, 3:])))
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
        ``x_hand`` (trans + rot6d) and ``x_object`` (trans + rot6d) windows and
        poses them itself.  Key names match ``StableGraspMotionDataset``
        (``x_hand`` / ``x_object`` / ``n_frames``) so both clip types can be
        consumed uniformly.
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
        x_object = np.asarray(self.x_obj[seq_idx][a:b], dtype=np.float32)  # (Tt, 9)

        item = {
            "x_hand":      x_hand,                        # (Tt, D) raw hand x-data
            "x_object":    x_object,                      # (Tt, 9) raw object x-data
            "hand_sides":  [side],
            "obj_name":    str(self.obj_name[seq_idx]),
            "n_frames":    Tt,
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


class BimanualTransitionMotionGRAB(TransitionMotionGRAB):
    """GRAB **bimanual** transition-motion dataset (rigid objects).

    Unlike :class:`TransitionMotionGRAB` (which treats each hand independently and
    emits one window per hand), here the two hands are considered **together**: the
    grasp state at any frame is the *number of hands currently in a stable grasp*
    (``0``, ``1`` or ``2``).  The left/right stable grasps stored per sequence are
    merged by sequence id into a single grasp-count timeline; a transition window is
    a **dwell at a constant grasp count** ``c`` between two consecutive stable-grasp
    edges, and it is labelled by the neighbouring (higher) grasp state.  Crucially,
    dwells at ``c == 2`` — where **both hands are mid stable grasp** — are
    **excluded**: no transition window overlaps such a frame.  The transition is
    always *the motion into or out of* a grasp, never the held-grasp itself.

    Transition types encode ``from``→``to`` grasp count (the count on the two sides
    of the window):
      - ``"01"`` 0→1: idle → one hand grasps.  Ends at the grasp start (front free).
      - ``"02"`` 0→2: idle → **both** hands grasp near-simultaneously (the two grasp
        *starts* within ``max_len``).  Ends at the later grasp start (front free).
      - ``"10"`` 1→0: one hand releases → idle.  Starts at the grasp end (back free).
      - ``"12"`` 1→2: one hand held while the other grasps.  Ends at that grasp
        start (front free); the held hand stays grasped throughout.
      - ``"20"`` 2→0: both hands release near-simultaneously (the two grasp *ends*
        within ``max_len``).  Starts at the earlier grasp end (back free).
      - ``"21"`` 2→1: one hand releases while the other stays grasped.  Starts at
        that release (back free); the held hand stays grasped throughout.
      - ``"11"`` — a count-0 gap between two single-hand grasps: a hand releases and
        then a hand grasps with nothing grasped in between (cross-hand handoff
        ``LE→RS``/``RE→LS``, or same-hand re-grasp ``LE→LS``/``RE→RS``).  This single
        clip **replaces** the ``"10"`` + ``"01"`` pair such a gap would otherwise
        yield.  ``hand_side`` names the mover for a same-hand re-grasp, else ``"both"``.
      - ``"22"`` — the count-2 analogue: a count-1 dwell between two both-hand
        grasps.  One hand stays grasped throughout while the other releases and
        re-grasps; **replaces** the ``"21"`` + ``"12"`` pair.  ``hand_side`` is the
        moving hand.
      Both ``"11"`` and ``"22"`` carry two valid anchors (the release edge and the
      grasp edge).

    Window derivation (see :meth:`_bimanual_transitions`).  Same-hand touching /
    overlapping grasps are coalesced first (a shared frame is one continuous grasp,
    not a release+regrasp).  Grasp *edges* (+1 start / -1 end) are collapsed into
    count *changes*, with near-simultaneous same-direction edges (within ``max_len``)
    merging into a single ``0<->2`` change anchored at the later edge.  Each dwell
    ``[L, R)`` between changes with inner count ``c < 2``:
      * if **both** neighbours equal ``c + 1`` (a symmetric release→grasp hold),
        emit one merged ``f"{c+1}{c+1}"`` clip (``"11"`` or ``"22"``) spanning the
        whole dwell, with two anchors;
      * otherwise emit, toward each strictly-higher-count neighbour, a directed
        window: an *increasing* type ``{c}{cR}`` ending at ``R`` (front trimmable)
        and/or a *decreasing* type ``{cL}{c}`` starting at ``L`` (back trimmable) —
        e.g. leading/trailing gaps, or a ``0``-gap bordered by a grasp on one side
        and a *higher* (count-2) grasp on the other.
    ``min_len`` discards and ``max_len`` clips (keeping the anchored, grasp-side
    end); the random-trim augmentation trims the free end.  For a hold type the
    augmentation additionally re-picks the anchor (release vs grasp edge) at random,
    so the clip count stays fixed.

    Every emitted window carries ``hand_side`` = the hand whose grasp edge anchors
    the transition: ``"left"``/``"right"`` for the single hand that (dis)engaged,
    or ``"both"`` for the near-simultaneous ``"02"``/``"20"``.  Both hands are still
    posed when the parent sequence has both (see :meth:`__getitem__`); ``hand_side``
    only records which hand *drives* the transition.
    """

    #: Symmetric "hold" transition types: a dwell whose grasp count is one lower
    #: than the equal count on both sides — a release immediately followed by a
    #: (same- or other-hand) grasp.  ``"11"`` is a count-0 gap between two
    #: single-hand grasps (cross-hand handoff or same-hand re-grasp); ``"22"`` is a
    #: count-1 dwell between two both-hand grasps (one hand held, the other releases
    #: and re-grasps).  These carry two valid anchors (release edge / grasp edge)
    #: and are the only types the augmentation may re-anchor.
    HOLD_TYPES = ("11", "22")

    def _build_transitions(self):
        """Populate ``self.transition_motions`` from merged left+right grasps.

        Adds two per-item fields beyond the base dict — ``gap_lo`` / ``gap_hi`` —
        holding the **raw** (min_len-passed, un-clipped) dwell boundaries.  For
        hold items (:data:`HOLD_TYPES`) these bracket the full dwell so
        ``__getitem__`` can pick either the release or the grasp edge as the anchor
        at augmentation time; for ordinary items they equal ``start_frame`` /
        ``end_frame``.
        """
        tm = self.transition_motions
        tm.setdefault("gap_lo", [])
        tm.setdefault("gap_hi", [])
        for seq_idx in range(len(self.stable_grasps)):
            N = int(self.nframes[seq_idx])
            obj_name = str(self.obj_name[seq_idx])
            grasps = self.stable_grasps[seq_idx]
            for a, b, ttype, side, gap in self._bimanual_transitions(grasps, N):
                tm["seq_idx"].append(seq_idx)
                tm["start_frame"].append(a)
                tm["end_frame"].append(b)
                tm["hand_side"].append(side)
                tm["obj_name"].append(obj_name)
                tm["transition_type"].append(ttype)
                tm["gap_lo"].append(gap[0])
                tm["gap_hi"].append(gap[1])

    def _bimanual_transitions(self, grasps, N):
        """Yield ``(start, end, type, side)`` merged-hand transition windows.

        ``grasps`` is ``{"left": [[s,e), ...], "right": [[s,e), ...]}`` (per-side
        sorted, clamped intervals).  We build a chronological list of grasp
        *edges* — ``(frame, delta, side)`` with ``delta`` +1 at a grasp start and
        -1 at a grasp end — then walk them, tracking the running grasp count.  Each
        contiguous run of edges within ``max_len`` of one another that shares the
        same monotone direction (all starts, or all ends) collapses into one state
        change; that change spans a count step of 1 (one hand) or 2 (both hands
        near-simultaneously), anchored at the *later* edge frame.  A window is the
        gap from the previous change frame (or 0) to this change's anchor frame.
        """
        if grasps is None:
            return

        def clamp(x):
            return max(0, min(int(x), N))

        def coalesce(segs):
            """Merge touching / overlapping same-hand intervals ``[s, e)``.

            Abutting grasps (``[a, m)`` then ``[m, b)``) or overlapping ones
            describe the *same* continuous grasp for the count timeline; without
            merging, the shared frame would read as a spurious release+regrasp
            (count flicker 2→1→2) and emit a zero-length window.
            """
            iv = sorted((clamp(s), clamp(e)) for s, e in segs)
            iv = [(s, e) for s, e in iv if e > s]
            merged = []
            for s, e in iv:
                if merged and s <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], e))
                else:
                    merged.append((s, e))
            return merged

        # Chronological grasp edges: +1 at a start, -1 at an end.
        edges = []  # (frame, delta, side)
        for side in ("left", "right"):
            for s, e in coalesce(grasps.get(side, []) or []):
                edges.append((s, +1, side))
                edges.append((e, -1, side))
        if not edges:
            return
        # Sort by frame; at equal frames put ends (-1) before starts (+1) so a
        # hand releasing exactly as another grasps reads as 1→1 (two events), not a
        # spurious 1→2→1.  (delta ascending → -1 first.)
        edges.sort(key=lambda ev: (ev[0], ev[1]))

        # Collapse near-simultaneous same-direction edges into count *changes*.
        # A change is where the active-grasp count steps; edges of the same
        # direction within ``max_len`` of the group's first edge merge into one
        # change (near-simultaneous → a 0<->2 step).  Each change:
        #   (anchor_frame, from_count, to_count, sides_involved)
        # ``anchor_frame`` is the *later* edge of the group; the change takes
        # effect there.
        changes = []
        count = 0
        i = 0
        n = len(edges)
        while i < n:
            frame0, delta0, side0 = edges[i]
            from_count = count
            sides = [side0]
            anchor = frame0
            count += delta0
            j = i + 1
            while (j < n and edges[j][1] == delta0
                   and edges[j][0] - frame0 < self.max_len):
                anchor = edges[j][0]
                sides.append(edges[j][2])
                count += edges[j][1]
                j += 1
            changes.append((anchor, from_count, count, sides))
            i = j

        # A transition window is the *dwell* at a constant grasp count ``c``
        # between two consecutive changes — the run ``[L, R)`` where ``L`` is the
        # frame the count became ``c`` and ``R`` the frame it next changes.  For a
        # run with inner count ``c``, left-neighbour count ``cL`` (before ``L``)
        # and right-neighbour count ``cR`` (after ``R``), we emit the transition
        # toward whichever neighbour has the **higher** count (the grasping side);
        # its anchor is that higher-count edge and the opposite (free) end is the
        # one the random-trim augmentation may shorten.  Crucially this **excludes**
        # the both-hands-grasped dwell (``c == 2``): only runs with ``c < 2`` are
        # emitted, so no window overlaps a frame where both hands are mid-grasp.
        #
        #   * ``cR > c``  → the count *rises* into a grasp at ``R``: type
        #     ``{c}{cR}``, window **ends** at the grasp boundary ``R`` (front ``L``
        #     trimmable).  e.g. #2 [LS,RS): c=1, cR=2 → "12", ends at RS.
        #   * ``cL > c``  → the count *fell* out of a grasp at ``L``: type
        #     ``{cL}{c}``, window **starts** at the grasp boundary ``L`` (back ``R``
        #     trimmable).  e.g. #4 [LE,RE): c=1, cL=2 → "21", starts at LE.
        #
        # Both branches may fire for the same dwell: an isolated low count ``c``
        # flanked by higher-count grasps on *both* sides emits two directed windows
        # over the same span — one leaving the earlier grasp (start-anchored) and
        # one approaching the next (end-anchored) — mirroring the unimanual split.
        # When only one neighbour is higher (the other is the sequence boundary or
        # an even-lower count) a single window is emitted.
        run_starts = [0] + [c[0] for c in changes]          # L for each dwell
        run_counts = [0] + [c[2] for c in changes]          # inner count c
        run_ends = [c[0] for c in changes] + [N]            # R for each dwell
        # side that owns each change's edge(s): "left"/"right"/"both"
        change_sides = [("both" if len(c[3]) >= 2 else c[3][0]) for c in changes]

        for k in range(len(run_starts)):
            L, R, c = run_starts[k], run_ends[k], run_counts[k]
            if c >= 2 or R <= L:
                continue  # skip the both-grasped dwell and empty runs
            cL = run_counts[k - 1] if k > 0 else None       # via the change at L
            cR = run_counts[k + 1] if k + 1 < len(run_counts) else None
            # Entry side: the change at L raised us from cL→c (edge owner = the
            # hand that grasped/released at L).  Exit side: the change at R.
            entry_side = change_sides[k - 1] if k > 0 else None   # change ending at L
            exit_side = change_sides[k] if k < len(changes) else None  # change at R

            # --- Symmetric "hold" dwell: inner count ``c`` flanked on BOTH sides by
            # the SAME higher count ``c + 1`` — a hand releases at ``L`` and (the
            # same or the other hand) grasps at ``R`` with the state constant on
            # both sides.  The otherwise-emitted directed pair
            # (``{cL}{c}`` start-anchored + ``{c}{cR}`` end-anchored) is replaced by
            # ONE clip typed ``f"{c+1}{c+1}"`` spanning the whole dwell, with two
            # valid anchors (release edge ``L`` / grasp edge ``R``) picked at
            # augmentation time (see __getitem__).  Covers:
            #   * ``"11"`` (c=0): count-0 gap between two single-hand grasps —
            #     cross-hand handoff (LR/RL) or same-hand re-grasp (LL/RR).
            #   * ``"22"`` (c=1): count-1 dwell between two both-hand grasps — one
            #     hand held throughout while the other releases and re-grasps.
            if (cL is not None and cR is not None
                    and cL == c + 1 and cR == c + 1):
                ttype = f"{c + 1}{c + 1}"
                # Name the moving hand when a single hand both released and grasped
                # (same-hand re-grasp, or the moving hand of a "22"); "both" when
                # the two edges are different hands (a genuine cross-hand handoff).
                if entry_side == exit_side and entry_side in ("left", "right"):
                    side = entry_side
                else:
                    side = "both"
                win = self._clip_window(L, R, anchored="end")
                if win is not None:
                    a, b = win
                    yield a, b, ttype, side, (L, R)
                continue

            # Higher-count neighbour on the RIGHT → this run leads *into* a grasp
            # (count increasing); window ends at the grasp boundary R.
            if cR is not None and cR > c:
                ttype = f"{c}{cR}"
                side = exit_side if exit_side is not None else "both"
                win = self._clip_window(L, R, anchored="end")
                if win is not None:
                    a, b = win
                    yield a, b, ttype, side, (a, b)
            # Higher-count neighbour on the LEFT → this run leads *out of* a grasp
            # (count decreasing); window starts at the grasp boundary L.
            if cL is not None and cL > c:
                ttype = f"{cL}{c}"
                side = entry_side if entry_side is not None else "both"
                win = self._clip_window(L, R, anchored="start")
                if win is not None:
                    a, b = win
                    yield a, b, ttype, side, (a, b)
            # Leading gap [0, first grasp): cL is None, cR>c handled above.
            # Trailing gap after last grasp: cR is None, cL>c handled above.

    def _clip_window(self, a, b, anchored="end"):
        """Apply ``min_len``/``max_len`` to a window with a fixed anchor end.

        ``anchored="end"`` keeps ``b`` (the grasp boundary) and trims from the
        front; ``anchored="start"`` keeps ``a`` and trims from the back.  Returns
        ``(a, b)`` or ``None`` if the window is shorter than ``min_len``.
        """
        L = b - a
        if L < self.min_len:
            return None
        if L > self.max_len:
            if anchored == "start":
                b = a + self.max_len
            else:
                a = b - self.max_len
        return a, b

    def _window_contact(self, seq_idx, a, b, sides):
        """Per-frame, per-hand, per-object-point contact map over ``[a, b)``.

        Poses the object at every frame of the window and, **separately for each
        hand**, runs the capsule contact calculation
        (``calculate_contact_capsule``) with that hand's mesh as the query
        surface.  Returns ``(Tt, 2, O)`` float32 — the left- and right-hand
        contact maps stacked on axis 1 (index 0 = left, 1 = right).  A hand the
        sequence lacks (not in ``sides``) contributes an all-zero map.  ``O`` is
        the object point-cloud size from ``self.object_model``.
        """
        import torch
        from pytorch3d.structures import Meshes
        from pytorch3d.transforms import matrix_to_axis_angle, rotation_6d_to_matrix

        Tt = b - a
        obj_name = str(self.obj_name[seq_idx])
        _, obj_pc, obj_pc_normal, _ = self.object_model(obj_name)          # (O,3),(O,3)
        O = obj_pc.shape[0]

        # Device follows the hand MANO layer rather than forcing .cuda(): with CPU
        # layers this whole computation runs on CPU, so it is safe inside forked
        # DataLoader workers (num_workers>0), where CUDA cannot be re-initialized.
        # (Called from the main process with CUDA layers, it still runs on CUDA.)
        ref_layer = self.lhand_layer if "left" in sides else self.rhand_layer
        device = ref_layer.th_faces.device       # MANO layers register buffers, not params

        # Object verts + normals per frame.  Verts get the full rigid transform;
        # normals are only rotated (translation-invariant).
        x_obj = torch.as_tensor(
            np.asarray(self.x_obj[seq_idx][a:b], dtype=np.float32), device=device)  # (Tt,9)
        obj_pc_t = torch.as_tensor(obj_pc.astype(np.float32), device=device)        # (O,3)
        obj_norm_t = torch.as_tensor(obj_pc_normal.astype(np.float32), device=device)  # (O,3)
        obj_verts = process_obj_result(obj_pc_t, x_obj, "grab")            # (Tt,O,3)
        rotmat = rotation_6d_to_matrix(x_obj[:, 3:9]).reshape(Tt, 3, 3)    # (Tt,3,3)
        # GRAB rotation convention (see process_obj_result "grab" einsum "ij,ki->kj").
        obj_normals = torch.einsum("ni,tij->tnj", obj_norm_t, rotmat)      # (Tt,O,3)

        # One contact map per hand (left, then right); absent hands stay zero.
        per_hand = []
        for side in ("left", "right"):
            if side not in sides:
                per_hand.append(np.zeros((Tt, O), dtype=np.float32))
                continue
            layer = self.lhand_layer if side == "left" else self.rhand_layer
            xh = torch.as_tensor(
                np.asarray(
                    (self.x_lhand if side == "left" else self.x_rhand)[seq_idx][a:b],
                    dtype=np.float32),
                device=device)                                            # (Tt,D)
            # MANO FK on the layer's device (a device-following inline of
            # transform_xdata_to_verts, which would otherwise force .cuda()).
            pose = matrix_to_axis_angle(
                rotation_6d_to_matrix(xh[:, 3:].reshape(-1, 6))).reshape(Tt, 48)
            betas = torch.zeros(Tt, 10, device=device)
            verts = layer(pose, betas, xh[:, :3]).vertices                # (Tt,778,3)
            faces = layer.th_faces.to(device).unsqueeze(0).expand(Tt, -1, -1)
            normals = Meshes(verts=verts, faces=faces).verts_normals_padded()
            obj_contact, _hc, _so, _sh, _nn = calculate_contact_capsule(
                verts, normals, obj_verts, obj_normals)
            per_hand.append(
                obj_contact[:, :, 0].detach().cpu().numpy().astype(np.float32))
        return np.stack(per_hand, axis=1)                                 # (Tt,2,O)

    def _last_frame_contact(self, seq_idx, last_frame, sides):
        """Per-hand contact map of the window's **last** frame — ``(O, 2)``.

        Column 0 is the left-hand contact, column 1 the right-hand contact (a
        hand the sequence lacks contributes zeros).  Thin wrapper over
        :meth:`_window_contact` for the single last frame.
        """
        lr = self._window_contact(seq_idx, last_frame, last_frame + 1, sides)[0]  # (2,O)
        return np.ascontiguousarray(lr.T)                                # (O,2) [L,R]

    # ------------------------------------------------------------------ mechanics
    def __getitem__(self, index):
        """Return one bimanual transition window: raw hand(s) + object x-data.

        ``hand_side`` is ``"left"``/``"right"``/``"both"``; ``hand_sides`` lists the
        hand(s) actually posed.  For a one-hand transition (``01``/``10``/``12``/
        ``21``) we still include *both* hands when the sequence has both — the
        non-transitioning hand may hold the object throughout the window — but the
        ``hand_side`` field records which hand *drives* the transition.  The raw
        per-hand windows are returned **separately** as ``x_lhand`` and ``x_rhand``,
        each ``(Tt, D)`` (or ``None`` for a hand the sequence lacks); the object
        window is ``x_object`` ``(Tt, 9)`` and ``n_frames`` is ``Tt``.
        ``target_contact`` is the ``(O, 2)`` per-object-point contact map of the
        window's last frame — column 0 left-hand, column 1 right-hand (a hand the
        sequence lacks contributes zeros; see :meth:`_last_frame_contact`).
        """
        tm = self.transition_motions
        seq_idx = tm["seq_idx"][index]
        ttype = tm["transition_type"][index]
        a = tm["start_frame"][index]
        b = tm["end_frame"][index]

        if ttype in self.HOLD_TYPES:
            # Hold dwell ("11"/"22"): two valid anchors (release edge gap_lo, grasp
            # edge gap_hi).  With prob aug_p pick one at random and build a
            # max_len window from it, then random-trim the free end; otherwise
            # keep the stored default window.  Exactly one clip either way.
            if np.random.rand() < self.aug_p:
                lo, hi = tm["gap_lo"][index], tm["gap_hi"][index]
                if np.random.rand() < 0.5:            # anchor at the grasp edge hi
                    a, b = self._clip_window(lo, hi, anchored="end")
                    slack = (b - a) - self.min_len
                    if slack > 0:
                        a = a + np.random.randint(0, slack + 1)
                else:                                 # anchor at the release edge lo
                    a, b = self._clip_window(lo, hi, anchored="start")
                    slack = (b - a) - self.min_len
                    if slack > 0:
                        b = b - np.random.randint(0, slack + 1)
        else:
            # Ordinary type: keep the stable-grasp *anchor* boundary and at least
            # min_len frames, trimming the free end.  The anchor is the grasp-side
            # edge: a count-increasing type (X<Y, e.g. "01"/"12") is end-anchored →
            # trim the front; a decreasing type (X>Y, e.g. "21"/"20") is
            # start-anchored → trim the back.
            slack = (b - a) - self.min_len
            if slack > 0 and np.random.rand() < self.aug_p:
                trim = np.random.randint(0, slack + 1)
                if ttype[0] < ttype[1]:      # count increasing → end-anchored
                    a = a + trim
                else:                        # count decreasing → start-anchored
                    b = b - trim

        drive = tm["hand_side"][index]          # "left" | "right" | "both"
        Tt = b - a

        # Pose whichever hands the parent sequence actually has.
        sides = []
        if bool(self.is_lhand[seq_idx]):
            sides.append("left")
        if bool(self.is_rhand[seq_idx]):
            sides.append("right")
        if not sides:  # defensive: fall back to the driving hand(s)
            sides = ["left", "right"] if drive == "both" else [drive]

        # Separate per-hand x-data, each (Tt, D); a hand the sequence lacks is None.
        x_lhand = (np.asarray(self.x_lhand[seq_idx][a:b], dtype=np.float32)
                   if "left" in sides else None)
        x_rhand = (np.asarray(self.x_rhand[seq_idx][a:b], dtype=np.float32)
                   if "right" in sides else None)
        x_object = np.asarray(self.x_obj[seq_idx][a:b], dtype=np.float32)  # (Tt, 9)

        # Contact map over the object point cloud at the window's last frame,
        # per hand: (O, 2) with column 0 = left, column 1 = right.
        target_contact = self._last_frame_contact(seq_idx, b - 1, sides)   # (O, 2)

        item = {
            "x_lhand":     x_lhand,                   # (Tt, D) raw left-hand x-data or None
            "x_rhand":     x_rhand,                   # (Tt, D) raw right-hand x-data or None
            "x_object":    x_object,                  # (Tt, 9) raw object x-data
            "hand_sides":  sides,                     # hands actually posed
            "hand_side":   drive,                     # hand(s) driving the transition
            "obj_name":    str(self.obj_name[seq_idx]),
            "n_frames":    Tt,
            "transition_type": ttype,                 # "01","02","10","11","12","20","21","22"
            "seq_idx":     int(seq_idx),
            "start_frame": int(a),
            "end_frame":   int(b),
            "target_contact": target_contact,        # (O, 2) [L,R] contact @ last frame
            "is_lhand":    "left" in sides,
            "is_rhand":    "right" in sides,
        }
        return item




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


class BimanualTransitionMotionGRABDataModule(MotionGRABDataModule):
    """LightningDataModule wrapping :class:`BimanualTransitionMotionGRAB`.

    Same preprocessing / split machinery as :class:`TransitionMotionGRABDataModule`,
    but transitions are the merged-hand grasp-count changes (``01/02/10/12/20/21``).
    """

    dataset_cls = BimanualTransitionMotionGRAB

    def _make_dataset(self, split: str):
        dc = self.cfg
        aug_p = dc.get("aug_p", 0.0) if split == "train" else 0.0
        return self.dataset_cls(
            data_path=dc.data_path,
            data_obj_pc_path=dc.data_obj_pc_path,
            text_json=dc.text_json,
            max_nframes=-1,
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


class StableGraspGRAB(StableGraspMotionDataset):
    """GRAB stable-grasp clips — one item per stable-grasp CSV row.

    CSV rows are keyed by ``{sbj_id}/{seq_name}`` (e.g. ``s1/airplane_lift``);
    the object x-data is rigid GRAB's 9-D ``trans(3) + rot6d(6)``.
    """

    all_obj_name = grab_obj_name
    val_obj_name = grab_val_obj_name
    test_obj_name = grab_test_obj_name

    def _sg_split_mask(self, data, split):
        # Same object-name split selection as MotionGRAB's evaluate branch.
        split_objs = self._split_objs(split)
        return np.array([n in split_objs for n in data["obj_name"]])

    def _sg_load_arrays(self, data, mask):
        self.obj_name = data["obj_name"][mask]
        self.proc_obj_name = data["proc_obj_name"][mask]
        self.x_lhand = data["x_lhand"][mask]
        self.x_rhand = data["x_rhand"][mask]
        self.x_obj = data["x_obj"][mask]
        self.action_name = data["action_name"][mask]
        self.nframes = data["nframes"][mask]
        self.is_lhand = data["is_lhand"][mask]
        self.is_rhand = data["is_rhand"][mask]
        self.sbj_id = data["sbj_id"][mask] if "sbj_id" in data else None
        self.seq_name = data["seq_name"][mask] if "seq_name" in data else None
        if self.sbj_id is None or self.seq_name is None:
            raise ValueError(
                "GRAB data.npz lacks `sbj_id` / `seq_name`, needed to match CSV "
                "rows to sequences — re-run preprocess_grab.py.")

    def _sg_seq_key(self, seq_idx):
        return (str(self.sbj_id[seq_idx]), str(self.seq_name[seq_idx]))

    def _sg_obj_name(self, seq_idx):
        return str(self.obj_name[seq_idx])


class StableGraspGRABDataModule(MotionGRABDataModule):
    """LightningDataModule wrapping :class:`StableGraspGRAB`.

    Reuses GRAB's preprocessing / split machinery; the stable-grasp CSV path
    comes from ``cfg.stable_grasp_csv``.
    """

    dataset_cls = StableGraspGRAB

    def _make_dataset(self, split: str):
        dc = self.cfg
        return self.dataset_cls(
            data_path=dc.data_path,
            data_obj_pc_path=dc.data_obj_pc_path,
            text_json=dc.text_json,
            max_nframes=-1,          # per-grasp clips; no windowing
            data_ratio=dc.get("data_ratio", 1.0),
            task=self.task,
            split=split,
            lhand_layer=self.lhand_layer,
            rhand_layer=self.rhand_layer,
            subject_layers=self.subject_layers,
            stable_grasp_csv=dc.get("stable_grasp_csv"),
        )

