"""GigaHands loader in this repo's ``x_hand`` / ``x_obj`` format.

Basic ``MotionGigaHands`` dataset: for every annotated clip it returns the two
hand motions (``x_lhand`` / ``x_rhand``) and the object trajectory (``x_obj``)
in the same trans + 6D-rotation layout used by the GRAB / ARCTIC loaders, plus
the text caption, per-hand availability flags, frame count, sequence name and
the MANO shape (beta) parameters.

The on-disk layout and index construction mirror
``GigaHands/dataset/dataset_multimodal.py`` (the reference loader); the only
difference is that hand / object poses are converted here into this repo's
x-data representation:

* ``x_lhand`` / ``x_rhand`` : (T, 99) = trans(3) + rot6d(16*6).  Built from the
  EasyMocap MANO params (``poses`` / ``shapes`` / ``Rh`` / ``Th``): the global
  orientation ``Rh`` becomes the pose root (``poses[:, :3]`` is zero on disk).
  EasyMocap rotates the hand about the canonical origin (``V @ Rh^T + Th``) while
  manopth rotates about the root joint, so the translation carries a root-pivot
  correction ``Th - J0 + Rh @ J0`` (``J0`` = the shape's root joint); posing the
  result with this repo's MANO layer then reproduces EasyMocap's vertices exactly.
* ``x_obj`` : (T, 9) = trans(3) + rot6d(6).  Built from the first object's
  optimized pose trajectory resampled onto the clip's absolute frame ids, using
  GigaHands' quaternion→rotation-matrix convention.

Expected dataset root (``data/GigaHands`` by default)::

    <root>/
      annotations_v2.jsonl
      hand_poses/<scene>/keypoints_3d_mano/<seq>.json
      hand_poses/<scene>/keypoints_3d/<seq:03d>/chosen_frames_{left,right}.json
      hand_poses/<scene>/params/<seq:03d>.json
      object_poses/<bare_scene>/<object>/<scene>_<seq:04d>/pose/optimized_pose.json
"""

import os
import json
import glob
from os.path import join as pjoin

import numpy as np
import torch
from torch.utils import data
from scipy.spatial.transform import Slerp, Rotation
from scipy.interpolate import interp1d

from hoidata.utils.proc import transform_hand_to_xdata, transform_obj_to_xdata


# --------------------------------------------------------------------------- #
# Object-pose helpers (numpy/scipy; ported from the reference loader so importing
# this module pulls in no rendering / EGL dependencies).
# --------------------------------------------------------------------------- #
def _moving_average_filter(signal, window_size=9):
    """Moving-average smoothing with edge padding."""
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
    pad_len = window_size // 2
    padded = np.pad(signal, ((pad_len, pad_len), (0, 0)), mode="edge")
    kernel = np.ones(window_size) / window_size
    smoothed = np.array(
        [np.convolve(padded[:, i], kernel, mode="valid") for i in range(signal.shape[1])]
    ).T
    return smoothed.squeeze()


def _interpolate_object_poses(object_poses, tracked_frames, smooth=True):
    """Densify a sparse ``{frame: {mesh_translation, mesh_rotation}}`` dict.

    Returns ``(dense_poses_dict, sorted_frame_ids)``; rotations keep their input
    format (3-vector rotvec or 4-vector quaternion) across interpolation.
    """
    trans, rots, idxs = [], [], []
    for cid in tracked_frames:
        frame = str(cid).zfill(6)
        if frame in object_poses:
            trans.append(np.asarray(object_poses[frame]["mesh_translation"]).squeeze())
            rots.append(np.asarray(object_poses[frame]["mesh_rotation"]))
            idxs.append(cid)
    trans, rots, idxs = np.array(trans), np.array(rots), np.array(idxs)

    full_idx = np.arange(idxs[0], idxs[-1] + 1)
    interp_t = interp1d(idxs, trans, axis=0, kind="linear", fill_value="extrapolate")(full_idx)
    if rots.shape[1] == 3:
        r, out_fmt = Rotation.from_rotvec(rots), "rotvec"
    else:
        r, out_fmt = Rotation.from_quat(rots), "quat"
    interp_r = Slerp(idxs, r)(full_idx)
    interp_r = interp_r.as_rotvec() if out_fmt == "rotvec" else interp_r.as_quat()

    if smooth:
        interp_t = _moving_average_filter(interp_t, window_size=9)
        interp_r = _moving_average_filter(interp_r, window_size=9)

    dense = {
        str(fid).zfill(6): {
            "mesh_translation": interp_t[i].tolist(),
            "mesh_rotation": interp_r[i].tolist(),
        }
        for i, fid in enumerate(full_idx)
    }
    return dense, [int(f) for f in full_idx]


def _load_object_trajectory(pose_json, smooth=True):
    """Load one object's dense trajectory from ``optimized_pose.json``.

    Returns ``(frames: List[int], translation: [N, 3], rotation: [N, 3 or 4])``.
    """
    with open(pose_json, "r") as f:
        object_poses = json.load(f)
    tracked = sorted(int(fid) for fid in object_poses.keys())
    dense, frames = _interpolate_object_poses(object_poses, tracked, smooth=smooth)
    trans = np.asarray([dense[str(fid).zfill(6)]["mesh_translation"] for fid in frames])
    rots = np.asarray([dense[str(fid).zfill(6)]["mesh_rotation"] for fid in frames])
    return frames, trans, rots


def _pytorch3d_quat_to_rotmat(quat_wxyz):
    """PyTorch3D quaternion ``[w, x, y, z]`` → rotation matrix (dataset handedness).

    A zero-norm quaternion (e.g. an untracked / padded frame) maps to identity.
    Mirrors the reference loader's transform exactly.
    """
    quat_xyzw = np.asarray([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    if np.linalg.norm(quat_xyzw) < 1e-8:
        return np.eye(3)
    return Rotation.from_quat(quat_xyzw).as_matrix().T


def _object_extmats(frames, trans, rots):
    """Per-frame ``(T, 3, 4)`` extrinsic matrices from a dense trajectory."""
    out = np.zeros((len(frames), 3, 4), dtype=np.float32)
    for i in range(len(frames)):
        R = (
            Rotation.from_rotvec(rots[i]).as_matrix()
            if rots.shape[1] == 3
            else _pytorch3d_quat_to_rotmat(rots[i])
        )
        out[i, :3, :3] = R
        out[i, :3, 3] = np.asarray(trans[i]).reshape(3)
    return out


def _resample_object(frames, trans, rots, clip_frames):
    """Interpolate an object's translation / rotation onto ``clip_frames``.

    Frames outside the object's tracked range are marked invalid and left at
    identity (rotation) / zero (translation).  Returns
    ``(trans[T, 3], rots[T, 3 or 4], valid[T])``.
    """
    frames = np.asarray(frames)
    valid = (clip_frames >= frames[0]) & (clip_frames <= frames[-1])
    rot_dim = rots.shape[1]
    out_t = np.zeros((len(clip_frames), 3), np.float32)
    out_r = np.zeros((len(clip_frames), rot_dim), np.float32)
    if rot_dim == 4:
        out_r[:, 0] = 1.0  # identity quaternion [w, x, y, z] = [1, 0, 0, 0]
    if valid.any():
        cf = clip_frames[valid]
        out_t[valid] = interp1d(frames, trans, axis=0)(cf)
        r = Rotation.from_rotvec(rots) if rot_dim == 3 else Rotation.from_quat(rots)
        interp_r = Slerp(frames, r)(cf)
        out_r[valid] = interp_r.as_rotvec() if rot_dim == 3 else interp_r.as_quat()
    return out_t, out_r, valid


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class MotionGigaHands(data.Dataset):
    """GigaHands clips in this repo's ``x_hand`` / ``x_obj`` format.

    Args:
        dataset_root: folder with ``annotations_v2.jsonl``, ``hand_poses/`` and
            ``object_poses/`` (defaults to ``data/GigaHands``).
        split: ``'train' | 'test' | 'val' | 'all'`` (same 0.8 / 0.15 / 0.05 split
            and seed as the reference loader, so indices line up).
        object_number: ``None`` (no filter), ``'single'`` or ``'multiple'`` — keep
            only clips the taxonomy labels ``single-object`` / ``multi-object`` in
            ``sequence_type.json``.
        object_rigidity: ``'rigid'`` (default), ``'deformable'`` or ``None`` (no
            filter) — keep only clips the taxonomy labels ``rigid`` / ``deformable``.
            When both ``object_number`` and ``object_rigidity`` are set the kept
            set is their INTERSECTION.
        require_object: drop clips with no object pose found.
        smooth_object: smooth / interpolate object trajectories.
        sequence_type_file: override the taxonomy path (defaults to
            ``<dataset_root>/sequence_type.json``); needed for ``object_number``.
        annotation_file: override annotation path.

    Each item is a dict with:
        ``seq_name``  : ``"<scene>-<seq>:<start>-<end>"`` (unique per clip).
        ``text``      : the clip's caption.
        ``nframes``   : int, number of frames T in the clip.
        ``is_lhand`` / ``is_rhand`` : bool, whether that hand is tracked in the clip.
        ``x_lhand`` / ``x_rhand`` : (T, 99) float32 trans + rot6d.
        ``x_obj``     : (T, 9) float32 trans + rot6d for the first object, or None.
        ``lhand_beta`` / ``rhand_beta`` : (10,) float32 MANO shape parameters.
    """

    #: ``object_number`` value → taxonomy key in ``sequence_type.json``.
    _OBJECT_NUMBER_KEYS = {"single": "single-object", "multiple": "multi-object"}
    #: ``object_rigidity`` value → taxonomy key in ``sequence_type.json``.
    _OBJECT_RIGIDITY_KEYS = {"rigid": "rigid", "deformable": "deformable"}

    def __init__(
        self,
        dataset_root="data/GigaHands",
        split="all",
        object_number=None,
        object_rigidity="rigid",
        require_object=False,
        smooth_object=True,
        sequence_type_file=None,
        annotation_file=None,
        mano_root="/home/zxc417/data/mano_v1_2/models",
        flat_hand=False,
    ):
        if object_number is not None and object_number not in self._OBJECT_NUMBER_KEYS:
            raise ValueError(
                f"object_number must be one of "
                f"{sorted(self._OBJECT_NUMBER_KEYS)} or None, got {object_number!r}")
        if object_rigidity is not None and object_rigidity not in self._OBJECT_RIGIDITY_KEYS:
            raise ValueError(
                f"object_rigidity must be one of "
                f"{sorted(self._OBJECT_RIGIDITY_KEYS)} or None, got {object_rigidity!r}")
        self.dataset_root = dataset_root
        self.split = split
        self.object_number = object_number
        self.object_rigidity = object_rigidity
        self.require_object = require_object
        self.smooth_object = smooth_object
        self.mano_root = mano_root
        self.flat_hand = flat_hand
        #: Lazily-built manopth layers + per-shape root-joint (J0) cache, used to
        #: map EasyMocap's origin-pivot global transform onto manopth's
        #: root-joint-pivot convention (see ``_hand_xdata``).
        self._j0_layers = {}

        self.hand_root = pjoin(dataset_root, "hand_poses")
        self.object_root = pjoin(dataset_root, "object_poses")

        # Taxonomy filter: keys are '<scene>-<seq>:<start>-<end>' (the raw
        # annotation frame window is part of the identity, since one (scene, seq)
        # can carry several annotated sub-clips).  With both an object_number and
        # an object_rigidity set the kept set is the INTERSECTION of the two.
        self.allowed_keys = None
        taxonomy_filters = []
        if object_number is not None:
            taxonomy_filters.append(self._OBJECT_NUMBER_KEYS[object_number])
        if object_rigidity is not None:
            taxonomy_filters.append(self._OBJECT_RIGIDITY_KEYS[object_rigidity])
        if taxonomy_filters:
            type_file = sequence_type_file or pjoin(dataset_root, "sequence_type.json")
            if not os.path.exists(type_file):
                raise FileNotFoundError(
                    f"object_number / object_rigidity filtering needs {type_file}")
            with open(type_file) as f:
                types = json.load(f)
            for tax_key in taxonomy_filters:
                members = set(types.get(tax_key, ()))
                self.allowed_keys = (members if self.allowed_keys is None
                                     else (self.allowed_keys & members))

        text_file = annotation_file or pjoin(dataset_root, "annotations_v2.jsonl")
        data_list = []
        with open(text_file, "r", encoding="utf-8") as f:
            for line in f:
                info = json.loads(line)
                seq = info["sequence"][0] if isinstance(info["sequence"], list) else info["sequence"]
                scene = info["scene"]
                script = info.get("clarify_annotation", "None")
                if split != "all" and (script == "None" or script == "Buggy"):
                    continue

                kp_path = pjoin(self.hand_root, scene, "keypoints_3d_mano", f"{seq}.json")
                params_path = pjoin(self.hand_root, scene, "params", f"{int(seq):03d}.json")
                if not (os.path.exists(kp_path) and os.path.exists(params_path)):
                    continue

                if self.allowed_keys is not None:
                    key = (f"{scene}-{seq}:"
                           f"{info['start_frame_id']}-{info['end_frame_id']}")
                    if key not in self.allowed_keys:
                        continue

                object_dirs = self._resolve_object_dirs(scene, seq)
                if self.require_object and not object_dirs:
                    continue

                data_list.append({
                    "scene": scene,
                    "seq": seq,
                    "params_path": params_path,
                    "kp3d_dir": pjoin(self.hand_root, scene, "keypoints_3d", f"{int(seq):03d}"),
                    "chosen_frames": (info["start_frame_id"], info["end_frame_id"]),
                    "caption": script,
                    "object_dirs": object_dirs,
                })

        self.data_list = self._apply_split(data_list, split)
        tags = [t for t in (object_number, object_rigidity) if t]
        subset = ("/" + "+".join(tags)) if tags else ""
        print(f"MotionGigaHands [{split}{subset}] {len(self.data_list)} sequences")

    # ---- index construction ------------------------------------------------ #
    def _resolve_object_dirs(self, scene, seq):
        """Every object folder for this ``(scene, seq)`` via a glob on the unique
        ``<scene>_<seq:04d>`` folder name.

        Returns ``[(object_name, bare_scene, pose_json)]``.
        """
        try:
            seq4 = f"{int(seq):04d}"
        except (TypeError, ValueError):
            seq4 = str(seq)
        pattern = pjoin(self.object_root, "*", "*", f"{scene}_{seq4}", "pose", "optimized_pose.json")
        found = []
        for pose_json in sorted(glob.glob(pattern)):
            parts = os.path.normpath(pose_json).split(os.sep)
            # .../object_poses/<bare_scene>/<object>/<scene>_<seq>/pose/optimized_pose.json
            bare_scene, object_name = parts[-5], parts[-4]
            found.append((object_name, bare_scene, pose_json))
        return found

    def find_object_mesh(self, object_name, bare_scene):
        """Path to the object's ``.obj`` mesh under ``object_meta/``, or ``None``.

        The mesh's object-name folder swaps some ``_`` for ``-`` relative to the
        pose folder — sometimes all (pose ``file_button_envelope`` → mesh
        ``file-button-envelope``), sometimes only some (pose ``tablet_iphone_scan``
        → mesh ``tablet-iphone_scan``).  Try the exact and all-dash spellings
        first, then fall back to a separator-insensitive match over the scene dir.
        """
        scene_dir = pjoin(self.dataset_root, "object_meta", bare_scene)
        for name in (object_name, object_name.replace("_", "-")):
            cand = pjoin(scene_dir, name)
            if os.path.isdir(cand):
                objs = sorted(glob.glob(pjoin(cand, "*.obj")))
                if objs:
                    return objs[0]
        # Separator-insensitive fallback: match the folder ignoring _ vs -.
        if os.path.isdir(scene_dir):
            norm = object_name.replace("-", "_")
            for entry in sorted(os.listdir(scene_dir)):
                if entry.replace("-", "_") == norm:
                    objs = sorted(glob.glob(pjoin(scene_dir, entry, "*.obj")))
                    if objs:
                        return objs[0]
        return None

    @staticmethod
    def _apply_split(data_list, split):
        if split == "all":
            return data_list
        from sklearn.model_selection import train_test_split
        train_data, remaining = train_test_split(data_list, test_size=0.2, random_state=42)
        test_data, val_data = train_test_split(remaining, test_size=0.25, random_state=42)
        return {"train": train_data, "test": test_data, "val": val_data}[split]

    # ---- API --------------------------------------------------------------- #
    def __len__(self):
        return len(self.data_list)

    def clip_meta(self, item):
        """Cheap per-clip metadata (no MANO posing / mesh loading) for listings.

        Returns ``{"seq_name", "text", "obj_name", "nframes"}``.  ``nframes`` is
        read from the clip's tracked-frame window (two small JSON reads).
        """
        entry = self.data_list[int(item)]
        scene, seq = entry["scene"], entry["seq"]
        start, end = entry["chosen_frames"]
        with open(pjoin(entry["kp3d_dir"], "chosen_frames_left.json")) as f:
            n_left = set(json.load(f))
        with open(pjoin(entry["kp3d_dir"], "chosen_frames_right.json")) as f:
            n_right = set(json.load(f))
        n_rows = len(n_left | n_right)
        start = max(0, start)
        end_row = (n_rows - 1) if end == -1 else min(end, n_rows - 1)
        nframes = int(end_row - start + 1)
        obj_name = entry["object_dirs"][0][0] if entry["object_dirs"] else ""
        return {
            "seq_name": f"{scene}-{seq}:{start}-{end}",
            "text": entry["caption"],
            "obj_name": obj_name,
            "nframes": nframes,
        }

    def __getitem__(self, item):
        entry = self.data_list[int(item)]
        scene, seq = entry["scene"], entry["seq"]
        start, end = entry["chosen_frames"]

        # Tracked-frame index: params / keypoints row i <-> hand_union[i]. The
        # annotation (start, end) are POSITIONAL rows into this array.
        with open(pjoin(entry["kp3d_dir"], "chosen_frames_left.json")) as f:
            left_frames = set(json.load(f))
        with open(pjoin(entry["kp3d_dir"], "chosen_frames_right.json")) as f:
            right_frames = set(json.load(f))
        hand_union = sorted(left_frames | right_frames)
        n_rows = len(hand_union)

        start = max(0, start)
        end_row = (n_rows - 1) if end == -1 else min(end, n_rows - 1)
        rows = np.arange(start, end_row + 1)
        frame_ids = np.asarray([hand_union[r] for r in rows])   # absolute ids
        nframes = int(len(rows))

        # A hand is "available" in the clip if it is tracked on any of its frames.
        clip_frame_set = set(int(f) for f in frame_ids)
        is_lhand = bool(clip_frame_set & left_frames)
        is_rhand = bool(clip_frame_set & right_frames)

        # ---- hands: EasyMocap MANO params -> (T, 99) trans + rot6d ---------- #
        with open(entry["params_path"], "r") as f:
            params = json.load(f)
        x_lhand, lhand_beta = self._hand_xdata(params["left"], rows, "left")
        x_rhand, rhand_beta = self._hand_xdata(params["right"], rows, "right")

        # ---- object: first object's trajectory -> (T, 9) trans + rot6d ----- #
        x_obj = None
        obj_valid = None
        obj_name = None
        obj_mesh = None
        if entry["object_dirs"]:
            obj_name, bare_scene, pose_json = entry["object_dirs"][0]
            frames, trans, rots = _load_object_trajectory(
                pose_json, smooth=self.smooth_object)
            res_t, res_r, valid = _resample_object(frames, trans, rots, frame_ids)
            extmat = _object_extmats(frame_ids, res_t, res_r)         # (T, 3, 4)
            x_obj = transform_obj_to_xdata(torch.from_numpy(extmat).float())
            obj_valid = valid
            obj_mesh = self.find_object_mesh(obj_name, bare_scene)

        return {
            "seq_name": f"{scene}-{seq}:{start}-{end}",
            "text": entry["caption"],
            "nframes": nframes,
            "is_lhand": is_lhand,
            "is_rhand": is_rhand,
            "x_lhand": x_lhand,          # (T, 99)
            "x_rhand": x_rhand,          # (T, 99)
            "x_obj": x_obj,              # (T, 9) or None
            "obj_valid": obj_valid,      # (T,) bool or None — object tracked this frame
            "obj_name": obj_name,        # str or None
            "obj_mesh": obj_mesh,        # path to the .obj mesh, or None
            "lhand_beta": lhand_beta,    # (10,)
            "rhand_beta": rhand_beta,    # (10,)
        }

    # ---- helpers ----------------------------------------------------------- #
    def _root_joint(self, side, beta):
        """MANO root joint ``J0`` (3,) for this hand shape, in the canonical frame.

        ``J0`` depends only on the shape (betas), not the pose, so it is cached
        per (side, rounded-beta).  Used to convert EasyMocap's global transform
        (rotation about the origin) to manopth's (rotation about the root joint).
        """
        key = (side, tuple(np.round(beta, 4)))
        if key in self._j0_layers:
            return self._j0_layers[key]
        from hoidata.utils.hand_proc import ManoLayerWrapper
        layer = getattr(self, f"_j0_layer_{side}", None)
        if layer is None:
            layer = ManoLayerWrapper(
                side=side, flat_hand_mean=self.flat_hand, mano_root=self.mano_root,
                use_pca=False, ncomps=45).cuda()
            setattr(self, f"_j0_layer_{side}", layer)
        pose = torch.zeros(1, 48, device="cuda")
        betas = torch.from_numpy(beta.reshape(1, 10).astype(np.float32)).cuda()
        trans = torch.zeros(1, 3, device="cuda")
        j0 = layer(pose, betas, trans).joints_w_tip[0, 0].detach().cpu().numpy()
        self._j0_layers[key] = j0.astype(np.float32)
        return self._j0_layers[key]

    def _hand_xdata(self, side_params, rows, side):
        """One hand's EasyMocap params -> ((T, 99) x-data, (10,) beta).

        EasyMocap stores the global orientation in ``Rh`` (``poses[:, :3]`` is
        zero) and the global translation in ``Th``, and composes vertices as
        ``V_world = V_canonical @ Rh^T + Th`` — i.e. ``Rh`` rotates the posed hand
        about the **canonical origin**.  manopth (this repo's layer) instead treats
        ``pose[:, :3]`` as the root-joint rotation, rotating about the **root joint
        J0** and adding the translation afterward.  The two agree only up to a
        translation: for identical vertices the manopth translation must be

            trans = Th - J0 + Rh @ J0

        (derived from ``R(V - J0) + J0 + trans == R V + Th``; verified to match
        EasyMocap vertices to < 1e-6 m).  Without this the hand is offset ~15 cm,
        differently per hand — hands miss the object and penetrate each other.
        """
        poses = np.asarray(side_params["poses"], np.float32)[rows]    # (T, 48)
        Rh = np.asarray(side_params["Rh"], np.float32)[rows]          # (T, 3)
        Th = np.asarray(side_params["Th"], np.float32)[rows]          # (T, 3)
        beta = np.asarray(side_params["shapes"], np.float32).reshape(-1)[:10]
        beta = beta.astype(np.float32)

        pose = poses.copy()
        pose[:, :3] = Rh                                              # global orient

        # Root-pivot correction: trans = Th - J0 + Rh @ J0.
        J0 = self._root_joint(side, beta)                            # (3,)
        from pytorch3d.transforms import axis_angle_to_matrix
        Rmat = axis_angle_to_matrix(torch.from_numpy(Rh)).numpy()     # (T, 3, 3)
        trans = Th - J0 + np.einsum("tij,j->ti", Rmat, J0)           # (T, 3)

        x_hand = transform_hand_to_xdata(
            torch.from_numpy(trans.astype(np.float32)),
            torch.from_numpy(pose))                                  # (T, 99)
        return x_hand, beta
