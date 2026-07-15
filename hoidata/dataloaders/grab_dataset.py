"""GRAB motion dataset (rigid objects, per-subject MANO vtemplates).

Subclass of the shared ``MotionDatasetBase`` / ``MotionDataModuleBase`` in
``common/data/action_dataset.py``; only the GRAB-specific data loading and the
object hooks live here.
"""

import os.path as osp

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


class MotionGRABDataModule(MotionDataModuleBase):
    """LightningDataModule wrapping MotionGRAB (with per-subject MANO layers)."""

    dataset_cls = MotionGRAB

    def __init__(self, cfg, lhand_layer, rhand_layer,
                 task="action_classify", batch_size=1, num_workers=0):
        super().__init__(cfg, lhand_layer, rhand_layer,
                         task=task, batch_size=batch_size, num_workers=num_workers)

        # Per-subject MANO layers (subject-specific hand vtemplates).  Built when
        # the dataset config points at the GRAB subject_meshes directory; samples
        # then use their own subject's hand shape.  Falls back to the shared
        # default layers when unavailable.
        subject_meshes_root = cfg.get('subject_meshes_root', None)
        if subject_meshes_root is not None and osp.isdir(subject_meshes_root):
            from hoidata.utils.mano_subject import SubjectManoLayers
            device = getattr(lhand_layer, 'th_v_template', None)
            device = device.device if device is not None else None
            self.subject_layers = SubjectManoLayers(
                subject_meshes_root,
                mano_root=cfg.get('mano_root', 'mano/models'),
                flat_hand=cfg.get('flat_hand', True),
                device=device,
            )

    def get_sbj_template(self, sbj_id, hand_side):
        sbj_info = self.sbj_info
        template_key = 'rh_vtemp' if hand_side == 'right' else 'lh_vtemp'
        sbj_template = sbj_info[sbj_id][template_key]
        return sbj_template
