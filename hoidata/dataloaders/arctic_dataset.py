"""ARCTIC motion dataset (articulated objects, +1 articulation DoF).

Subclass of the shared ``MotionDatasetBase`` / ``MotionDataModuleBase`` in
``common/data/action_dataset.py``.  ARCTIC objects have a top + bottom part joined
by a 1-DoF revolute joint, so the object x-data is **10-D**:
``trans(3) + rot6d(6) + articulation_angle(1)``.  Hands use MANO exactly as GRAB
(mean template; ARCTIC has no per-subject vtemplates).

``process_text`` is identical to GRAB's, so it is reused directly.
"""

import numpy as np

from .action_dataset import (
    MotionDatasetBase, MotionDataModuleBase, compute_xdata_delta,
    StableGraspMotionDataset)
from hoidata.utils.proc import pc_normalize, proc_numpy, proc_torch_cuda, process_text
from hoidata.constants.arctic_constants import (
    arctic_obj_name, arctic_val_obj_name, arctic_test_obj_name,
)


class MotionARCTIC(MotionDatasetBase):
    """Articulated ARCTIC objects; object x-data is (T, 10) = trans+rot6d+angle."""

    all_obj_name = arctic_obj_name
    val_obj_name = arctic_val_obj_name
    test_obj_name = arctic_test_obj_name

    def _load_data(self, data, task, split):
        if task in ("action_classify", "evaluate"):
            split_objs = self._split_objs(split)
            all_obj_names = data["object_name"]
            mask = np.array([n in split_objs for n in all_obj_names])

            self.object_name = all_obj_names[mask]
            self.x_lhand     = data["x_lhand"][mask]
            self.x_rhand     = data["x_rhand"][mask]
            self.is_lhand    = data["is_lhand"][mask]
            self.is_rhand    = data["is_rhand"][mask]
            self.action_name = data["action_name"][mask]
            self.nframes     = data["nframes"][mask]
            # Optional per-sequence annotation intervals + subject + stable-grasp
            # intervals (present only in full_motion-preprocessed data.npz);
            # consumed by the web viewer.
            self.labels  = data["labels"][mask] if "labels" in data else None
            self.subject = data["subject"][mask] if "subject" in data else None
            self.grasps  = data["grasps"][mask] if "grasps" in data else None
            if task == "evaluate":
                self.x_obj       = data["x_obj"][mask]
                self.x_obj_angle = data["x_obj_angle"][mask]
            else:
                self.x_obj = self.x_obj_angle = None
            self.lcf_idx = self.lcov_idx = self.lchj_idx = self.ldist_value = None
            self.rcf_idx = self.rcov_idx = self.rchj_idx = self.rdist_value = None
        else:  # generation
            # Train uses every clip; val/test restrict to their held-out objects
            # so reconstruction can be inspected on unseen objects.
            all_obj_names = data["object_name"]
            if split in ("val", "test"):
                split_objs = self._split_objs(split)
                mask = np.array([n in split_objs for n in all_obj_names])
            else:
                mask = np.ones(len(all_obj_names), dtype=bool)
            self.object_name = all_obj_names[mask]
            self.x_lhand = data["x_lhand"][mask]
            self.x_rhand = data["x_rhand"][mask]
            self.x_obj = data["x_obj"][mask]
            self.x_obj_angle = data["x_obj_angle"][mask]
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

    # ------------------------------------------------------------------ hooks
    def _object_name(self, index):
        return self.object_name[index]

    def _text(self, index, is_lhand, is_rhand):
        return process_text(
            self.action_name[index], self.object_name[index],
            is_lhand, is_rhand, self.text_description,
        )

    def _slice_obj(self, index, init_frame, nframes):
        """Object window (slice_len, 10): trans(3)+rot6d(6), then +angle(1).

        Returns slice_len = nframes (+1 successor) frames so the last delta is real;
        the caller stores only the first max_nframes as the absolute stream.
        """
        sl = self._slice_len
        x_obj = self.x_obj[index][init_frame:init_frame + sl]
        x_obj_angle = self.x_obj_angle[index][init_frame:init_frame + sl]
        x_obj = np.concatenate([x_obj, x_obj_angle], axis=1)  # (slice_len, 10)
        return x_obj

    def _obj_delta(self, x_obj, nframes, max_nframes):
        """Per-frame delta for the 10-D articulated object, padded to max_nframes.

        ``x_obj`` is the extended window (slice_len, 10).
        - dims [:9] (trans + rot6d): rotation-space delta via compute_xdata_delta.
        - dim 9 (articulation angle): scalar delta angle[t+1] - angle[t].
        Every window frame gets a real delta when a successor was read; only a
        window flush against the end of the sequence leaves the final frame zero.
        """
        delta = np.zeros((max_nframes, x_obj.shape[1]), dtype=np.float32)
        if x_obj.shape[0] >= 2:
            d9 = compute_xdata_delta(x_obj[:, :9])            # (slice_len-1, 9)
            delta[:d9.shape[0], :9] = d9[:max_nframes]
            angle = x_obj[:, 9:10]
            d_ang = angle[1:] - angle[:-1]                    # (slice_len-1, 1)
            delta[:d_ang.shape[0], 9:10] = d_ang[:max_nframes]
        return delta

    def _object_pc(self, object_name, item):
        # ARCTIC object_model returns a 5-tuple (extra obj_pc_top over sampled pts).
        _, obj_pc, obj_pc_normal, _, obj_pc_top_idx = self.object_model(object_name)
        normalized_obj_pc, obj_norm_cent, obj_norm_scale = pc_normalize(
            obj_pc, return_params=True)
        item["obj_pc_top_idx"] = obj_pc_top_idx
        item["obj_pc"] = obj_pc
        item["normalized_obj_pc"] = normalized_obj_pc
        item["obj_pc_normal"] = obj_pc_normal
        item["obj_cent"] = obj_norm_cent
        item["obj_scale"] = obj_norm_scale

    def _obj_traj_from_xobj(self, x_obj, nframes):
        """Eval object trajectory + articulation angle.

        x_obj here is the stored 9-D pose (trans + rot6d); the articulation angle
        is carried separately so the caller can pose the articulated mesh.  Unlike
        GRAB, the rotvec is **not** negated — the ARCTIC articulation/posing path
        (``process_obj_result``) applies R directly (no transpose) on the raw rotvec.
        """
        from pytorch3d.transforms import matrix_to_axis_angle, rotation_6d_to_matrix
        obj_trans = x_obj[:, :3]
        obj_rotvec = proc_numpy(
            matrix_to_axis_angle(rotation_6d_to_matrix(proc_torch_cuda(x_obj[:, 3:9])))
        )
        obj_traj = np.concatenate([obj_trans, obj_rotvec], axis=1)  # (T, 6)
        obj_angle = self.x_obj_angle[self._eval_index][:nframes, 0]  # (T,)
        return {
            "obj_traj": obj_traj.astype(np.float32),
            "obj_angle": obj_angle.astype(np.float32),
        }

    def _getitem_full(self, index):
        # Stash the index so _obj_traj_from_xobj can fetch the matching angle.
        self._eval_index = index
        return super()._getitem_full(index)


class MotionARCTICDataModule(MotionDataModuleBase):
    """LightningDataModule wrapping MotionARCTIC (no per-subject MANO layers)."""

    dataset_cls = MotionARCTIC

    def prepare_data(self):
        """Run any missing ARCTIC preprocessing steps before setup.

        Detects each preprocessed file independently and only regenerates the
        missing ones (see ``_run_missing_preprocessing``).
        """
        from hoidata.preprocess import preprocess_arctic
        self._run_missing_preprocessing(preprocess_arctic)


class StableGraspARCTIC(StableGraspMotionDataset):
    """ARCTIC stable-grasp clips — one item per stable-grasp CSV row.

    CSV rows are keyed by ``{subject}/{action_name}`` (e.g. ``s01/box_grab_01``);
    ``action_name`` is already the unique sequence filename for ARCTIC.  The
    object x-data is the articulated 10-D ``trans(3) + rot6d(6) + angle(1)``,
    assembled from the separately-stored ``x_obj`` and ``x_obj_angle``.
    """

    all_obj_name = arctic_obj_name
    val_obj_name = arctic_val_obj_name
    test_obj_name = arctic_test_obj_name

    def _sg_split_mask(self, data, split):
        split_objs = self._split_objs(split)
        return np.array([n in split_objs for n in data["object_name"]])

    def _sg_load_arrays(self, data, mask):
        self.object_name = data["object_name"][mask]
        self.x_lhand = data["x_lhand"][mask]
        self.x_rhand = data["x_rhand"][mask]
        self.x_obj = data["x_obj"][mask]
        self.x_obj_angle = data["x_obj_angle"][mask]
        self.action_name = data["action_name"][mask]
        self.nframes = data["nframes"][mask]
        self.is_lhand = data["is_lhand"][mask]
        self.is_rhand = data["is_rhand"][mask]
        self.subject = data["subject"][mask] if "subject" in data else None
        if self.subject is None:
            raise ValueError(
                "ARCTIC data.npz lacks `subject`, needed to match CSV rows to "
                "sequences — re-run preprocess_arctic.py with full_motion.")

    def _sg_seq_key(self, seq_idx):
        return (str(self.subject[seq_idx]), str(self.action_name[seq_idx]))

    def _sg_obj_name(self, seq_idx):
        return str(self.object_name[seq_idx])

    def _sg_x_object(self, seq_idx, a, b):
        """(T, 10) = trans(3) + rot6d(6) + articulation angle(1)."""
        x_obj = np.asarray(self.x_obj[seq_idx][a:b], dtype=np.float32)
        x_ang = np.asarray(self.x_obj_angle[seq_idx][a:b], dtype=np.float32)
        return np.concatenate([x_obj, x_ang], axis=1)


class StableGraspARCTICDataModule(MotionARCTICDataModule):
    """LightningDataModule wrapping :class:`StableGraspARCTIC`.

    Reuses ARCTIC's preprocessing / split machinery; the stable-grasp CSV path
    comes from ``cfg.stable_grasp_csv``.
    """

    dataset_cls = StableGraspARCTIC

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
