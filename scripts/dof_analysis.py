"""
Analyze the actual degrees of freedom (DOFs) of stable grasp & transfer actions.

Currently this script only *loads* the relevant datasets and reports what was
loaded — the DOF analysis itself is left to be filled in (see ``analyze()``).

Datasets loaded:
  - stable grasps : one clip per stable-grasp CSV row (StableGraspGRAB / StableGraspARCTIC)
  - transitions   : GRAB transition motions between stable grasps (TransitionMotionGRAB)

Each item exposes the raw x-data (no MANO posing):
    x_hand   (T, 99)     trans(3) + rot6d(16*6)
    x_object (T, 9|10)   GRAB: trans+rot6d;  ARCTIC: +articulation angle
    is_lhand / is_rhand, obj_name, n_frames

Run in the ``hoi_common`` conda env (CUDA 12.4):

    python -m scripts.dof_analysis                               # GRAB, test split
    python -m scripts.dof_analysis --config configs/arctic.yaml
    python -m scripts.dof_analysis --split train --transitions
"""

import argparse

import numpy as np
from omegaconf import OmegaConf
import torch

from hoidata.utils.hand_proc import ManoLayerWrapper
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle
from hoidata.analysis.joint_rotation import (
    normalize_joint_rotations, geodesic_angle)
from hoidata.dataloaders.grab_dataset import (
    StableGraspGRABDataModule, TransitionMotionGRABDataModule)
from hoidata.dataloaders.arctic_dataset import StableGraspARCTICDataModule


def build_hand_layers(cfg):
    """Shared left/right MANO layers (48-D axis-angle pose convention).

    ``use_pca`` must be off (ncomps=45) so the 45-D finger pose is treated as
    axis-angle rather than PCA coefficients.
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
    return lhand_layer, rhand_layer


def load_dataset(dm_cls, cfg, split, lhand_layer, rhand_layer):
    """Build a datamodule, run setup, and return the dataset for ``split``."""
    dm = dm_cls(cfg, lhand_layer, rhand_layer, task="evaluate",
                batch_size=1, num_workers=0)
    # 'evaluate' only builds the test set; use _make_dataset directly so any
    # split (train/val/test) can be inspected.
    return dm._make_dataset(split)


def describe(name, ds):
    """Print a short summary of a loaded dataset."""
    n = len(ds)
    print(f"\n=== {name} ===")
    print(f"  items: {n}")
    if n == 0:
        return
    lens = np.array([ds[i]["n_frames"] for i in range(n)])
    sides = [("left" if ds[i]["is_lhand"] else "right") for i in range(n)]
    objs = sorted({ds[i]["obj_name"] for i in range(n)})
    print(f"  frames: min={lens.min()} max={lens.max()} "
          f"mean={lens.mean():.1f} total={lens.sum()}")
    print(f"  hands:  left={sides.count('left')} right={sides.count('right')}")
    print(f"  objects ({len(objs)}): {objs[:8]}{' ...' if len(objs) > 8 else ''}")
    it = ds[0]
    print(f"  item[0]: x_hand={it['x_hand'].shape} x_object={it['x_object'].shape} "
          f"obj={it['obj_name']} n_frames={it['n_frames']}")


# MANO finger → its 3 joint indices within the 15 finger joints (x_hand[:, 9:]).
# The pose is fed to manopth in its native order (index, middle, pinky, ring,
# thumb), each finger a proximal→distal chain of 3 joints — confirmed by the
# kintree parents [0,1,2, 0,4,5, 0,7,8, 0,10,11, 0,13,14].
FINGER_JOINTS = {
    "index":  [0, 1, 2],
    "middle": [3, 4, 5],
    "pinky":  [6, 7, 8],
    "ring":   [9, 10, 11],
    "thumb":  [12, 13, 14],
}
FINGERS = list(FINGER_JOINTS)


def analyze(datasets, lhand_layer, rhand_layer):
    """Fixed-axis / fixed-pivot DOF fit per clip, in the transition hand's frame.

    For each clip the object pose is expressed in the wrist frame exactly as the
    ``--wrist-aligned`` viewer does, then the shared estimator
    ``_fit_fixed_axis_motion`` fits a single rotation axis and a fixed pivot.
    Sharing the viewer's origin (``_wrist_origin``) and fit keeps the reported
    ``resid_angle`` / ``pivot`` in lock-step with what the viewer draws.

    Per-finger R2 (fraction of each finger's mean-relative joint-rotation
    variance explained by the object's single-DOF harmonic angle) is collected
    across every clip and the dataset mean AND median are printed at the end.
    The median is reported alongside the mean because a few near-static clips
    (a finger that barely moves has total_var ~ 0, so its R2 is ill-defined and
    can blow up hugely negative) otherwise dominate the plain mean.
    """
    from hoidata.vis.o3d_visualizer import _wrist_origin, _fit_fixed_axis_motion

    for name, ds in datasets.items():
        print(f"  {name}: {len(ds)} clips")
        # Collect per-finger R2 over all clips for the dataset mean/median.
        finger_r2_all = {f: [] for f in FINGERS}
        # Collect per-clip residual-PCA cumulative EVR (K1,K2,K3) for the average.
        topk_evr_all = []
        for i, it in enumerate(ds):
            # Transition hand for this clip (stable grasps expose is_lhand/is_rhand;
            # transitions expose hand_sides — accept either, like the viewer).
            if "hand_sides" in it:
                side = it["hand_sides"][0]
            else:
                side = "left" if it["is_lhand"] else "right"
            layer = lhand_layer if side == "left" else rhand_layer

            x_object = it["x_object"]                              # (T, 9|10) numpy
            x_hand = it["x_hand"]                                  # (T, 99)   numpy
            xo = torch.from_numpy(np.asarray(x_object, np.float32))
            xh = torch.from_numpy(np.asarray(x_hand, np.float32))

            # Wrist frame: R_w from x_hand rot6d, origin from the POSED wrist joint
            # (not x_hand[:, :3], MANO's global offset — see _wrist_origin).
            wrist_R = rotation_6d_to_matrix(xh[:, 3:9])           # (T,3,3) R_w
            wrist_o = torch.from_numpy(_wrist_origin(x_hand, layer))  # (T,3)

            # GRAB stores R_o un-inverted, but the rotation *applied* to the mesh
            # is R_o^T -- see the grab branch of process_obj_result (einsum
            # "btij,bki->btkj" == verts @ R_o == applies R_o^T).  Object pose in
            # the wrist frame:  R_true = R_w^T R_o^T,  t_true = R_w^T (t_o - o_w).
            obj_R = rotation_6d_to_matrix(xo[:, 3:9])             # R_o (stored)
            obj_t = xo[:, :3]                                     # (T,3)
            R_true = torch.einsum('tji,tkj->tik', wrist_R, obj_R)          # R_w^T R_o^T
            t_true = torch.einsum('tji,tj->ti', wrist_R, obj_t - wrist_o)  # (T,3)

            _R_rec, _t_rec, _axis, _angle, resid_deg, pivot = \
                _fit_fixed_axis_motion(R_true.numpy(), t_true.numpy())

            # Joint-rotation normalization: per finger joint, factor out the
            # Karcher-mean pose and keep the mean-relative rotations.  The per-
            # joint range of motion is the max geodesic angle of those relatives.
            hand_pose = xh[:, 9:]                     # (T, 90) = 15 joints rot6d
            _mean_R, rel_R = normalize_joint_rotations(hand_pose)
            rel_y = matrix_to_axis_angle(rel_R).view(-1, 45)
            # rom_deg = torch.rad2deg(geodesic_angle(rel_R)).amax(dim=0)  # (J,)

            harmonic_mat = torch.ones(_t_rec.shape[0], 3)
            harmonic_mat[:, 1] = torch.cos(torch.from_numpy(_angle))
            harmonic_mat[:, 2] = torch.sin(torch.from_numpy(_angle))
            # harmonic_mat[:, 3] = torch.cos(2 * torch.from_numpy(_angle))
            # harmonic_mat[:, 4] = torch.sin(2 * torch.from_numpy(_angle))

            C = torch.linalg.solve(harmonic_mat.transpose(-1, -2) @ harmonic_mat, harmonic_mat.transpose(-1, -2) @ rel_y)
            predicted_y = harmonic_mat @ C
            residual = predicted_y - rel_y                       # (T, 45)
            residual_var = (residual ** 2).mean(dim=0)
            total_var = torch.var(rel_y, dim=0)
            R2 = 1 - residual_var / (total_var + 1e-6)          # (45,) per axis-angle dim

            # PCA on the harmonic-fit residual: how much joint-rotation variation
            # is left *after* the single object DOF is regressed out (its
            # principal directions are the candidate extra DOFs).  Covariance
            # over the T frames of the 45-D residual; eigenvalues = variance
            # along the residual modes.  The residual is already zero-mean per
            # dim (LSQ residual is orthogonal to harmonic_mat's constant column),
            # but center explicitly to be safe.
            res_c = residual - residual.mean(dim=0, keepdim=True)
            cov = res_c.transpose(0, 1) @ res_c / max(res_c.shape[0] - 1, 1)  # (45,45)
            pca_eigvals = torch.linalg.eigvalsh(cov).flip(0)    # (45,) descending
            # Fraction of residual variance captured by the top-K components
            # (cumulative explained-variance ratio), K = 1, 2, 3.
            evr = pca_eigvals / pca_eigvals.sum().clamp_min(1e-12)
            topk_evr = torch.cumsum(evr, dim=0)[:3]             # [top1, top2, top3]
            topk_evr_all.append(topk_evr.tolist())

            # Per-finger R2: average over that finger's 3 joints x 3 axis-angle
            # dims (rel_y is laid out (T, 15*3) joint-major, so joint j -> 3j:3j+3).
            R2j = R2.view(15, 3)                                # (15 joints, 3 dims)
            finger_r2 = {f: R2j[FINGER_JOINTS[f]].mean().item() for f in FINGERS}
            for f in FINGERS:
                finger_r2_all[f].append(finger_r2[f])

            fingers_str = " ".join(f"{f[:3]}={finger_r2[f]:.3f}" for f in FINGERS)
            evr_str = ", ".join(f"K{k+1}={v:.2%}" for k, v in enumerate(topk_evr.tolist()))
            print(f"    ds[{i}]: {side:5s} obj={it['obj_name']:12s} "
                  f"n_frames={it['n_frames']:3d} | "
                  f"resid_angle={resid_deg.mean():.2f}deg "
                  f"pivot={pivot.round(4)} | R2 {fingers_str} | "
                  f"resid PCA cum-var [{evr_str}]")

        # Dataset per-finger R2: mean AND median (median is robust to the few
        # near-static clips whose R2 blows up; see the docstring).
        n_clips = len(next(iter(finger_r2_all.values())))
        if n_clips:
            mean_str = "  ".join(
                f"{f}={np.mean(finger_r2_all[f]):.3f}" for f in FINGERS)
            med_str = "  ".join(
                f"{f}={np.median(finger_r2_all[f]):.3f}" for f in FINGERS)
            print(f"  [{name}] per-finger R2 over {n_clips} clips:")
            print(f"    mean:    {mean_str}")
            print(f"    median:  {med_str}")

        # Dataset-average residual-PCA cumulative EVR for K = 1, 2, 3.
        if topk_evr_all:
            k1, k2, k3 = np.mean(topk_evr_all, axis=0)
            print(f"  [{name}] avg residual-PCA cum-var over {len(topk_evr_all)} "
                  f"clips:  K1={k1:.2%}  K2={k2:.2%}  K3={k3:.2%}")


def main():
    parser = argparse.ArgumentParser(
        description="Load stable-grasp / transition datasets for DOF analysis.")
    parser.add_argument("--config", default="configs/grab.yaml",
                        help="Dataset config yaml (configs/grab.yaml or configs/arctic.yaml).")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--transitions", action="store_true",
                        help="Also load GRAB transition motions (GRAB config only).")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    dataset_name = cfg.name
    print(f"config={args.config} dataset={dataset_name} split={args.split}")

    lhand_layer, rhand_layer = build_hand_layers(cfg)

    datasets = {}

    sg_cls = {"grab": StableGraspGRABDataModule,
              "arctic": StableGraspARCTICDataModule}[dataset_name]
    sg = load_dataset(sg_cls, cfg, args.split, lhand_layer, rhand_layer)
    datasets[f"stable_grasp_{dataset_name}"] = sg
    describe(f"stable grasps ({dataset_name}, {args.split})", sg)

    if args.transitions:
        if dataset_name != "grab":
            parser.error("--transitions is only available for the GRAB config.")
        tr = load_dataset(TransitionMotionGRABDataModule, cfg, args.split,
                          lhand_layer, rhand_layer)
        datasets["transition_grab"] = tr
        describe(f"transitions (grab, {args.split})", tr)
        if len(tr):
            types = [tr[i]["transition_type"] for i in range(len(tr))]
            counts = {t: types.count(t) for t in sorted(set(types))}
            print(f"  types:  {counts}")

    analyze(datasets, lhand_layer, rhand_layer)


if __name__ == "__main__":
    main()
