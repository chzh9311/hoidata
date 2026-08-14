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
from pytorch3d.transforms import axis_angle_to_matrix, rotation_6d_to_matrix
from pytorch3d.transforms.so3 import hat_inv
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


def avg_rotation_axis(Rs):
    """Dominant rotation axis of a batch of rotation matrices (T, 3, 3).

    The skew part of each R is ``sin(theta) * [u]_x``, so ``vee`` of it gives
    ``sin(theta) * u`` — a vector along that frame's rotation axis, weighted by
    how much it rotates.  Stacking those into the scatter matrix
    ``M = S^T S`` and taking the eigenvector of the largest eigenvalue yields the
    axis that best explains the whole batch (a least-squares fit, so small-angle
    frames contribute little and noise averages out).

    Note this is an *axis*, not a full mean rotation: sign is arbitrary (``u``
    and ``-u`` span the same eigenspace) and the angle is not recovered.  Frames
    rotating by ~180 deg have ``sin(theta) ~ 0`` and drop out.

    Returns a unit ``(3,)`` tensor.
    """
    sinaa = hat_inv((Rs - Rs.transpose(1, 2)) / 2)  # (T, 3) = sin(theta) * u
    M = sinaa.transpose(0, 1) @ sinaa  # (3, 3), symmetric PSD
    # eigh returns eigenvalues ascending, so the last column is the dominant axis.
    _, evecs = torch.linalg.eigh(M)
    return evecs[:, -1]


def analyze(datasets):
    """Placeholder for the DOF analysis.

    ``datasets`` is a dict {name: dataset}.  Each dataset yields items with
    ``x_hand`` (T, 99) — trans(3) + rot6d(16*6) — which is the natural input for
    measuring effective DOFs (e.g. PCA over per-frame hand pose, or per-joint
    range of motion).
    """
    print("\n[analyze] not implemented yet — datasets are loaded and ready.")
    for name, ds in datasets.items():
        print(f"  {name}: {len(ds)} clips available as ds[i]['x_hand']")
        for i, it in enumerate(ds):
            print(f"    ds[{i}]: x_hand={it['x_hand'].shape} x_object={it['x_object'].shape} "
                  f"obj={it['obj_name']} n_frames={it['n_frames']}")
            # Datasets yield numpy; the pytorch3d converters are torch, so
            # convert at the boundary.
            x_object = torch.from_numpy(it['x_object']).float()  # (T, 9|10)
            x_hand = torch.from_numpy(it['x_hand']).float()      # (T, 99)

            obj_rot = x_object[:, 3:9]  # (T, 6) rot6d
            obj_R = rotation_6d_to_matrix(obj_rot)  # (T, 3, 3)
            obj_t = x_object[:, :3]  # (T, 3) translation
            wrist_rot = x_hand[:, 3:9]  # (T, 6) rot6d
            wrist_R = rotation_6d_to_matrix(wrist_rot)  # (T, 3, 3)
            wrist_t = x_hand[:, :3]  # (T, 3) translation
            obj2wrist_R = wrist_R.transpose(1, 2) @ obj_R  # (T, 3, 3)
            sinaa = hat_inv((obj2wrist_R - obj2wrist_R.transpose(1, 2)) / 2)  # (T, 3) = sin(theta) * u
            M = sinaa.transpose(0, 1) @ sinaa  # (3, 3), symmetric PSD
            # eigh returns eigenvalues ascending, so the last column is the dominant axis.
            _, evecs = torch.linalg.eigh(M)
            axis = evecs[:, -1]  # (3,) dominant rotation axis
            # Per-frame signed angle about `axis`.  cos from the trace, sin from
            # projecting sin(theta)*u onto the axis; atan2 of the two resolves
            # the full +-180 deg range (arccos/arcsin alone would fold it).
            cos_a = (torch.diagonal(obj2wrist_R, dim1=1, dim2=2).sum(-1) - 1) / 2  # (T,)
            sin_a = sinaa @ axis  # (T,)
            angle = torch.atan2(sin_a, cos_a)  # (T,) signed, sign tied to `axis`
            recovered_aa = angle.unsqueeze(-1) * axis  # (T, 3) axis-angle per frame
            recovered_R = axis_angle_to_matrix(recovered_aa)  # (T, 3, 3) rotation matrix of the average rotation
            delta_R = obj2wrist_R.transpose(1, 2) @ recovered_R  # (T, 3, 3) difference from average rotation
            # Per-frame residual angle; clamp guards acos against |cos|>1 from roundoff.
            cos_d = (torch.diagonal(delta_R, dim1=1, dim2=2).sum(-1) - 1) / 2  # (T,)
            delta_aa = torch.acos(cos_d.clamp(-1.0, 1.0))  # (T,) angle of the difference rotation

            # Constant pivot t, assuming the clip is a rotation about a fixed point:
            #   x -> R_i (x - t) + t = R_i x + (I - R_i) t.
            # The observed relative translation is the object origin in the wrist
            # frame, so each frame contributes 3 equations (I - R_i) t = d_i.
            obj2wrist_t = torch.einsum(
                'tij,tj->ti', wrist_R.transpose(1, 2), obj_t - wrist_t)  # (T, 3)
            I = torch.eye(3, dtype=obj2wrist_R.dtype).expand_as(obj2wrist_R)  # (T, 3, 3)
            A = (I - recovered_R).reshape(-1, 3)  # (3T, 3)
            # The system is homogeneous in t (the constant is implied by the
            # model), so obj2wrist_t must NOT be mean-centred -- centring removes
            # the signal the pivot is fitted from and biases the solution.
            b = obj2wrist_t.reshape(-1)  # (3T,)
            # (I - R) is singular along `axis` for every frame, so A is rank <= 2: the
            # component of t along the axis is unobservable (sliding the pivot along
            # the axis leaves the transform unchanged).  gelsd handles the rank
            # deficiency and returns the minimum-norm solution, pinning that free
            # component to 0 -- so t is a point *on* the axis, not a unique 3-D pivot.
            pivot = torch.linalg.lstsq(A, b, driver='gelsd').solution  # (3,)
            resid = (A @ pivot - b).reshape(-1, 3).norm(dim=-1)  # (T,) per-frame fit error

            print(f"      resid_angle={(delta_aa * 180 / torch.pi).mean():.2f}deg "
                  f"pivot={pivot.numpy().round(4)} "
                  f"pivot_resid={resid.mean():.4f}m")


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

    analyze(datasets)


if __name__ == "__main__":
    main()
