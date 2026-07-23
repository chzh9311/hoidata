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

from hoidata.utils.hand_proc import ManoLayerWrapper
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
            if i >= 2:
                break


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
