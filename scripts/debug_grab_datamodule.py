"""Debugging harness for ``MotionGRABDataModule``.

Instantiates the GRAB LightningDataModule exactly the way the training/eval code
does — build the shared MANO layers, construct the module, run ``prepare_data()``
(which fills in any missing preprocessing) and ``setup()``, then pull one batch
off the requested dataloader and print its structure.

Unlike ``build_dataset`` in ``hoidata.vis.o3d_visualizer`` (which bypasses the
DataModule and instantiates ``MotionGRAB`` directly), this goes through the real
``MotionGRABDataModule`` code path, so it's the right place to set breakpoints on
``prepare_data`` / ``setup`` / ``_make_dataset`` / ``__getitem__``.

Run under the ``hoi_common`` conda env (CUDA 12.4):

    conda run -n hoi_common python -m scripts.debug_grab_datamodule
    conda run -n hoi_common python -m scripts.debug_grab_datamodule --task evaluate --split test
    conda run -n hoi_common python -m scripts.debug_grab_datamodule --task generation --index 0

Or launch it from VS Code via the "Debug GRAB DataModule" configuration in
.vscode/launch.json (uses ${command:pickArgs} so you can pass flags interactively).
"""

import argparse

import numpy as np
import torch
from omegaconf import OmegaConf

from hoidata.utils.hand_proc import ManoLayerWrapper
from hoidata.dataloaders.grab_dataset import MotionGRABDataModule


def _build_hand_layers(cfg):
    """Shared left/right MANO layers on CUDA (48-D axis-angle pose convention).

    Mirrors ``preprocess_grab._build_hand_layers`` / ``o3d_visualizer.build_dataset``:
    ``use_pca`` must be off (ncomps=45) so the finger pose is treated as axis-angle,
    not PCA coefficients.
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


def _describe(name, value, indent="  "):
    """Print a compact, type-aware one-liner for a sample/batch entry."""
    if isinstance(value, torch.Tensor):
        print(f"{indent}{name:<20} Tensor  shape={tuple(value.shape)} "
              f"dtype={value.dtype} device={value.device}")
    elif isinstance(value, np.ndarray):
        print(f"{indent}{name:<20} ndarray shape={value.shape} dtype={value.dtype}")
    elif isinstance(value, (list, tuple)):
        print(f"{indent}{name:<20} {type(value).__name__:<7} len={len(value)} "
              f"-> {value if len(value) <= 4 else value[:4]}")
    elif isinstance(value, dict):
        print(f"{indent}{name:<20} dict    keys={list(value.keys())}")
    else:
        print(f"{indent}{name:<20} {type(value).__name__:<7} {value!r}")


def main():
    parser = argparse.ArgumentParser(description="Debug MotionGRABDataModule.")
    parser.add_argument("--config", default="configs/grab.yaml",
                        help="GRAB dataset config yaml.")
    parser.add_argument("--task", default="evaluate",
                        choices=["evaluate", "generation", "action_classify"],
                        help="DataModule task (selects which splits/getitem path).")
    parser.add_argument("--split", default="test",
                        help="Split to pull a sample from (evaluate only uses test).")
    parser.add_argument("--index", type=int, default=0,
                        help="Sample index to fetch directly from the dataset.")
    parser.add_argument("--batch", action="store_true",
                        help="Also pull one batch through the DataLoader (collated).")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers (keep 0 for breakpoint debugging).")
    args = parser.parse_args()

    print(f"[1] Loading config: {args.config}")
    cfg = OmegaConf.load(args.config)

    # The 'evaluate' task expects full-length sequences (max_nframes == -1), which
    # is how build_dataset() runs it: __getitem__ then takes the _getitem_full
    # path.  configs/grab.yaml sets max_nframes for the windowed generation task,
    # so override it here for evaluate (otherwise _getitem_windowed tries to read
    # the contact indices that evaluate intentionally leaves as None).
    if args.task == "evaluate":
        cfg.max_nframes = -1

    print("[2] Building shared MANO hand layers (CUDA) ...")
    lhand_layer, rhand_layer = _build_hand_layers(cfg)

    print(f"[3] Constructing MotionGRABDataModule (task={args.task}) ...")
    dm = MotionGRABDataModule(
        cfg, lhand_layer, rhand_layer,
        task=args.task, batch_size=1, num_workers=args.num_workers,
    )

    print("[4] prepare_data()  (runs any missing preprocessing) ...")
    dm.prepare_data()

    print("[5] setup() ...")
    # 'evaluate' only builds the test set; 'fit' builds train (+ val for classify).
    stage = "test" if args.task == "evaluate" else None
    dm.setup(stage=stage)

    # Pick the dataset that setup() populated for this task.
    if args.task == "evaluate":
        ds = dm.test_set
        dl = dm.test_dataloader()
    elif args.task == "action_classify":
        ds = getattr(dm, "train_set", None) or dm.test_set
        dl = dm.train_dataloader() if hasattr(dm, "train_set") else dm.test_dataloader()
    else:  # generation
        ds = dm.train_set
        dl = dm.train_dataloader()

    print(f"\n[6] Dataset built: {type(ds).__name__}  len={len(ds)}")
    print(f"    subject_layers set: {ds.subject_layers is not None} "
          f"(None => shared mean-template MANO, no per-subject vtemplate)")
    print(f"    sbj_id available:   {getattr(ds, 'sbj_id', None) is not None}")
    print(f"    seq_name available: {getattr(ds, 'seq_name', None) is not None}")

    idx = max(0, min(args.index, len(ds) - 1))
    print(f"\n[7] Fetching ds[{idx}] directly (__getitem__) ...")
    item = ds[idx]
    if isinstance(item, dict):
        for k, v in item.items():
            _describe(k, v)
    else:
        _describe("item", item)

    if args.batch:
        print(f"\n[8] Pulling one collated batch from {type(ds).__name__} DataLoader ...")
        batch = next(iter(dl))
        if isinstance(batch, dict):
            for k, v in batch.items():
                _describe(k, v)
        else:
            _describe("batch", batch)

    print("\nDone.")


if __name__ == "__main__":
    main()
