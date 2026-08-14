"""Convert a preprocessed ``data.npz`` to pytorch3d's 6D rotation convention.

Before the pytorch3d migration this repo's ``rot6d`` stored the first two
**columns** of R, interleaved as ``[R00,R01,R10,R11,R20,R21]`` (see the deleted
``hoidata/utils/rot.py``).  ``pytorch3d.transforms.matrix_to_rotation_6d`` stores
the first two **rows**, ``[R00,R01,R02,R10,R11,R12]``.  The two are not
interchangeable — feeding a legacy vector to ``rotation_6d_to_matrix`` yields a
plausible-looking but wrong rotation, with no error raised.

This script rewrites only the rotation blocks of the affected fields:

    x_lhand / x_rhand : (N, T, 3 + 16*6)   -> [..., 3:]
    x_obj             : (N, T, 9 | 10)     -> [..., 3:9]

Everything else — translations, ``x_obj_angle``, joints, betas, the ragged
object-dtype arrays and all metadata — is copied through byte-identically, and a
``rot_repr`` stamp is added so ``MotionDatasetBase`` can reject unconverted
caches.  No MANO, no raw dataset and no GPU are needed; the conversion is
``legacy_6d -> R -> matrix_to_rotation_6d``.

Zero-padded frames (``align_frame`` pads past ``nframes``) map to zeros in both
conventions, so the padding is preserved; this is asserted rather than assumed.

Run in the ``hoi_common`` conda env:

    python -m scripts.migrate_rot6d data/output/grab/data.npz --dry-run
    python -m scripts.migrate_rot6d data/output/grab/data.npz
    python -m scripts.migrate_rot6d data/output/arctic/data.npz
"""

import argparse
import os
import os.path as osp
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix

from hoidata.dataloaders.action_dataset import ROT_REPR

BACKUP_SUFFIX = ".bak_before_rot6d_p3d"
#: field -> slice of the rot6d block within the last axis.
ROT6D_FIELDS = {"x_lhand": slice(3, None),
                "x_rhand": slice(3, None),
                "x_obj": slice(3, 9)}
CHUNK = 1_000_000  # rotations converted per batch


def _legacy_rot6d_to_matrix(x):
    """Verbatim ``hoidata.utils.rot.rot6d_to_rotmat`` (Gram-Schmidt on columns)."""
    x = x.reshape(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum("bi,bi->b", b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def _convert_block(block):
    """(..., n_rot*6) legacy rot6d -> pytorch3d rot6d, same shape and dtype."""
    shape = block.shape
    flat = torch.from_numpy(np.ascontiguousarray(block, dtype=np.float32)).reshape(-1, 6)
    out = torch.empty_like(flat)
    for i in range(0, flat.shape[0], CHUNK):
        chunk = flat[i:i + CHUNK]
        out[i:i + CHUNK] = matrix_to_rotation_6d(_legacy_rot6d_to_matrix(chunk))
    return out.reshape(shape).numpy().astype(block.dtype)


def _verify(old, new, rng, name):
    """Assert new and old decode to the same R, and that padding stays zero."""
    old_f = old.reshape(-1, 6)
    new_f = new.reshape(-1, 6)

    zero = np.abs(old_f).max(axis=1) == 0
    n_zero = int(zero.sum())
    if n_zero and np.abs(new_f[zero]).max() != 0:
        raise AssertionError(f"{name}: zero padding did not stay zero")

    nz = np.flatnonzero(~zero)
    if nz.size == 0:
        print(f"    {name}: no non-zero rotations to check")
        return
    sample = rng.choice(nz, size=min(20000, nz.size), replace=False)
    R_old = _legacy_rot6d_to_matrix(torch.from_numpy(old_f[sample].astype(np.float32)))
    R_new = rotation_6d_to_matrix(torch.from_numpy(new_f[sample].astype(np.float32)))
    err = (R_old - R_new).abs().max().item()
    if not err < 1e-5:
        raise AssertionError(f"{name}: R mismatch after conversion, max err {err:.3e}")
    print(f"    {name}: {nz.size} rotations, {n_zero} zero-padded, "
          f"max |R_old - R_new| = {err:.2e} over {sample.size} samples")


def migrate(path, dry_run=False):
    with np.load(path, allow_pickle=True) as data:
        keys = list(data.files)
        if "rot_repr" in keys and str(data["rot_repr"]) == ROT_REPR:
            print(f"{path}: already rot_repr={ROT_REPR!r} — nothing to do.")
            return
        missing = [k for k in ROT6D_FIELDS if k not in keys]
        if missing:
            raise KeyError(f"{path} is missing expected fields: {missing}")
        print(f"{path}: {len(keys)} arrays")
        out = {}
        rng = np.random.default_rng(0)
        for key in keys:
            arr = data[key]
            if key in ROT6D_FIELDS:
                sl = ROT6D_FIELDS[key]
                arr = arr.copy()
                old = arr[..., sl]
                new = _convert_block(old)
                _verify(old, new, rng, key)
                arr[..., sl] = new
            out[key] = arr
        out["rot_repr"] = np.array(ROT_REPR)

    if dry_run:
        print(f"{path}: dry run OK — nothing written.")
        return

    backup = path + BACKUP_SUFFIX
    if not osp.exists(backup):
        print(f"  backing up → {backup}")
        shutil.copy2(path, backup)
    else:
        print(f"  backup already exists → {backup} (kept)")

    tmp = path + ".tmp"
    print(f"  writing {tmp}")
    np.savez(tmp, **out)
    tmp_npz = tmp if tmp.endswith(".npz") else tmp + ".npz"
    os.replace(tmp_npz, path)
    print(f"{path}: migrated to rot_repr={ROT_REPR!r}.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="preprocessed data.npz file(s)")
    ap.add_argument("--dry-run", action="store_true",
                    help="convert and verify in memory without writing")
    args = ap.parse_args()
    for path in args.paths:
        migrate(path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
