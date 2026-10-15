"""HOT3D preprocessing: single-object interaction segments for ``MotionHOT3D``.

HOT3D recordings are long (~2 min) and several objects are manipulated one
after another.  This pipeline turns them into short **single-object** samples
in the GRAB-style ``data.npz`` layout consumed by
``hoidata.dataloaders.hot3d_dataset.MotionHOT3D``:

  1. preprocessing_object() — sample every dynamic object's ``.glb`` mesh to a
     1024-point cloud (+ normals) → ``obj.pkl`` (``ObjectModel`` format).
  2. preprocessing_data()   — parse every recording with
     ``MotionHOT3DMultiObj``, compute per-frame hand–object contact, split the
     recording with ``segment_by_activity`` and save one sample per segment
     → ``data.npz``.

Activity (``seg_activity``) — which frames count as "object k is interacted
with":
  * ``contact`` (default): either hand is within ``seg_contact_thr`` (m) of the
    object's surface (nearest MANO vertex ↔ dense surface samples spaced
    ``seg_contact_spacing``; see ``hand_object_contact``).
  * ``moving``: a sliding ``seg_window``-frame window over the object's
    tracked frames has max − min of any translation dim > ``seg_trans_thr`` (m)
    or any rot6d dim > ``seg_rot_thr`` (``object_moving_mask``).
Active runs with gaps < ``seg_merge_gap`` frames are joined; afterwards active
runs shorter than ``seg_min_active_len`` frames are discarded (a hand brushing
a neighbouring object for a frame or two would otherwise make those frames
multi-object and split the held object's interval).

Segmentation (``segment_by_activity``): frames with >1 object active are
dropped.  With ``seg_merge_same_obj``, consecutive active intervals of the same
object separated only by all-inactive frames are merged, absorbing gaps up to
``seg_same_obj_max_gap`` frames (any length when null).  Each remaining
single-object active interval (the segment's core, ``move_start`` /
``move_end``) is extended over the adjacent all-inactive frames (transitions),
capped at ``seg_max_transition_len`` frames when set.  Segments whose core is
shorter than ``seg_min_move_len`` frames, or whose object is tracked on fewer
than ``seg_min_obj_tracked`` (fraction) of their frames, are discarded;
segments whose object is never tracked are always discarded.  Every segment
also stores the raw per-hand contact masks with its object
(``lhand_contact`` / ``rhand_contact``), whichever activity is used.

Expected config keys (configs/hot3d.yaml): ``dataset_root``, ``obj_root``,
``data_path``, ``data_obj_pc_path``, ``mano_root``, ``merge_tol_ms``,
``headsets`` and the ``seg_*`` parameters above.

Usage:
    python -m hoidata.preprocess.preprocess_hot3d --config configs/hot3d.yaml [--force]
"""

import argparse
import json
import os
import os.path as osp
import pickle
import time

import numpy as np
import torch
import tqdm
import trimesh
from omegaconf import OmegaConf

from hoidata.dataloaders.action_dataset import ROT_REPR
from hoidata.dataloaders.hot3d_dataset import (
    MotionHOT3DMultiObj, segment_by_activity, object_moving_mask,
    hand_object_contact, merge_short_gaps, drop_short_runs)
from hoidata.utils.hand_proc import (
    ManoLayerWrapper, fix_mano_left_shapedirs, shaped_mano_layer)
from hoidata.utils.proc import farthest_point_sample, unit_vertex_normals


def segmentation_params(cfg):
    """The config values that determine ``data.npz`` (stamped into it)."""
    headsets = cfg.get("headsets", None)
    activity = str(cfg.get("seg_activity", "contact"))
    if activity not in ("contact", "moving"):
        raise ValueError(f"seg_activity must be 'contact' or 'moving', got {activity!r}")
    return {
        "activity": activity,
        "contact_thr": float(cfg.get("seg_contact_thr", 0.005)),
        "contact_spacing": float(cfg.get("seg_contact_spacing", 0.002)),
        "flat_hand": bool(cfg.get("flat_hand", True)),
        "window": int(cfg.get("seg_window", 5)),
        "trans_thr": float(cfg.get("seg_trans_thr", 0.005)),
        "rot_thr": float(cfg.get("seg_rot_thr", 0.05)),
        "merge_gap": int(cfg.get("seg_merge_gap", 10)),
        "min_active_len": int(cfg.get("seg_min_active_len", 10)),
        "max_transition_len": (None if cfg.get("seg_max_transition_len") is None
                               else int(cfg.seg_max_transition_len)),
        "min_move_len": int(cfg.get("seg_min_move_len", 0)),
        "min_obj_tracked": float(cfg.get("seg_min_obj_tracked", 0.0)),
        # Always on (not a config knob); stamped so older caches get rebuilt.
        "drop_untracked": True,
        "merge_same_obj": bool(cfg.get("seg_merge_same_obj", True)),
        "same_obj_max_gap": (None if cfg.get("seg_same_obj_max_gap") is None
                             else int(cfg.seg_same_obj_max_gap)),
        "merge_tol_ms": float(cfg.get("merge_tol_ms", 5.0)),
        "canonical_z_up": bool(cfg.get("canonical_z_up", True)),
        "headsets": sorted(headsets) if headsets else None,
    }


def object_surface_points(mesh_path, spacing=0.002, min_points=4096,
                          max_points=200000):
    """Dense object-frame surface samples (metres) for contact distances.

    The mesh vertices plus ``area / spacing**2`` area-uniform surface samples
    (clamped to ``[min_points, max_points]``, fixed seed), so the nearest-sample
    distance overestimates the true point-to-surface distance by at most about
    ``spacing / sqrt(2)``.
    """
    mesh = trimesh.load(mesh_path, process=False, force="mesh")
    n = int(np.clip(mesh.area / spacing ** 2, min_points, max_points))
    samples, _ = trimesh.sample.sample_surface(mesh, n, seed=0)
    return np.concatenate([np.asarray(mesh.vertices), samples]).astype(np.float32)


def preprocessing_object(cfg):
    """Sample each dynamic object mesh to a 1024-point cloud → obj.pkl.

    Meshes are loaded with ``force="mesh"`` so the glb node transform (mm → m)
    is applied; points are farthest-point samples of the mesh vertices, as for
    GRAB.  Keys are the object instance names (``instance.json``).
    """
    obj_root = cfg.get("obj_root", osp.join(cfg.dataset_root, "assets"))
    with open(osp.join(obj_root, "instance.json")) as f:
        instances = json.load(f)
    objects = sorted(
        (v["instance_name"], uid) for uid, v in instances.items()
        if v["instance_type"] == "object" and v["motion_type"] == "dynamic")

    obj_pcs, obj_pc_normals, point_sets, obj_path = {}, {}, {}, {}
    for name, uid in tqdm.tqdm(objects, desc="objects"):
        mesh = trimesh.load(osp.join(obj_root, f"{uid}.glb"), process=False,
                            force="mesh")
        verts = torch.FloatTensor(np.asarray(mesh.vertices)).unsqueeze(0).cuda()
        normal = unit_vertex_normals(mesh.vertices, mesh.faces, mesh.vertex_normals)
        normal = torch.FloatTensor(normal).unsqueeze(0).cuda()

        point_set = farthest_point_sample(verts, 1024)
        obj_pcs[name] = verts[0, point_set[0]].cpu().numpy()
        obj_pc_normals[name] = normal[0, point_set[0]].cpu().numpy()
        point_sets[name] = point_set[0].cpu().numpy()
        obj_path[name] = f"{uid}.glb"

    os.makedirs(osp.dirname(cfg.data_obj_pc_path), exist_ok=True)
    with open(cfg.data_obj_pc_path, "wb") as f:
        pickle.dump({
            "object_name": [name for name, _ in objects],
            "obj_pcs": obj_pcs,
            "obj_pc_normals": obj_pc_normals,
            "point_sets": point_sets,
            "obj_path": obj_path,
        }, f)
    print(f"Saved object pkl → {cfg.data_obj_pc_path}")


def preprocessing_data(cfg):
    """Segment every recording into single-object samples → data.npz."""
    start_time = time.time()
    params = segmentation_params(cfg)
    ds = MotionHOT3DMultiObj(
        dataset_root=cfg.dataset_root,
        split="all",
        clip_len=-1,
        headsets=params["headsets"],
        merge_tol_ms=params["merge_tol_ms"],
        mano_root=cfg.mano_root,
        cache_size=1,
        canonical_z_up=params["canonical_z_up"],
    )

    # MANO layers for contact: absolute-pose x-data (flat hand mean), HOT3D's
    # corrected MANO_LEFT shape blend shapes; each recording's beta is baked in
    # per recording (shaped_mano_layer).
    layer_kw = dict(flat_hand_mean=params["flat_hand"], mano_root=cfg.mano_root,
                    use_pca=False, ncomps=45)
    base_l = ManoLayerWrapper(side="left", **layer_kw).cuda()
    base_r = ManoLayerWrapper(side="right", **layer_kw).cuda()
    fix_mano_left_shapedirs(base_l, base_r)
    obj_root = cfg.get("obj_root", osp.join(cfg.dataset_root, "assets"))
    surface = {}   # uid -> object-frame surface samples

    keys = ["x_lhand", "x_rhand", "x_obj", "lhand_valid", "rhand_valid",
            "obj_valid", "lhand_contact", "rhand_contact", "is_lhand",
            "is_rhand", "nframes", "obj_name", "obj_uid", "seq_name",
            "recording", "participant", "headset", "up_axis", "start_frame",
            "end_frame", "move_start", "move_end", "beta"]
    out = {k: [] for k in keys}
    for ri in tqdm.tqdm(range(len(ds.recordings)), desc="recordings"):
        meta = ds.recordings[ri]
        rec = ds._recording(ri)
        N, K = rec["obj_valid"].shape
        for uid in rec["obj_uids"]:
            if uid not in surface:
                surface[uid] = object_surface_points(
                    osp.join(obj_root, f"{uid}.glb"), params["contact_spacing"])
        contact, _ = hand_object_contact(
            rec["x_lhand"], rec["x_rhand"], rec["lhand_valid"], rec["rhand_valid"],
            rec["x_obj"], rec["obj_valid"], [surface[u] for u in rec["obj_uids"]],
            shaped_mano_layer(base_l, rec["beta"]),
            shaped_mano_layer(base_r, rec["beta"]),
            thr=params["contact_thr"])                               # (N, K, 2)
        if params["activity"] == "contact":
            raw = contact.any(-1)
        else:
            raw = np.stack([
                object_moving_mask(rec["x_obj"][:, k], rec["obj_valid"][:, k],
                                   params["window"], params["trans_thr"],
                                   params["rot_thr"], merge_gap=1)
                for k in range(K)], axis=1) if K else np.zeros((N, 0), bool)
        # Join short gaps, then forget very short activity (brief brushes of a
        # neighbouring object would otherwise split another object's interval).
        active = np.stack([
            drop_short_runs(merge_short_gaps(raw[:, k], params["merge_gap"]),
                            params["min_active_len"])
            for k in range(K)], axis=1) if K else raw
        segments = segment_by_activity(
            active, merge_same_obj=params["merge_same_obj"],
            same_obj_max_gap=params["same_obj_max_gap"],
            max_transition_len=params["max_transition_len"])
        for seg in segments:
            if seg["move_end"] - seg["move_start"] < params["min_move_len"]:
                continue
            s, e, k = seg["start"], seg["end"], seg["obj"]
            tracked = rec["obj_valid"][s:e, k]
            # No pose of the object at all (e.g. a moving-mode segment inside a
            # long tracking gap) — nothing to show or learn from.
            if not tracked.any() or tracked.mean() < params["min_obj_tracked"]:
                continue
            uid = rec["obj_uids"][k]
            lvalid = rec["lhand_valid"][s:e].copy()
            rvalid = rec["rhand_valid"][s:e].copy()
            out["x_lhand"].append(rec["x_lhand"][s:e].copy())
            out["x_rhand"].append(rec["x_rhand"][s:e].copy())
            out["x_obj"].append(rec["x_obj"][s:e, k].copy())
            out["lhand_valid"].append(lvalid)
            out["rhand_valid"].append(rvalid)
            out["obj_valid"].append(rec["obj_valid"][s:e, k].copy())
            out["lhand_contact"].append(contact[s:e, k, 0].copy())
            out["rhand_contact"].append(contact[s:e, k, 1].copy())
            out["is_lhand"].append(bool(lvalid.any()))
            out["is_rhand"].append(bool(rvalid.any()))
            out["nframes"].append(e - s)
            out["obj_name"].append(ds.uid_to_name.get(uid, uid))
            out["obj_uid"].append(uid)
            out["seq_name"].append(f"{meta['recording_name']}:{s}-{e}")
            out["recording"].append(meta["recording_name"])
            out["participant"].append(meta["participant_id"])
            out["headset"].append(meta["headset"])
            out["up_axis"].append(ds.up_axis_of(meta))
            out["start_frame"].append(s)
            out["end_frame"].append(e)
            out["move_start"].append(seg["move_start"])
            out["move_end"].append(seg["move_end"])
            out["beta"].append(rec["beta"])

    ragged = {"x_lhand", "x_rhand", "x_obj", "lhand_valid", "rhand_valid",
              "obj_valid", "lhand_contact", "rhand_contact"}

    def _to_array(key, value):
        if key in ragged:
            arr = np.empty(len(value), dtype=object)
            arr[:] = value
            return arr
        return np.array(value)

    os.makedirs(osp.dirname(cfg.data_path), exist_ok=True)
    np.savez(
        cfg.data_path,
        rot_repr=np.array(ROT_REPR),
        seg_params=np.array(json.dumps(params, sort_keys=True)),
        **{k: _to_array(k, v) for k, v in out.items()},
    )
    print(f"Saved data → {cfg.data_path}  ({len(out['nframes'])} segments from "
          f"{len(ds.recordings)} recordings, {time.time() - start_time:.1f}s)")


def _stale_reason(cfg):
    """Why ``data.npz`` must be (re)built, or ``None`` if it is up to date."""
    if not osp.exists(cfg.data_path):
        return "missing"
    with np.load(cfg.data_path, allow_pickle=True) as data:
        stored = json.loads(str(data["seg_params"])) if "seg_params" in data.files else None
    want = json.loads(json.dumps(segmentation_params(cfg), sort_keys=True))
    if stored != want:
        return f"segmentation params changed ({stored} → {want})"
    return None


def run_missing_preprocessing(cfg, force=False):
    """Build obj.pkl / data.npz when missing (data.npz also when stale)."""
    if force or not osp.exists(cfg.data_obj_pc_path):
        print(f"[prepare_data] object point clouds → {cfg.data_obj_pc_path}")
        preprocessing_object(cfg)
    else:
        print(f"[prepare_data] object point clouds: found {cfg.data_obj_pc_path}")
    reason = "forced" if force else _stale_reason(cfg)
    if reason:
        print(f"[prepare_data] single-object segments ({reason}) → {cfg.data_path}")
        preprocessing_data(cfg)
    else:
        print(f"[prepare_data] single-object segments: found {cfg.data_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default="configs/hot3d.yaml")
    parser.add_argument("--force", action="store_true",
                        help="rebuild obj.pkl and data.npz even if present")
    args = parser.parse_args()
    run_missing_preprocessing(OmegaConf.load(args.config), force=args.force)


if __name__ == "__main__":
    main()
