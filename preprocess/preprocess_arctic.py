"""ARCTIC dataset preprocessing pipeline.

Converts raw ARCTIC per-sequence .mano.npy / .object.npy files into a
single processed .npz ready for training, plus auxiliary JSON/pkl files for
text descriptions, balance weights, and text-to-length statistics.

Source: Text2HOI (https://github.com/JunukCha/Text2HOI.git)
  preprocess/preprocessing_arctic.py

Pipeline (run all steps via main()):
  1. preprocessing_object()       — sample object meshes to 1024-pt clouds → obj.pkl
  2. preprocessing_data()         — extract motion params, contacts → data.npz
  3. preprocessing_text()         — build text-description JSON
  4. preprocessing_balance_weights() — compute per-sample sampling weights
  5. preprocessing_text2length()  — build text → frame-count statistics
  6. print_text_data_num()        — sanity-check print

Expected config keys in configs/dataset/arctic.yaml:
    root:                 raw ARCTIC data root (contains {subj}/{obj}*.mano.npy)
    text_root:            root for per-sequence description.txt files
    obj_root:             path to object_vtemplates dir (one sub-dir per object)
    data_path:            output path for data.npz
    data_obj_pc_path:     output path for obj.pkl
    text_json:            output path for text descriptions JSON
    balance_weights_path: output path for balance_weights.pkl
    t2c_json:             output path for text-count JSON
    t2l_json:             output path for text-to-length JSON
    mano_root:            path to MANO model files
    flat_hand:            bool, use flat-hand MANO mean pose
"""

import glob
import json
import os
import os.path as osp
import pickle
import re
import time
from collections import Counter, defaultdict

import hydra
import numpy as np
import torch
import tqdm
import trimesh

from utils.hand_proc import ManoLayerWrapper
from dataloaders.action_dataset import ObjectModel
from utils.object_tensors import ObjectTensors
from utils.frame import align_frame
from utils.proc import (
    farthest_point_sample,
    proc_numpy,
    proc_torch_cuda,
    transform_hand_to_xdata,
    transform_obj_to_xdata,
    transform_xdata_to_joints,
)
from utils.proc_arctic import (
    get_contact_info_arctic,
    process_hand_keypoints,
    process_object,
    process_text,
    processe_params,
)
from utils.rot import axis_angle_to_rotmat
from constants.arctic_constants import (
    arctic_obj_name,
    subj_list,
    not_save_list,
    action_verb_obj_list,
    present_participle,
    third_verb,
    passive_verb,
)


def parse_description_labels(text_file):
    """Tolerant parser for a sequence's ``description.txt`` annotation.

    Each line looks like ``<start>-<end> <action>, <hand_type>`` but the raw
    annotations are messy: hand types carry typos (``rigth``, ``hnad``,
    ``both hand``), the comma is sometimes missing, and the *same* interval is
    often listed several times (e.g. ``open, both hands`` and ``open, right
    hand``).  Per the labelling convention, an interval tagged with ``both`` —
    or with both ``left`` and ``right`` across its lines — is resolved to
    ``both``; otherwise it takes whichever single side it carries.

    Frame numbers in the file are 1-indexed inclusive; the returned intervals are
    0-indexed with an exclusive ``end`` (``start = sf - 1``, ``end = ef``), so
    they index directly into the full-sequence (``full_motion``) frames.

    Returns a list of dicts sorted by (start, end):
        {"start": int, "end": int, "hand": "left"|"right"|"both", "action": str}
    """
    groups = defaultdict(lambda: {"hands": set(), "action": None})
    with open(text_file) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            m = re.match(r"\s*(\d+)\s*-\s*(\d+)\s+(.*)$", line)
            if not m:
                continue
            sf, ef = int(m.group(1)), int(m.group(2))
            rest = m.group(3).replace(",", " ").lower()

            if "both" in rest:
                hand = "both"
            else:
                has_l = "left" in rest or "lhand" in rest
                has_r = bool(re.search(r"rig?th|right|rihgt|rhand", rest))
                if has_l and has_r:
                    hand = "both"
                elif has_l:
                    hand = "left"
                elif has_r:
                    hand = "right"
                else:
                    continue  # unparseable hand token — skip the line

            action = re.sub(
                r"\b(left|right|rigth|rihgt|both|hand|hands|hnad)\b", "", rest)
            action = re.sub(r"\s+", " ", action).strip()

            g = groups[(sf, ef)]
            g["hands"].add(hand)
            if g["action"] is None and action:
                g["action"] = action

    labels = []
    for (sf, ef), g in groups.items():
        hands = g["hands"]
        if "both" in hands or ("left" in hands and "right" in hands):
            hand = "both"
        elif "left" in hands:
            hand = "left"
        elif "right" in hands:
            hand = "right"
        else:
            continue
        start, end = sf - 1, ef
        if end <= start:
            continue  # drop degenerate intervals
        labels.append({"start": start, "end": end, "hand": hand,
                       "action": g["action"] or ""})
    labels.sort(key=lambda d: (d["start"], d["end"]))
    return labels


def preprocessing_object(cfg):
    """Sample each object mesh to a 1024-point cloud and save as obj.pkl."""
    obj_pcs = {}
    obj_pc_normals = {}
    obj_pc_top = {}
    point_sets = {}
    obj_path = {}

    for obj_name in tqdm.tqdm(arctic_obj_name, desc="objects"):
        object_folder = osp.join(cfg.obj_root, obj_name)
        object_path = osp.join(object_folder, "mesh.obj")
        parts_path = osp.join(object_folder, "parts.json")
        with open(parts_path) as f:
            parts = np.array(json.load(f))   # 0: bottom, 1: top

        mesh = trimesh.load(object_path, maintain_order=True)
        verts = torch.FloatTensor(mesh.vertices).unsqueeze(0).cuda()
        normal = torch.FloatTensor(mesh.vertex_normals).unsqueeze(0).cuda()
        normal = normal / torch.norm(normal, dim=2, keepdim=True)

        point_set = farthest_point_sample(verts, 1024)
        sampled_pc = verts[0, point_set[0]].cpu().numpy() / 1000   # mm → m
        sampled_normal = normal[0, point_set[0]].cpu().numpy()
        sampled_top = 1 - parts[point_set[0].cpu().numpy()]         # 0→bottom 1→top

        obj_pcs[obj_name] = sampled_pc
        obj_pc_normals[obj_name] = sampled_normal
        obj_pc_top[obj_name] = sampled_top
        point_sets[obj_name] = point_set[0].cpu().numpy()
        obj_path[obj_name] = f"{obj_name}/mesh.obj"

    os.makedirs(osp.dirname(cfg.data_obj_pc_path), exist_ok=True)
    with open(cfg.data_obj_pc_path, "wb") as f:
        pickle.dump(
            {
                "object_name": arctic_obj_name,
                "obj_pcs": obj_pcs,
                "obj_pc_normals": obj_pc_normals,
                "obj_pc_top": obj_pc_top,
                "point_sets": point_sets,
                "obj_path": obj_path,
            }, f)
    print(f"Saved object pkl → {cfg.data_obj_pc_path}")


def preprocessing_data(cfg):
    """Process raw ARCTIC sequences into a single .npz with motion + contact data.

    If cfg.full_motion is True, each raw sequence is saved as a single whole-sequence
    sample (no clip partitioning, no text/contact processing): motion fields
    (x_*hand, j_*hand, x_obj, x_obj_angle, *hand_beta, *hand_org) cover all frames,
    contact fields (l/r cf/cov/chj_idx, dist_value, is_lhand, is_rhand) are left None,
    action_name is the sequence file name (e.g. box_grab_01), and nframes is the
    sequence length.
    """
    start_time = time.time()

    object_model = ObjectModel(cfg.data_obj_pc_path)
    arctic_object_model = ObjectTensors(cfg.obj_root)
    arctic_object_model.to("cuda")

    # This repo's x-data convention stores a full 48-D axis-angle pose (3 global
    # + 45 per-joint, one 3-vector per finger joint), not manopth's default 6-D
    # PCA coefficient pose — use_pca must be off (ncomps=45) or ManoLayer slices
    # the finger pose down to 6 dims and treats them as PCA coefficients,
    # silently collapsing per-frame finger articulation into a near-constant
    # shape (j_lhand/j_rhand below would then carry wrong joint positions).
    use_pca = cfg.get("use_pca", False)
    ncomps = 6 if use_pca else 45
    lhand_layer = ManoLayerWrapper(
        side='left', flat_hand_mean=cfg.flat_hand, mano_root=cfg.mano_root,
        use_pca=use_pca, ncomps=ncomps,
    ).cuda()
    rhand_layer = ManoLayerWrapper(
        side='right', flat_hand_mean=cfg.flat_hand, mano_root=cfg.mano_root,
        use_pca=use_pca, ncomps=ncomps,
    ).cuda()

    lists = {k: [] for k in [
        "x_lhand", "x_rhand", "j_lhand", "j_rhand",
        "x_obj", "x_obj_angle",
        "lhand_beta", "rhand_beta", "lhand_org", "rhand_org",
    ]}
    extra = {k: [] for k in [
        "lcf_idx", "lcov_idx", "lchj_idx", "ldist_value",
        "rcf_idx", "rcov_idx", "rchj_idx", "rdist_value",
        "is_lhand", "is_rhand",
        "object_name", "action_name", "nframes",
        "subject", "labels",
    ]}

    full_motion = cfg.full_motion

    for subj in subj_list:
        for obj_name in tqdm.tqdm(arctic_obj_name, desc=subj):
            hand_paths = sorted(glob.glob(
                osp.join(cfg.root, subj, f"{obj_name}*.mano.npy")))
            obj_paths = sorted(glob.glob(
                osp.join(cfg.root, subj, f"{obj_name}*.object.npy")))

            point_set, _, _, _, _ = object_model(obj_name)

            for hand_path, obj_path in zip(hand_paths, obj_paths):
                hand_data, object_data = processe_params(hand_path, obj_path)

                lhand_poses = hand_data["left.pose"]    # (T, 48)
                lhand_betas = hand_data["left.shape"]   # (T, 10)
                lhand_trans = hand_data["left.trans"]   # (T, 3)
                rhand_poses = hand_data["right.pose"]
                rhand_betas = hand_data["right.shape"]
                rhand_trans = hand_data["right.trans"]

                # World-space joints for org positions (used below).
                lhand_joints = proc_numpy(
                    process_hand_keypoints(lhand_poses, lhand_betas, lhand_trans, lhand_layer))
                rhand_joints = proc_numpy(
                    process_hand_keypoints(rhand_poses, rhand_betas, rhand_trans, rhand_layer))

                obj_angle = object_data["object.angle"]         # (T, 1)
                obj_rots = object_data["object.global_rot"]     # (T, 3)
                obj_trans = object_data["object.trans"]         # (T, 3) mm

                obj_dict = process_object(
                    obj_angle, obj_rots, obj_trans / 1000,
                    obj_name, arctic_object_model,
                )
                obj_verts = obj_dict["v"]   # (T, V_max, 3) metres

                if full_motion:
                    # Save the whole sequence as a single sample; no clipping,
                    # no text/contact processing.
                    nframes = len(lhand_poses)
                    seq_name = osp.basename(obj_path).replace(".object.npy", "")

                    # Parse the per-sequence annotation into frame-labelled
                    # intervals (left/right/both), clamped to this sequence's
                    # frame count.  The web viewer colours the progress bar and
                    # recolours the meshes inside these intervals.
                    desc_file = osp.join(
                        cfg.text_root, subj, seq_name, "description.txt")
                    if osp.exists(desc_file):
                        labels = parse_description_labels(desc_file)
                        for lab in labels:
                            lab["start"] = max(0, min(lab["start"], nframes))
                            lab["end"] = max(0, min(lab["end"], nframes))
                        labels = [l for l in labels if l["end"] > l["start"]]
                    else:
                        labels = []

                    obj_rot_th = proc_torch_cuda(obj_rots)
                    obj_trans_th = proc_torch_cuda(obj_trans) / 1000  # mm → m
                    obj_rotmat_th = axis_angle_to_rotmat(obj_rot_th)
                    obj_extmat_th = torch.cat(
                        [obj_rotmat_th, obj_trans_th.unsqueeze(2)], dim=2)

                    x_lhand = transform_hand_to_xdata(
                        proc_torch_cuda(lhand_trans), proc_torch_cuda(lhand_poses))
                    x_rhand = transform_hand_to_xdata(
                        proc_torch_cuda(rhand_trans), proc_torch_cuda(rhand_poses))
                    j_lhand = transform_xdata_to_joints(x_lhand, lhand_layer)
                    j_rhand = transform_xdata_to_joints(x_rhand, rhand_layer)
                    x_obj = transform_obj_to_xdata(obj_extmat_th)

                    lists["x_lhand"].append(x_lhand)
                    lists["x_rhand"].append(x_rhand)
                    lists["j_lhand"].append(j_lhand)
                    lists["j_rhand"].append(j_rhand)
                    lists["x_obj"].append(x_obj)
                    lists["x_obj_angle"].append(obj_angle)
                    lists["lhand_beta"].append(lhand_betas)
                    lists["rhand_beta"].append(rhand_betas)
                    lists["lhand_org"].append(lhand_joints[:, 0])
                    lists["rhand_org"].append(rhand_joints[:, 0])

                    extra["lcf_idx"].append(None)
                    extra["lcov_idx"].append(None)
                    extra["lchj_idx"].append(None)
                    extra["ldist_value"].append(None)
                    extra["rcf_idx"].append(None)
                    extra["rcov_idx"].append(None)
                    extra["rchj_idx"].append(None)
                    extra["rdist_value"].append(None)
                    extra["is_lhand"].append(None)
                    extra["is_rhand"].append(None)
                    extra["object_name"].append(obj_name)
                    extra["action_name"].append(seq_name)
                    extra["nframes"].append(nframes)
                    extra["subject"].append(subj)
                    extra["labels"].append(labels)

                else:
                    # Read annotation file for this sequence.
                    seq_basename = osp.basename(obj_path).replace(".object.npy", "")
                    text_file = osp.join(
                        cfg.text_root, subj, seq_basename, "description.txt")
                    with open(text_file) as f:
                        text_lines = f.readlines()

                    seen_frame_hand = []
                    for line in text_lines:
                        if line.strip() == "":
                            continue
                        start_end, action_name, hand_type, _ = line.split(" ")

                        # Deduplication guard from the original preprocessing.
                        tag = f"{start_end} {hand_type}"
                        if tag in seen_frame_hand:
                            raise ValueError(f"Duplicate annotation: {tag} in {text_file}")
                        seen_frame_hand.append(tag)

                        action_name = action_name.replace(",", "")
                        check_text = f"{action_name} {obj_name}"
                        if check_text in not_save_list:
                            continue

                        sf, ef = start_end.split("-")
                        sf = int(sf) - 1
                        ef = int(ef) + 1 - 1
                        if ef - sf < 20:
                            continue

                        hand_type = hand_type.strip()

                        lp = lhand_poses[sf:ef]
                        lb = lhand_betas[sf:ef]
                        lt = lhand_trans[sf:ef]
                        rp = rhand_poses[sf:ef]
                        rb = rhand_betas[sf:ef]
                        rt = rhand_trans[sf:ef]

                        # Object extrinsic matrix for contact computation.
                        obj_rot_th = proc_torch_cuda(obj_rots[sf:ef])
                        obj_trans_th = proc_torch_cuda(obj_trans[sf:ef]) / 1000   # mm → m
                        obj_rotmat_th = axis_angle_to_rotmat(obj_rot_th)
                        obj_extmat_th = torch.cat(
                            [obj_rotmat_th, obj_trans_th.unsqueeze(2)], dim=2)

                        (lcf_idx, lcov_idx, lchj_idx, ldist_value,
                        rcf_idx, rcov_idx, rchj_idx, rdist_value,
                        is_lhand, is_rhand) = get_contact_info_arctic(
                            lp, lb, lt,
                            rp, rb, rt,
                            obj_verts[sf:ef, point_set],
                            lhand_layer, rhand_layer,
                        )
                        assert is_lhand != 0 or is_rhand != 0, (
                            f"No contact: {check_text} frames {sf}–{ef}")

                        if hand_type == "both":
                            assert is_lhand == 1 and is_rhand == 1
                        elif hand_type == "left":
                            assert is_lhand == 1
                            is_rhand = 0
                        elif hand_type == "right":
                            assert is_rhand == 1
                            is_lhand = 0
                        else:
                            raise ValueError(f"Unknown hand_type: {hand_type!r}")

                        x_lhand = transform_hand_to_xdata(
                            proc_torch_cuda(lt), proc_torch_cuda(lp))
                        x_rhand = transform_hand_to_xdata(
                            proc_torch_cuda(rt), proc_torch_cuda(rp))
                        j_lhand = transform_xdata_to_joints(x_lhand, lhand_layer)
                        j_rhand = transform_xdata_to_joints(x_rhand, rhand_layer)
                        x_obj = transform_obj_to_xdata(obj_extmat_th)

                        lists["x_lhand"].append(x_lhand)
                        lists["x_rhand"].append(x_rhand)
                        lists["j_lhand"].append(j_lhand)
                        lists["j_rhand"].append(j_rhand)
                        lists["x_obj"].append(x_obj)
                        lists["x_obj_angle"].append(obj_angle[sf:ef])
                        lists["lhand_beta"].append(lb)
                        lists["rhand_beta"].append(rb)
                        lists["lhand_org"].append(lhand_joints[sf:ef, 0])
                        lists["rhand_org"].append(rhand_joints[sf:ef, 0])

                        extra["lcf_idx"].append(lcf_idx)
                        extra["lcov_idx"].append(lcov_idx)
                        extra["lchj_idx"].append(lchj_idx)
                        extra["ldist_value"].append(ldist_value)
                        extra["rcf_idx"].append(rcf_idx)
                        extra["rcov_idx"].append(rcov_idx)
                        extra["rchj_idx"].append(rchj_idx)
                        extra["rdist_value"].append(rdist_value)
                        extra["is_lhand"].append(is_lhand)
                        extra["is_rhand"].append(is_rhand)
                        extra["object_name"].append(obj_name)
                        extra["action_name"].append(action_name)
                        extra["nframes"].append(ef - sf)
                        # subject/labels are only consumed by the full_motion
                        # (whole-sequence) viewer path; keep the columns aligned
                        # for the clip path with the subject and no intervals.
                        extra["subject"].append(subj)
                        extra["labels"].append([])

    final_dict = align_frame({k: lists[k] for k in [
        "x_lhand", "x_rhand", "j_lhand", "j_rhand",
        "x_obj", "lhand_beta", "rhand_beta", "lhand_org", "rhand_org",
    ]})

    # x_obj_angle is ragged (variable T), align separately.
    max_nframes = max(len(a) for a in lists["x_obj_angle"])
    x_obj_angle_arr = np.zeros((len(lists["x_obj_angle"]), max_nframes, 1),
                               dtype=np.float32)
    for i, ang in enumerate(lists["x_obj_angle"]):
        x_obj_angle_arr[i, :len(ang)] = ang
    final_dict["x_obj_angle"] = x_obj_angle_arr

    ragged_keys = {
        "lcf_idx", "lcov_idx", "lchj_idx", "ldist_value",
        "rcf_idx", "rcov_idx", "rchj_idx", "rdist_value",
        "labels",  # per-row list of interval dicts (variable length)
    }

    def _to_array(key, value):
        if key in ragged_keys:
            arr = np.empty(len(value), dtype=object)
            arr[:] = value
            return arr
        return np.array(value)

    os.makedirs(osp.dirname(cfg.data_path), exist_ok=True)
    np.savez(
        cfg.data_path,
        **final_dict,
        **{k: _to_array(k, v) for k, v in extra.items()},
    )
    print(f"Saved data → {cfg.data_path}  "
          f"({len(extra['object_name'])} clips, {time.time()-start_time:.1f}s)")


def preprocessing_text(cfg):
    """Build a JSON mapping each (action_verb_obj, hand) text key to 4 paraphrases."""
    text_description = {}
    for action in action_verb_obj_list:
        # action is "verb object", e.g. "close laptop"
        parts = action.split(" ")
        action_v = " ".join(parts[:-1])
        action_o = parts[-1]

        action_ving = present_participle[action_v]
        action_3rd = third_verb[action_v]
        action_pass = passive_verb[action_v]

        for hand_str, hand_cap in [
            ("left hand", "Left hand"),
            ("right hand", "Right hand"),
            ("both hands", "Both hands"),
        ]:
            key = f"{action} with {hand_str}.".capitalize()
            text_description[key] = [
                key,
                f"{action_ving} {action_o} with {hand_str}.".capitalize(),
                f"{hand_cap} {action_3rd} {action_o}.",
                f"{action_o} {action_pass} with {hand_str}.".capitalize(),
            ]

    os.makedirs(osp.dirname(cfg.text_json), exist_ok=True)
    with open(cfg.text_json, "w") as f:
        json.dump(text_description, f)
    print(f"Saved text JSON → {cfg.text_json}")


def preprocessing_balance_weights(cfg):
    """Compute inverse-frequency sampling weights and save as pkl + count JSON."""
    with np.load(cfg.data_path, allow_pickle=True) as data:
        is_lhand = data["is_lhand"]
        is_rhand = data["is_rhand"]
        action_name = data["action_name"]
        object_name = data["object_name"]

    text_list = [
        process_text(action_name[i], object_name[i],
                     is_lhand[i], is_rhand[i],
                     text_descriptions=None, return_key=True)
        for i in range(len(action_name))
    ]

    text_counter = Counter(text_list)
    text_prob = {k: 1 / v for k, v in text_counter.items()}
    balance_weights = [text_prob[t] for t in text_list]

    with open(cfg.balance_weights_path, "wb") as f:
        pickle.dump(balance_weights, f)
    with open(cfg.t2c_json, "w") as f:
        json.dump(dict(text_counter), f)
    print(f"Saved balance weights → {cfg.balance_weights_path}")


def preprocessing_text2length(cfg):
    """Build a JSON mapping each text key to a list of observed frame counts."""
    with np.load(cfg.data_path, allow_pickle=True) as data:
        is_lhand = data["is_lhand"]
        is_rhand = data["is_rhand"]
        action_name = data["action_name"]
        object_name = data["object_name"]
        nframes = data["nframes"]

    text_dict = {}
    for i in range(len(action_name)):
        key = process_text(action_name[i], object_name[i],
                           is_lhand[i], is_rhand[i],
                           text_descriptions=None, return_key=True)
        n = min(int(nframes[i]), 150)
        text_dict.setdefault(key, []).append(n)

    with open(cfg.t2l_json, "w") as f:
        json.dump(text_dict, f)
    print(f"Saved text2length JSON → {cfg.t2l_json}")


def print_text_data_num(cfg):
    with np.load(cfg.data_path, allow_pickle=True) as data:
        print(f"data num: {len(data['action_name'])}")
    with open(cfg.t2l_json) as f:
        print(f"text num: {len(json.load(f))}")



@hydra.main(version_base=None, config_path="../configs", config_name="arctic")
def main(cfg):
    preprocessing_object(cfg)
    preprocessing_data(cfg)
    if not cfg.full_motion:
        preprocessing_text(cfg)
        preprocessing_balance_weights(cfg)
        preprocessing_text2length(cfg)
        print_text_data_num(cfg)


if __name__ == "__main__":
    main()
