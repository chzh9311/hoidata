"""GRAB dataset preprocessing pipeline.

Converts raw GRAB per-sequence ``.npz`` files into a single processed ``.npz``
ready for training, plus auxiliary JSON/pkl files for text descriptions, balance
weights, and text-to-length statistics.

Source: Text2HOI (https://github.com/JunukCha/Text2HOI.git)
  preprocess/preprocessing_grab.py

This mirrors ``preprocess/preprocess_arctic.py`` in structure and config-driven
CLI, but for GRAB's **rigid** objects (object x-data is 9-D: trans + rot6d; no
articulation angle) and per-subject sequences (subjects ``s1``…``s10``).  Unlike
ARCTIC, GRAB has no per-sequence description-file intervals or stable grasps, so
the ``full_motion`` path only carries the whole-sequence motion plus subject id.

Pipeline (run all steps via main()):
  1. preprocessing_object()          — sample object meshes to 1024-pt clouds → obj.pkl
  2. preprocessing_data()            — extract motion params, contacts → data.npz
  3. preprocessing_text()            — build text-description JSON
  4. preprocessing_balance_weights() — compute per-sample sampling weights
  5. preprocessing_text2length()     — build text → frame-count statistics
  6. print_text_data_num()           — sanity-check print

Expected config keys in configs/grab.yaml:
    root:                 raw GRAB data root (contains s{1..10}/*.npz)
    obj_root:             dir of object ``.ply`` meshes (contact_meshes)
    data_path:            output path for data.npz
    data_obj_pc_path:     output path for obj.pkl
    text_json:            output path for text descriptions JSON
    balance_weights_path: output path for balance_weights.pkl
    t2c_json:             output path for text-count JSON
    t2l_json:             output path for text-to-length JSON
    mano_root:            path to MANO model files
    flat_hand:            bool, use flat-hand MANO mean pose
    full_motion:          bool, save whole sequences (no clip/text/contact split)
"""

import glob
import json
import os
import os.path as osp
import pickle
import time
from collections import Counter

import hydra
import numpy as np
import torch
import tqdm
import trimesh

from hoidata.utils.hand_proc import ManoLayerWrapper
from hoidata.dataloaders.action_dataset import ROT_REPR, ObjectModel
from hoidata.preprocess.preprocess_arctic import load_stable_grasps
from hoidata.utils.frame import align_frame
from hoidata.utils.proc import (
    farthest_point_sample,
    get_contact_info,
    get_hand_org,
    process_text,
    proc_numpy,
    proc_torch_cuda,
    transform_hand_to_xdata,
    transform_obj_to_xdata,
    transform_xdata_to_joints,
)
from pytorch3d.transforms import axis_angle_to_matrix
from hoidata.constants.grab_constants import (
    grab_obj_name,
    object_proc,
    motion_proc,
    present_participle,
    third_verb,
    passive_verb,
)


# Subjects to include (GRAB ships s1…s10).
subj_list = [f"s{i}" for i in range(1, 11)]


def _build_hand_layers(cfg):
    """Left/right MANO layers on CUDA (axis-angle, 45-D finger pose convention).

    Like ARCTIC, this repo's x-data stores a full 48-D axis-angle pose (3 global +
    45 per-joint), so ``use_pca`` must be off (ncomps=45); otherwise ManoLayer
    would treat the 45-D finger pose as PCA coefficients and collapse per-frame
    articulation.  GRAB has no per-subject vtemplates here — the shared mean
    template is used (per-subject shapes are applied via betas downstream).
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


def _list_sequences(cfg):
    """Sorted list of all GRAB sequence ``.npz`` paths across subjects s1…s10."""
    data_list = []
    for subj in subj_list:
        data_list += glob.glob(osp.join(cfg.root, subj, "*.npz"))
    data_list.sort()
    return data_list


def preprocessing_object(cfg):
    """Sample each object mesh to a 1024-point cloud and save as obj.pkl.

    GRAB objects are rigid ``.ply`` meshes in a flat directory (no per-object
    subdir / articulation parts).  Meshes are already in metres, so — unlike
    ARCTIC (mm → m) — no unit conversion is applied.
    """
    object_paths = sorted(glob.glob(osp.join(cfg.obj_root, "*.ply")))

    obj_pcs = {}
    obj_pc_normals = {}
    point_sets = {}
    obj_path = {}

    for object_path in tqdm.tqdm(object_paths, desc="objects"):
        mesh = trimesh.load(object_path, maintain_order=True)
        verts = torch.FloatTensor(mesh.vertices).unsqueeze(0).cuda()
        normal = torch.FloatTensor(mesh.vertex_normals).unsqueeze(0).cuda()
        normal = normal / torch.norm(normal, dim=2, keepdim=True)

        point_set = farthest_point_sample(verts, 1024)
        sampled_pc = verts[0, point_set[0]].cpu().numpy()
        sampled_normal = normal[0, point_set[0]].cpu().numpy()

        obj_name = osp.basename(object_path).replace(".ply", "")
        obj_pcs[obj_name] = sampled_pc
        obj_pc_normals[obj_name] = sampled_normal
        point_sets[obj_name] = point_set[0].cpu().numpy()
        obj_path[obj_name] = osp.basename(object_path)

    os.makedirs(osp.dirname(cfg.data_obj_pc_path), exist_ok=True)
    with open(cfg.data_obj_pc_path, "wb") as f:
        pickle.dump(
            {
                "object_name": grab_obj_name,
                "obj_pcs": obj_pcs,
                "obj_pc_normals": obj_pc_normals,
                "point_sets": point_sets,
                "obj_path": obj_path,
            }, f)
    print(f"Saved object pkl → {cfg.data_obj_pc_path}")


def preprocessing_data(cfg):
    """Process raw GRAB sequences into a single .npz with motion + contact data.

    Each raw sequence ``.npz`` is subsampled (``::4``, ~120→30 fps) and trimmed to
    the object-contact span (first…last frame with any object contact).  Hand pose
    is fullpose (45) prefixed by global_orient (3) → 48-D axis-angle; betas are
    zeroed (GRAB stores per-subject shape separately and Text2HOI's GRAB path uses
    zero betas).  Objects are rigid, so x_obj is 9-D (trans + rot6d), no angle.

    If ``cfg.full_motion`` is True, each sequence is saved as a single
    whole-sequence sample (no clip partitioning, no contact-index computation):
    motion fields cover all (subsampled, contact-trimmed) frames, contact fields
    are None, action_name is the motion intent, and subject is carried through.
    """
    start_time = time.time()

    object_model = ObjectModel(cfg.data_obj_pc_path)
    lhand_layer, rhand_layer = _build_hand_layers(cfg)

    lists = {k: [] for k in [
        "x_lhand", "x_rhand", "j_lhand", "j_rhand",
        "x_obj", "lhand_beta", "rhand_beta", "lhand_org", "rhand_org",
    ]}
    extra = {k: [] for k in [
        "lcf_idx", "lcov_idx", "lchj_idx", "ldist_value",
        "rcf_idx", "rcov_idx", "rchj_idx", "rdist_value",
        "is_lhand", "is_rhand",
        "obj_name", "proc_obj_name", "action_name", "seq_name",
        "nframes", "sbj_id", "grasps",
    ]}

    # Per-sequence stable-grasp intervals (per side), keyed by (subject, seq_name)
    # — for GRAB the CSV's ``sid_seq_name`` is ``{sbj_id}/{seq_name}`` (e.g.
    # ``s1/airplane_lift``).  Consumed by the web viewer's grasp tracks.
    stable_grasps = load_stable_grasps(cfg.get("stable_grasp_csv", ""))

    full_motion = cfg.get("full_motion", False)

    for data_path in tqdm.tqdm(_list_sequences(cfg), desc="sequences"):
        data = np.load(data_path, allow_pickle=True)
        obj_name = data["obj_name"].item()
        action_name = data["motion_intent"].item()
        proc_obj_name = object_proc.get(obj_name, obj_name)
        action_name = motion_proc.get(action_name, action_name)
        sbj_id = osp.basename(osp.dirname(data_path))   # e.g. "s1"
        # Full raw filename (no ext), e.g. "airplane_lift" — unique per sequence,
        # unlike action_name (the motion intent, shared across objects).
        seq_name = osp.basename(data_path).replace(".npz", "")

        # Trim to the object-contact span (on the ::4-subsampled frames).
        contact = data["contact"][()]["object"][::4]
        contact_frame = np.nonzero(np.sum(contact, axis=1))
        start_frame = int(contact_frame[0][0])
        end_frame = int(contact_frame[0][-1])

        lhand_params = data["lhand"][()]["params"]
        lhand_glob_ori = lhand_params["global_orient"][::4][start_frame:end_frame]
        lhand_pose = lhand_params["fullpose"][::4][start_frame:end_frame]
        lhand_trans = lhand_params["transl"][::4][start_frame:end_frame]
        lhand_beta = np.zeros((lhand_pose.shape[0], 10))

        lhand_glob_ori = proc_torch_cuda(lhand_glob_ori)
        lhand_pose = torch.cat([lhand_glob_ori, proc_torch_cuda(lhand_pose)], dim=1)
        lhand_trans = proc_torch_cuda(lhand_trans)
        lhand_beta = proc_torch_cuda(lhand_beta)

        rhand_params = data["rhand"][()]["params"]
        rhand_glob_ori = rhand_params["global_orient"][::4][start_frame:end_frame]
        rhand_pose = rhand_params["fullpose"][::4][start_frame:end_frame]
        rhand_trans = rhand_params["transl"][::4][start_frame:end_frame]
        rhand_beta = np.zeros((rhand_pose.shape[0], 10))

        rhand_glob_ori = proc_torch_cuda(rhand_glob_ori)
        rhand_pose = torch.cat([rhand_glob_ori, proc_torch_cuda(rhand_pose)], dim=1)
        rhand_trans = proc_torch_cuda(rhand_trans)
        rhand_beta = proc_torch_cuda(rhand_beta)

        obj_params = data["object"][()]["params"]
        obj_glob_ori = obj_params["global_orient"][::4][start_frame:end_frame]
        obj_trans = obj_params["transl"][::4][start_frame:end_frame]

        obj_glob_ori = proc_torch_cuda(obj_glob_ori)
        obj_trans = proc_torch_cuda(obj_trans)
        obj_rotmat = axis_angle_to_matrix(obj_glob_ori)
        obj_extmat = torch.cat([obj_rotmat, obj_trans.unsqueeze(2)], dim=2)

        _, obj_pc, _, _ = object_model(obj_name)

        x_lhand = transform_hand_to_xdata(lhand_trans, lhand_pose)
        x_rhand = transform_hand_to_xdata(rhand_trans, rhand_pose)
        j_lhand = transform_xdata_to_joints(x_lhand, lhand_layer)
        j_rhand = transform_xdata_to_joints(x_rhand, rhand_layer)
        x_obj = transform_obj_to_xdata(obj_extmat)

        lhand_org = proc_numpy(get_hand_org(lhand_pose, lhand_beta, lhand_trans, lhand_layer))
        rhand_org = proc_numpy(get_hand_org(rhand_pose, rhand_beta, rhand_trans, rhand_layer))

        if full_motion:
            # Whole-sequence sample; contact indices not computed.
            lcf_idx = lcov_idx = lchj_idx = ldist_value = None
            rcf_idx = rcov_idx = rchj_idx = rdist_value = None
            is_lhand = is_rhand = None
        else:
            (lcf_idx, lcov_idx, lchj_idx, ldist_value,
             rcf_idx, rcov_idx, rchj_idx, rdist_value,
             is_lhand, is_rhand) = get_contact_info(
                lhand_pose, lhand_beta, lhand_trans,
                rhand_pose, rhand_beta, rhand_trans,
                obj_extmat, lhand_layer, rhand_layer,
                obj_pc, mul_rv=False,
            )
            assert is_lhand != 0 or is_rhand != 0, (
                f"No contact: {action_name} {obj_name} {start_frame}–{end_frame}")

        lists["x_lhand"].append(x_lhand)
        lists["x_rhand"].append(x_rhand)
        lists["j_lhand"].append(j_lhand)
        lists["j_rhand"].append(j_rhand)
        lists["x_obj"].append(x_obj)
        lists["lhand_beta"].append(proc_numpy(lhand_beta))
        lists["rhand_beta"].append(proc_numpy(rhand_beta))
        lists["lhand_org"].append(lhand_org)
        lists["rhand_org"].append(rhand_org)

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
        extra["obj_name"].append(obj_name)
        extra["proc_obj_name"].append(proc_obj_name)
        extra["action_name"].append(action_name)
        extra["seq_name"].append(seq_name)
        nframes = len(obj_extmat)
        extra["nframes"].append(nframes)
        extra["sbj_id"].append(sbj_id)

        # Stable-grasp intervals for this sequence (per side), clamped to the
        # frame count.  CSV frames are 0-indexed against this same trimmed
        # sequence (the finder runs on the posed nframes-length sequence).
        raw_grasps = stable_grasps.get((sbj_id, seq_name),
                                       {"left": [], "right": []})
        grasps = {"left": [], "right": []}
        for side in ("left", "right"):
            for s, e in raw_grasps[side]:
                s = max(0, min(s, nframes))
                e = max(0, min(e, nframes))
                if e > s:
                    grasps[side].append([s, e])
        extra["grasps"].append(grasps)

    final_dict = align_frame({k: lists[k] for k in [
        "x_lhand", "x_rhand", "j_lhand", "j_rhand",
        "x_obj", "lhand_beta", "rhand_beta", "lhand_org", "rhand_org",
    ]})

    ragged_keys = {
        "lcf_idx", "lcov_idx", "lchj_idx", "ldist_value",
        "rcf_idx", "rcov_idx", "rchj_idx", "rdist_value",
        "grasps",  # per-row {"left": [[s,e],...], "right": [[s,e],...]}
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
        rot_repr=np.array(ROT_REPR),   # rot6d convention stamp, checked on load
        **final_dict,
        **{k: _to_array(k, v) for k, v in extra.items()},
    )
    print(f"Saved data → {cfg.data_path}  "
          f"({len(extra['obj_name'])} clips, {time.time()-start_time:.1f}s)")


def preprocessing_text(cfg):
    """Build a JSON mapping each (action, proc_obj, hand) text key to 4 paraphrases.

    Actions/objects come from the raw sequences directly (motion_intent + obj_name,
    run through motion_proc/object_proc), matching the reference implementation.
    """
    action_list = []
    for data_path in tqdm.tqdm(_list_sequences(cfg), desc="text"):
        data = np.load(data_path, allow_pickle=True)
        motion = data["motion_intent"].item()
        motion = motion_proc.get(motion, motion)
        obj_name = data["obj_name"].item()
        obj_name = object_proc.get(obj_name, obj_name)
        action_list.append(f"{motion},{obj_name}")
    action_list = np.unique(action_list)

    text_description = {}
    for action in action_list:
        action_v, action_o = action.split(",")

        action_ving = present_participle[action_v]
        action_3rd = third_verb[action_v]
        action_pass = passive_verb[action_v]

        for hand_str, hand_cap in [
            ("left hand", "Left hand"),
            ("right hand", "Right hand"),
            ("both hands", "Both hands"),
        ]:
            key = f"{action_v} {action_o} with {hand_str}.".capitalize()
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
        proc_obj_name = data["proc_obj_name"]

    text_list = [
        process_text(action_name[i], proc_obj_name[i],
                     is_lhand[i], is_rhand[i],
                     text_descriptions=None, return_key=True)
        for i in range(len(action_name))
    ]

    text_counter = Counter(text_list)
    text_prob = {k: 1 / v for k, v in text_counter.items()}
    balance_weights = [text_prob[t] for t in text_list]

    os.makedirs(osp.dirname(cfg.balance_weights_path), exist_ok=True)
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
        proc_obj_name = data["proc_obj_name"]
        nframes = data["nframes"]

    text_dict = {}
    for i in range(len(action_name)):
        key = process_text(action_name[i], proc_obj_name[i],
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


@hydra.main(version_base=None, config_path="../configs", config_name="grab")
def main(cfg):
    preprocessing_object(cfg)
    preprocessing_data(cfg)
    if not cfg.get("full_motion", False):
        preprocessing_text(cfg)
        preprocessing_balance_weights(cfg)
        preprocessing_text2length(cfg)
        print_text_data_num(cfg)


if __name__ == "__main__":
    main()
