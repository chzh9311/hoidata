"""ARCTIC-specific preprocessing utilities.

Source: Text2HOI (https://github.com/JunukCha/Text2HOI.git)
  lib/utils/proc_arctic.py  (adapted to this repo's MANO calling convention)
  lib/utils/data.py         (processe_params)

MANO calling convention in this repo (ManoLayer_C):
    hand_layer(pose_48d, betas_10d, trans_3d)
  where pose_48d = [global_orient_3d || hand_pose_45d] in axis-angle.

Text2HOI uses:
    hand_layer(betas, global_orient_3d, hand_pose_45d) + manual trans add.
All hand-layer calls below use the convention of *this* repo.
"""

import numpy as np
import torch

from hoidata.utils.proc import (
    proc_numpy,
    proc_torch_cuda,
    proc_torch_frame,
    get_contact_idx,
    get_hand_object_dist,
    get_which_hands_inter,
)


# ---------------------------------------------------------------------------
# Params loader
# ---------------------------------------------------------------------------

def processe_params(hand_data_path, obj_data_path):
    """Load raw ARCTIC .mano.npy and .object.npy files.

    Returns:
        hand_data: dict with keys
            "left.pose"  (T, 48), "left.shape"  (T, 10), "left.trans"  (T, 3)
            "right.pose" (T, 48), "right.shape" (T, 10), "right.trans" (T, 3)
        object_data: dict with keys
            "object.angle" (T, 1), "object.global_rot" (T, 3), "object.trans" (T, 3)
    """
    raw_hand = np.load(hand_data_path, allow_pickle=True)[()]
    raw_obj = np.load(obj_data_path, allow_pickle=True)

    def _load_side(side_data):
        global_rot = side_data["rot"]           # (T, 3)
        pose = side_data["pose"]                # (T, 45)
        full_pose = np.concatenate([global_rot, pose], axis=1)   # (T, 48)
        shape = side_data["shape"]              # (10,)
        T = full_pose.shape[0]
        shape_rep = np.tile(shape, (T, 1))      # (T, 10)
        trans = side_data["trans"]              # (T, 3)
        return full_pose, shape_rep, trans

    l_pose, l_shape, l_trans = _load_side(raw_hand["left"])
    r_pose, r_shape, r_trans = _load_side(raw_hand["right"])

    hand_data = {
        "left.pose": l_pose,
        "left.shape": l_shape,
        "left.trans": l_trans,
        "right.pose": r_pose,
        "right.shape": r_shape,
        "right.trans": r_trans,
    }

    object_data = {
        "object.angle": raw_obj[:, :1],         # (T, 1)
        "object.global_rot": raw_obj[:, 1:4],   # (T, 3)
        "object.trans": raw_obj[:, 4:],         # (T, 3)
    }
    return hand_data, object_data


# ---------------------------------------------------------------------------
# Text helper (identical to GRAB's process_text; duplicated here so imports
# stay within common/utils/proc_arctic.py without a circular dependency)
# ---------------------------------------------------------------------------

def process_text(action_name, object_name, is_lhand, is_rhand,
                 text_descriptions, return_key=False):
    if is_lhand and is_rhand:
        text = f"{action_name} {object_name} with both hands."
    elif is_lhand:
        text = f"{action_name} {object_name} with left hand."
    else:
        text = f"{action_name} {object_name} with right hand."
    text_key = text.capitalize()
    if return_key:
        return text_key
    return np.random.choice(text_descriptions[text_key])


# ---------------------------------------------------------------------------
# Hand keypoints
# ---------------------------------------------------------------------------

def process_hand_keypoints(hand_poses, hand_betas, hand_trans, hand_layer):
    """Run MANO forward pass and return world-space joints with fingertips.

    Args:
        hand_poses:  numpy (T, 48) axis-angle pose.
        hand_betas:  numpy (T, 10) shape betas.
        hand_trans:  numpy (T, 3)  wrist translation.
        hand_layer:  ManoLayer_C instance (this repo's convention).

    Returns:
        joints_w_tip: torch cuda (T, 21, 3) in metres.
    """
    poses_th = proc_torch_cuda(hand_poses)   # (T, 48)
    betas_th = proc_torch_cuda(hand_betas)   # (T, 10)
    trans_th = proc_torch_cuda(hand_trans)   # (T, 3)
    out = hand_layer(poses_th, betas_th, trans_th)
    return out.joints_w_tip                  # (T, 21, 3)


# ---------------------------------------------------------------------------
# Articulated object forward pass
# ---------------------------------------------------------------------------

def process_object(obj_angle, obj_rot, obj_trans, obj_name, object_model):
    """Articulate and globally-transform an ARCTIC object for every frame.

    Args:
        obj_angle:     numpy (T, 1)  articulation angles in radians.
        obj_rot:       numpy (T, 3)  axis-angle global rotation.
        obj_trans:     numpy (T, 3)  translation in **metres**.
        obj_name:      str           object name.
        object_model:  ObjectTensors instance.

    Returns:
        dict as returned by ObjectTensors.forward(); key ``"v"`` has shape
        (T, V_max, 3) world-space vertices in metres.
    """
    T = obj_angle.shape[0]
    names = [obj_name] * T
    angles_th = proc_torch_cuda(obj_angle)   # (T, 1)
    rot_th = proc_torch_cuda(obj_rot)        # (T, 3)
    trans_th = proc_torch_cuda(obj_trans)    # (T, 3)
    return object_model(angles_th, rot_th, trans_th, names)


# ---------------------------------------------------------------------------
# Contact info
# ---------------------------------------------------------------------------

def get_contact_info_arctic(
    lhand_pose_list, lhand_beta_list, lhand_trans_list,
    rhand_pose_list, rhand_beta_list, rhand_trans_list,
    sampled_obj_pc_tf,
    lhand_layer, rhand_layer,
):
    """Compute hand-object contact indices and distances for ARCTIC sequences.

    The object point cloud ``sampled_obj_pc_tf`` must already be in world space
    (i.e. articulated and globally-rotated/translated for each frame):
    shape (T, K, 3) in metres.

    Hand layers use this repo's ManoLayer_C convention:
        hand_layer(pose_48d, betas_10d, trans_3d)

    Returns:
        (lcf_idx, lcov_idx, lchj_idx, ldist_value,
         rcf_idx, rcov_idx, rchj_idx, rdist_value,
         is_lhand, is_rhand)
    """
    contact_threshold = 0.02
    sampled_obj_pc_tf = proc_torch_cuda(sampled_obj_pc_tf)

    def _side_contact(pose_list, beta_list, trans_list, hand_layer):
        if len(pose_list) == 0:
            empty = np.array([])
            return empty, empty, empty, empty
        pl = proc_torch_frame(pose_list)    # (T, 48)
        bl = proc_torch_frame(beta_list)    # (T, 10)
        tl = proc_torch_frame(trans_list)   # (T, 3)
        out = hand_layer(pl, bl, tl)
        joints = out.joints_w_tip            # (T, 21, 3)
        dist = get_hand_object_dist(joints, sampled_obj_pc_tf)
        cf_idx, cov_idx, chj_idx = get_contact_idx(dist, contact_threshold)
        dist_value = dist[cf_idx, cov_idx, chj_idx]
        return (proc_numpy(cf_idx), proc_numpy(cov_idx),
                proc_numpy(chj_idx), proc_numpy(dist_value))

    lcf_idx, lcov_idx, lchj_idx, ldist_value = _side_contact(
        lhand_pose_list, lhand_beta_list, lhand_trans_list, lhand_layer)
    rcf_idx, rcov_idx, rchj_idx, rdist_value = _side_contact(
        rhand_pose_list, rhand_beta_list, rhand_trans_list, rhand_layer)

    is_lhand, is_rhand = get_which_hands_inter(lcf_idx, rcf_idx)
    return (lcf_idx, lcov_idx, lchj_idx, ldist_value,
            rcf_idx, rcov_idx, rchj_idx, rdist_value,
            is_lhand, is_rhand)
