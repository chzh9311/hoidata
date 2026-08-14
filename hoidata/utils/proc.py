"""Processing utilities shared across preprocessing and training.

Source: Text2HOI (https://github.com/JunukCha/Text2HOI.git)
  lib/utils/proc.py
"""

import numpy as np
import torch
from pytorch3d.ops.knn import knn_points

from pytorch3d.transforms import (
    axis_angle_to_matrix,
    axis_angle_to_quaternion,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
    quaternion_apply,
    rotation_6d_to_matrix,
)


def xdata_delta_torch(xdata):
    """Differentiable per-frame delta of x-data (trans + rot6d), batched.

    Mirrors common/data/action_dataset.compute_xdata_delta (world-frame rotation
    delta R_delta = R_{t+1} @ R_t^T) but in torch so it can flow gradients — used
    e.g. for delta-consistency losses on predicted motion.

    Args:
        xdata: (..., T, 3 + n_rot*6) tensor.  Leading dims are treated as batch.
    Returns:
        (..., T-1, 3 + n_rot*6) tensor of deltas.
    """
    *lead, T, D = xdata.shape
    trans = xdata[..., :3]
    rot6d = xdata[..., 3:]
    n_rot = rot6d.shape[-1] // 6

    trans_delta = trans[..., 1:, :] - trans[..., :-1, :]            # (..., T-1, 3)

    rotmat = rotation_6d_to_matrix(rot6d.reshape(*lead, T, n_rot, 6))
    R_cur = rotmat[..., :-1, :, :, :]                              # R_t
    R_nxt = rotmat[..., 1:, :, :, :]                               # R_{t+1}
    R_delta = R_nxt @ R_cur.transpose(-1, -2)                      # R_{t+1} @ R_t^T
    rot6d_delta = matrix_to_rotation_6d(R_delta).reshape(*lead, T - 1, n_rot * 6)

    return torch.cat([trans_delta, rot6d_delta], dim=-1)           # (..., T-1, D)


# ---------------------------------------------------------------------------
# Tensor / numpy helpers
# ---------------------------------------------------------------------------

def proc_torch_cuda(d):
    if not isinstance(d, torch.Tensor):
        d = torch.FloatTensor(d)
    if not d.is_cuda:
        d = d.cuda()
    return d


def proc_long_torch_cuda(d):
    if not isinstance(d, torch.Tensor):
        d = torch.LongTensor(d)
    if not d.is_cuda:
        d = d.cuda()
    return d


def proc_numpy(d):
    if isinstance(d, torch.Tensor):
        if d.requires_grad:
            d = d.detach()
        if d.is_cuda:
            d = d.cpu()
        d = d.numpy()
    return d


def proc_torch_frame(l):
    if isinstance(l, (list, np.ndarray)):
        l = [torch.FloatTensor(_l).unsqueeze(0) for _l in l]
        l = torch.cat(l).cuda()
    return l


# ---------------------------------------------------------------------------
# Hand / object transform helpers
# ---------------------------------------------------------------------------

def transform_hand_to_xdata(trans, pose):
    """Convert MANO trans + axis-angle pose to x-data representation (trans + rot6d).

    rot6d follows pytorch3d's convention (``matrix_to_rotation_6d``): the first
    two *rows* of R, flattened as ``[R00,R01,R02,R10,R11,R12]``.  This is the
    on-disk format of ``x_lhand`` / ``x_rhand`` in ``data.npz``; caches written
    before the pytorch3d migration used a different (column-major) layout and
    must be converted with ``scripts/migrate_rot6d.py``.
    """
    nframes = pose.shape[0]
    rot6d = matrix_to_rotation_6d(
        axis_angle_to_matrix(pose.reshape(-1, 3))).reshape(nframes, 16 * 6)
    xdata = torch.cat([trans, rot6d], dim=1)
    return proc_numpy(xdata)


def transform_xdata_to_joints(xdata, hand_layer):
    """Convert x-data (trans + rot6d) to 21 world-space joints (with fingertips)."""
    xdata = proc_torch_cuda(xdata).unsqueeze(0)
    hand_joints = _get_hand_joints_w_tip(xdata, hand_layer)
    return proc_numpy(hand_joints.squeeze(0))


def transform_xdata_to_verts(xdata, hand_layer):
    """Convert x-data (trans + rot6d) to (T, 778, 3) world-space MANO vertices."""
    xdata = proc_torch_cuda(xdata)
    nframes = xdata.shape[0]
    pred_trans = xdata[:, :3]
    pred_rot6d = xdata[:, 3:]
    pred_pose = matrix_to_axis_angle(
        rotation_6d_to_matrix(pred_rot6d.reshape(-1, 6))).reshape(nframes, 48)
    betas = torch.zeros(nframes, 10, device=xdata.device)
    out = hand_layer(pred_pose, betas, pred_trans)
    return proc_numpy(out.vertices)


def transform_obj_to_xdata(obj_extmat):
    """Convert (T, 3, 4) object extrinsic matrices to (T, 9) x-data (trans + rot6d).

    rot6d follows pytorch3d's convention — see ``transform_hand_to_xdata``.
    """
    orl = obj_extmat  # already a cuda tensor from preprocessing
    obj_rotmat = orl[:, :3, :3]
    obj_trans = orl[:, :3, 3]
    nframes = obj_rotmat.shape[0]
    rot6d = matrix_to_rotation_6d(obj_rotmat).reshape(nframes, 6)
    xdata = torch.cat([obj_trans, rot6d], dim=1)
    return proc_numpy(xdata)


def get_hand_org(hand_pose, hand_beta, hand_trans, hand_layer):
    """Return the wrist (root) joint position for each frame."""
    out = hand_layer(hand_pose, hand_beta, hand_trans)
    return out.joints_w_tip[:, 0]


# ---------------------------------------------------------------------------
# Nearest-neighbour / interior helpers (require pytorch3d)
# ---------------------------------------------------------------------------

def _get_NN(src_xyz, trg_xyz, k=1):
    src_lengths = torch.full((src_xyz.shape[0],), src_xyz.shape[1],
                             dtype=torch.int64, device=src_xyz.device)
    trg_lengths = torch.full((trg_xyz.shape[0],), trg_xyz.shape[1],
                             dtype=torch.int64, device=trg_xyz.device)
    src_nn = knn_points(src_xyz, trg_xyz, lengths1=src_lengths,
                        lengths2=trg_lengths, K=k)
    return src_nn.dists[..., 0], src_nn.idx[..., 0]


def _batched_index_select(input, index, dim=1):
    views = [input.shape[0]] + [1 if i != dim else -1
                                  for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim, index)


def _get_interior(src_face_normal, src_xyz, trg_xyz, trg_NN_idx):
    NN_src_xyz = _batched_index_select(src_xyz, trg_NN_idx)
    NN_vector = NN_src_xyz - trg_xyz
    NN_src_normal = _batched_index_select(src_face_normal, trg_NN_idx)
    return (NN_vector * NN_src_normal).sum(dim=-1) > 0


def _get_hand_joints_w_tip(hand_params, hand_layer):
    """(B, T, 99) x-data → (B, T, 21, 3) world-space joints with fingertips."""
    bs, nframes = hand_params.shape[:2]
    pred_trans = hand_params[..., :3]
    pred_rot6d = hand_params[..., 3:]

    pred_pose = matrix_to_axis_angle(
        rotation_6d_to_matrix(pred_rot6d.reshape(-1, 6))).reshape(-1, 48)
    pred_trans_flat = pred_trans.reshape(-1, 3)
    betas = torch.zeros(bs * nframes, 10, device=hand_params.device)
    out = hand_layer(pred_pose, betas, pred_trans_flat)
    joints = out.joints_w_tip.reshape(bs, nframes, 21, 3)
    return joints


# ---------------------------------------------------------------------------
# Contact info
# ---------------------------------------------------------------------------

def get_contact_info(
    lhand_pose_list, lhand_beta_list, lhand_trans_list,
    rhand_pose_list, rhand_beta_list, rhand_trans_list,
    object_rotmat_list, lhand_layer, rhand_layer,
    sampled_obj_verts_org, mul_rv=True,
):
    """Compute per-frame hand-object contact indices and distances.

    Source: Text2HOI (https://github.com/JunukCha/Text2HOI.git)
      lib/utils/proc.py — get_contact_info
    """
    contact_threshold = 0.02
    sampled_obj_verts_org = proc_torch_cuda(sampled_obj_verts_org)
    orl = object_rotmat_list
    obj_rotmat = orl[:, :3, :3]
    obj_trans = orl[:, :3, 3]

    if mul_rv:
        sampled_obj_verts = (torch.einsum("tij,kj->tki", obj_rotmat, sampled_obj_verts_org)
                             + obj_trans.unsqueeze(1))
    else:
        sampled_obj_verts = (torch.einsum("tij,ki->tkj", obj_rotmat, sampled_obj_verts_org)
                             + obj_trans.unsqueeze(1))

    def _hand_contact(pose_list, beta_list, trans_list, hand_layer):
        if len(pose_list) == 0:
            empty = np.array([])
            return empty, empty, empty, empty
        pl = proc_torch_frame(pose_list)
        bl = proc_torch_frame(beta_list)
        tl = proc_torch_frame(trans_list)
        out = hand_layer(pl, bl, tl)
        joints = out.joints_w_tip
        dist = torch.cdist(sampled_obj_verts, joints)
        cf_idx, cov_idx, chj_idx = torch.where(dist < contact_threshold)
        dist_value = dist[cf_idx, cov_idx, chj_idx]
        return (proc_numpy(cf_idx), proc_numpy(cov_idx),
                proc_numpy(chj_idx), proc_numpy(dist_value))

    lcf_idx, lcov_idx, lchj_idx, ldist_value = _hand_contact(
        lhand_pose_list, lhand_beta_list, lhand_trans_list, lhand_layer)
    rcf_idx, rcov_idx, rchj_idx, rdist_value = _hand_contact(
        rhand_pose_list, rhand_beta_list, rhand_trans_list, rhand_layer)

    is_lhand = int(len(lcf_idx) > 0)
    is_rhand = int(len(rcf_idx) > 0)

    return (lcf_idx, lcov_idx, lchj_idx, ldist_value,
            rcf_idx, rcov_idx, rchj_idx, rdist_value,
            is_lhand, is_rhand)


def get_hand_object_dist(hand_joints, sampled_obj_verts):
    """Pairwise distance: (T, 21, 3) × (T, K, 3) → (T, K, 21)."""
    hand_joints = proc_torch_cuda(hand_joints)
    sampled_obj_verts = proc_torch_cuda(sampled_obj_verts)
    return torch.cdist(sampled_obj_verts, hand_joints)


def get_contact_idx(dist, contact_threshold):
    """Return (frame, obj_vert, hand_joint) indices where dist < threshold."""
    cf_idx, cov_idx, chj_idx = torch.where(dist < contact_threshold)
    return cf_idx, cov_idx, chj_idx


def get_which_hands_inter(lcf_idx, rcf_idx):
    """Return (is_lhand, is_rhand) as 0/1 ints from contact index arrays."""
    is_lhand = int(len(lcf_idx) > 0)
    is_rhand = int(len(rcf_idx) > 0)
    return is_lhand, is_rhand


# ---------------------------------------------------------------------------
# Point cloud helpers
# ---------------------------------------------------------------------------

def farthest_point_sample(xyz, npoint):
    """Farthest point sampling on a point cloud.

    Args:
        xyz: (B, N, 3) cuda tensor
        npoint: number of points to sample
    Returns:
        centroids: (B, npoint) indices
    """
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.zeros(B, dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]
    return centroids


def get_contact_map(idx, v_num, is_hand):
    contact_map = np.zeros(v_num)
    if is_hand:
        contact_map[idx] = 1
    return contact_map


def pc_normalize(pc, return_params=False):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    scale = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / scale
    if return_params:
        return pc, centroid, scale
    else:
        return pc


def process_dist_map(
    max_nframes, init_frame,
    cf_idx, cov_idx, chj_idx,
    dist_value, is_hand
):
    dist_map = np.zeros((max_nframes, 1024, 21), dtype=np.float32)
    if is_hand:
        f_idx_filtered = np.where((init_frame <= cf_idx) & (cf_idx < init_frame + max_nframes))[0]
        cf_idx_selected = cf_idx[f_idx_filtered]
        cf_idx_moved = cf_idx_selected - init_frame
        cov_idx_selected = cov_idx[f_idx_filtered]
        chj_idx_selected = chj_idx[f_idx_filtered]
        dist_value_selected = dist_value[f_idx_filtered]
        dist_map[cf_idx_moved, cov_idx_selected, chj_idx_selected] = dist_value_selected
    return dist_map


# ---------------------------------------------------------------------------
# Object point-cloud transforms (batched & single-sequence)
#
# Source: Text2HOI (https://github.com/JunukCha/Text2HOI.git)
#   lib/utils/proc_output.py (get_transformed_obj_pc) and
#   lib/utils/data.py (process_obj_result).
# GRAB applies R^T via einsum("...ij,...ki->...kj", R, v); the arctic branch
# (articulated objects) is kept verbatim for completeness but is never used by
# the GRAB-only pipeline.
# ---------------------------------------------------------------------------

def get_transformed_obj_pc(obj_params, obj_pc, dataset_name, obj_top_idx=None):
    """Batched: (B, T, 9) obj params + (B, N, 3) point cloud → (B, T, N, 3)."""
    bs, nframes = obj_params.shape[:2]
    obj_trans = obj_params[..., :3]
    obj_rot6d = obj_params[..., 3:9]
    obj_rotmat = rotation_6d_to_matrix(obj_rot6d).reshape(bs, nframes, 3, 3)
    if dataset_name == "arctic" and obj_params.shape[-1] == 10 and obj_top_idx is not None:
        obj_top_idx = obj_top_idx.bool()
        obj_angle = obj_params[..., 9:10]
        quat_arti = axis_angle_to_quaternion(torch.FloatTensor([0, 0, -1]).to(obj_params.device).view(1, 1, 3)*obj_angle)
        obj_pc = obj_pc.unsqueeze(1).expand(-1, nframes, -1, -1)
        obj_pc2 = obj_pc.clone()
        obj_top_idx = obj_top_idx.unsqueeze(1).expand(-1, nframes, -1)
        obj_pc2[obj_top_idx] = quaternion_apply(quat_arti[:, :, None], obj_pc)[obj_top_idx]
        obj_pc_rotated = torch.einsum("btij,btkj->btki", obj_rotmat, obj_pc2)
    elif dataset_name == "h2o":
        obj_pc_rotated = torch.einsum("btij,bkj->btki", obj_rotmat, obj_pc)
    elif dataset_name == "grab":
        obj_pc_rotated = torch.einsum("btij,bki->btkj", obj_rotmat, obj_pc)
    else:
        obj_pc_rotated = torch.einsum("btij,bkj->btki", obj_rotmat, obj_pc)
    obj_pc_transformed = obj_pc_rotated + obj_trans.unsqueeze(2)
    return obj_pc_transformed


def process_obj_result(obj_verts, obj_params, dataset_name, obj_top_idx=None):
    """Single-sequence: (V, 3) obj mesh verts + (T, 9) params → (T, V, 3)."""
    nframes = obj_params.shape[0]
    obj_trans = obj_params[:, :3]
    obj_rot6d = obj_params[:, 3:9]
    obj_rotmat = rotation_6d_to_matrix(obj_rot6d).reshape(-1, 3, 3)
    if dataset_name == "arctic" and obj_params.shape[-1] == 10 and obj_top_idx is not None:
        obj_top_idx = obj_top_idx.bool()
        obj_angle = obj_params[..., 9:10]
        quat_arti = axis_angle_to_quaternion(torch.FloatTensor([0, 0, -1]).to(obj_params.device).view(1, 3)*obj_angle)
        obj_verts = obj_verts.unsqueeze(0).expand(nframes, -1, -1)
        obj_verts2 = obj_verts.clone()
        obj_top_idx = obj_top_idx.unsqueeze(0).expand(nframes, -1)
        obj_verts2[obj_top_idx] = quaternion_apply(quat_arti[:, None], obj_verts)[obj_top_idx]
        obj_pc_rotated = torch.einsum("tij,tkj->tki", obj_rotmat, obj_verts2)
    elif dataset_name == "h2o":
        obj_pc_rotated = torch.einsum("tij,kj->tki", obj_rotmat, obj_verts)
    elif dataset_name == "grab":
        obj_pc_rotated = torch.einsum("tij,ki->tkj", obj_rotmat, obj_verts)
    else:
        obj_pc_rotated = torch.einsum("tij,kj->tki", obj_rotmat, obj_verts)
    obj_verts_transformed = obj_pc_rotated + obj_trans.unsqueeze(1)
    return obj_verts_transformed


def get_hand_obj_dist_map(
    pred_lhand, pred_rhand, pred_obj, obj_pc,
    lhand_layer, rhand_layer, dataset_name, obj_pc_top_idx=None
):
    """Per-frame hand-joint ↔ object-point distance maps (training loss helper)."""
    bs, nframes = pred_obj.shape[:2]
    npts = obj_pc.shape[1]
    lhand_joints_w_tip = _get_hand_joints_w_tip(pred_lhand, lhand_layer)
    rhand_joints_w_tip = _get_hand_joints_w_tip(pred_rhand, rhand_layer)
    obj_pc_transformed = get_transformed_obj_pc(pred_obj, obj_pc, dataset_name, obj_pc_top_idx)

    pred_ldist_map = torch.cdist(
        obj_pc_transformed.reshape(-1, npts, 3),
        lhand_joints_w_tip.reshape(-1, 21, 3)
    ).reshape(bs, nframes, npts, 21)
    pred_rdist_map = torch.cdist(
        obj_pc_transformed.reshape(-1, npts, 3),
        rhand_joints_w_tip.reshape(-1, 21, 3)
    ).reshape(bs, nframes, npts, 21)
    return pred_ldist_map, pred_rdist_map


# ---------------------------------------------------------------------------
# Object feature assembly for the diffusion conditioning
#
# Source: Text2HOI lib/utils/proc.py
# ---------------------------------------------------------------------------

def proc_cond_contact_estimator(obj_scale, obj_feat, enc_text, npts, use_scale):
    if use_scale:
        enc_text_expand = enc_text.unsqueeze(1)
        enc_text_expand = enc_text_expand.expand(-1, npts, -1)
        obj_scale_expand2 = obj_scale.unsqueeze(1).unsqueeze(2)
        obj_scale_expand2 = obj_scale_expand2.expand(-1, npts, -1)
        condition = torch.cat([obj_scale_expand2, obj_feat, enc_text_expand], dim=2)
    else:
        enc_text_expand = enc_text.unsqueeze(1).expand(-1, npts, -1)
        condition = torch.cat([obj_feat, enc_text_expand], dim=2)
    return condition


def proc_obj_feat_cov(
    contact_estimator,
    obj_scale, obj_feat,
    enc_text, npts,
    use_scale, use_contact_feat,
):
    obj_feat_global = obj_feat[:, 0, :1024]
    if use_contact_feat:
        condition = proc_cond_contact_estimator(obj_scale, obj_feat, enc_text, npts, use_scale)
        est_contact_map = contact_estimator.decode(condition)
        est_contact_map_plot = est_contact_map.squeeze(-1).clone()
        est_contact_map = (est_contact_map.squeeze(-1) > 0.5).long()
        obj_feat_cov = torch.cat([obj_feat_global, est_contact_map], dim=1)
    else:
        obj_feat_cov = obj_feat_global
        est_contact_map = None
        est_contact_map_plot = None
    return obj_feat_cov, est_contact_map, est_contact_map_plot


def proc_obj_feat_final(
    contact_estimator,
    obj_scale, obj_cent,
    obj_feat, enc_text, npts,
    use_obj_scale_centroid,
    use_scale, use_contact_feat,
    return_plot=False,
):
    obj_feat_cov, est_contact_map, est_contact_map_plot = proc_obj_feat_cov(
        contact_estimator,
        obj_scale, obj_feat, enc_text, npts,
        use_scale, use_contact_feat,
    )
    if use_obj_scale_centroid:
        obj_scale_expand1 = obj_scale.unsqueeze(1)
        obj_feat_final = torch.cat([obj_feat_cov, obj_scale_expand1, obj_cent], dim=1)
    else:
        obj_feat_final = obj_feat_cov
    if return_plot:
        return obj_feat_final, est_contact_map, est_contact_map_plot
    else:
        return obj_feat_final, est_contact_map


# ---------------------------------------------------------------------------
# Refiner input assembly
#
# Source: Text2HOI lib/utils/proc.py (get_hand2obj_dist, proc_refiner_input).
# Hand joints come from the target ManoLayer_C helper _get_hand_joints_w_tip
# (translation already applied), matching the source semantics.
# ---------------------------------------------------------------------------

def get_hand2obj_dist(hand_joints, obj_pc, obj_pc_normal):
    B, T = hand_joints.shape[:2]
    hand_joints = hand_joints.reshape(B*T, -1, 3)
    obj_pc = obj_pc.reshape(B*T, -1, 3)
    obj_pc_normal = obj_pc_normal.reshape(B*T, -1, 3)
    hand_nn_dist, hand_nn_idx = _get_NN(hand_joints, obj_pc)
    _ = _get_interior(obj_pc_normal, obj_pc, hand_joints, hand_nn_idx)
    hand_nn_dist = hand_nn_dist.sqrt()
    hand_nn_dist = torch.abs(hand_nn_dist)
    hand_nn_idx_expand = hand_nn_idx.unsqueeze(-1).expand(*hand_nn_idx.shape, 3)
    obj_pc_contact = torch.gather(obj_pc, 1, hand_nn_idx_expand)
    hand_dist_values_xyz = (hand_joints-obj_pc_contact)**2
    hand_dist_values_xyz = hand_dist_values_xyz.reshape(B, T, -1, 3)
    hand_nn_dist = hand_nn_dist.reshape(B, T, -1)
    obj_pc_contact = obj_pc_contact.reshape(B, T, -1, 3)
    return hand_dist_values_xyz, hand_nn_dist, obj_pc_contact


def proc_refiner_input(
    pred_lhand, pred_rhand, pred_obj,
    lhand_layer, rhand_layer, obj_pc_org, obj_pc_normal_org,
    valid_mask_lhand, valid_mask_rhand, valid_mask_obj,
    cov_map, dataset_name, return_psuedo_gt=False, obj_pc_top_idx=None,
    is_lhand=True, is_rhand=True,
):
    threshold = 0.02
    bs, T = pred_obj.shape[:2]
    njoints = 21
    tf_obj_pc = get_transformed_obj_pc(pred_obj, obj_pc_org, dataset_name, obj_pc_top_idx)
    tf_obj_pc_normal = get_transformed_obj_pc(pred_obj, obj_pc_normal_org, dataset_name, obj_pc_top_idx)
    npts = tf_obj_pc.shape[2]

    if is_lhand:
        lhand_joints = _get_hand_joints_w_tip(pred_lhand, lhand_layer)
        lhand_dist_values_xyz, lhand_dist_values, obj_pc_contact_lhand \
            = get_hand2obj_dist(lhand_joints, tf_obj_pc, tf_obj_pc_normal)
        lhand_attn = torch.exp(-50*lhand_dist_values_xyz)
    else:
        lhand_joints = torch.zeros(bs, T, njoints, 3, device=pred_lhand.device)
        lhand_dist_values_xyz = torch.zeros(bs, T, njoints, 3, device=pred_lhand.device)
        lhand_dist_values = torch.zeros(bs, T, njoints, device=pred_lhand.device)
        obj_pc_contact_lhand = torch.zeros(bs, T, npts, 3, device=pred_lhand.device)
        lhand_attn = torch.zeros(bs, T, njoints, 3, device=pred_lhand.device)

    if is_rhand:
        rhand_joints = _get_hand_joints_w_tip(pred_rhand, rhand_layer)
        rhand_dist_values_xyz, rhand_dist_values, obj_pc_contact_rhand \
            = get_hand2obj_dist(rhand_joints, tf_obj_pc, tf_obj_pc_normal)
        rhand_attn = torch.exp(-50*rhand_dist_values_xyz)
    else:
        rhand_joints = torch.zeros(bs, T, njoints, 3, device=pred_rhand.device)
        rhand_dist_values_xyz = torch.zeros(bs, T, njoints, 3, device=pred_rhand.device)
        rhand_dist_values = torch.zeros(bs, T, njoints, device=pred_rhand.device)
        obj_pc_contact_rhand = torch.zeros(bs, T, npts, 3, device=pred_rhand.device)
        rhand_attn = torch.zeros(bs, T, njoints, 3, device=pred_rhand.device)

    cov_map = cov_map.unsqueeze(1)
    cov_map = cov_map.expand(-1, T, -1)
    input_lhand = torch.cat([pred_lhand, lhand_joints.reshape(bs, T, -1), lhand_attn.reshape(bs, T, -1), tf_obj_pc.norm(dim=-1), cov_map], dim=2)
    input_rhand = torch.cat([pred_rhand, rhand_joints.reshape(bs, T, -1), rhand_attn.reshape(bs, T, -1), tf_obj_pc.norm(dim=-1), cov_map], dim=2)
    if return_psuedo_gt:
        lhand_contact_joint_mask = (lhand_dist_values < threshold)*(torch.logical_and(valid_mask_obj, valid_mask_lhand)).unsqueeze(2)
        rhand_contact_joint_mask = (rhand_dist_values < threshold)*(torch.logical_and(valid_mask_obj, valid_mask_rhand)).unsqueeze(2)
        obj_pc_contact_lhand_psuedo = obj_pc_contact_lhand.clone()
        obj_pc_contact_rhand_psuedo = obj_pc_contact_rhand.clone()
        return (
            input_lhand, input_rhand, pred_obj,
            obj_pc_contact_lhand_psuedo, obj_pc_contact_rhand_psuedo,
            lhand_contact_joint_mask, rhand_contact_joint_mask,
        )
    else:
        return input_lhand, input_rhand, pred_obj


def process_text(
    action_name, obj_name,
    is_lhand, is_rhand,
    text_descriptions, return_key=False,
):
    if is_lhand and is_rhand:
        text = f"{action_name} {obj_name} with both hands."
    elif is_lhand:
        text = f"{action_name} {obj_name} with left hand."
    elif is_rhand:
        text = f"{action_name} {obj_name} with right hand."
    text_key = text.capitalize()
    if return_key:
        return text_key
    else:
        text_description = text_descriptions[text_key]
        text = np.random.choice(text_description)
        return text


def thing2dev(thing, dev):
    if hasattr(thing, "to"):
        thing = thing.to(dev)
        return thing
    if isinstance(thing, list):
        return [thing2dev(ten, dev) for ten in thing]
    if isinstance(thing, tuple):
        return tuple(thing2dev(list(thing), dev))
    if isinstance(thing, dict):
        return {k: thing2dev(v, dev) for k, v in thing.items()}
    if isinstance(thing, torch.Tensor):
        return thing.to(dev)
    return thing


def pad_tensor_list(v_list: list):
    dev = v_list[0].device
    num_meshes = len(v_list)
    num_dim = 1 if len(v_list[0].shape) == 1 else v_list[0].shape[1]
    v_len_list = []
    for verts in v_list:
        v_len_list.append(verts.shape[0])

    pad_len = max(v_len_list)
    dtype = v_list[0].dtype
    if num_dim == 1:
        padded_tensor = torch.zeros(num_meshes, pad_len, dtype=dtype)
    else:
        padded_tensor = torch.zeros(num_meshes, pad_len, num_dim, dtype=dtype)
    for idx, (verts, v_len) in enumerate(zip(v_list, v_len_list)):
        padded_tensor[idx, :v_len] = verts
    padded_tensor = padded_tensor.to(dev)
    v_len_list = torch.LongTensor(v_len_list).to(dev)
    return padded_tensor, v_len_list