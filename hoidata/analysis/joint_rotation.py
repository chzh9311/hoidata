"""Joint-rotation analysis for MANO hand poses.

Step 1 — *normalization*.  A grasp clip gives, per finger joint, a sequence of
rotations ``R_1..R_T`` on SO(3).  To analyze how a joint *moves* (its effective
DOFs) independently of its resting pose, we first factor out the mean pose:

  1. compute the **Karcher / Fréchet mean** ``M`` of the ``R_t`` — the rotation
     minimizing the sum of squared geodesic (angular) distances,
     ``M = argmin_R  sum_t  d(R, R_t)^2``  with  ``d(A,B) = ||log(A^T B)||`` ;
  2. express each frame **relative to that mean**, ``Rel_t = M^T R_t`` — a
     rotation that is identity at the mean pose and captures only the deviation.

The relative rotations ``Rel_t`` are the normalized signal the later DOF steps
(axis fitting, range of motion, PCA over the log map) operate on.

Rotations are represented as 3x3 matrices throughout; helpers accept the MANO
6-D rotation encoding (``rotation_6d_to_matrix``) and reshape as needed.  The
geodesic mean and log/exp use ``pytorch3d.transforms`` SO(3) maps.

Run in the ``hoi_common`` conda env (CUDA 12.4).
"""

from __future__ import annotations

import torch
from pytorch3d.transforms import (
    rotation_6d_to_matrix, so3_exp_map, so3_log_map)


def karcher_mean(
    R: torch.Tensor,
    weights: torch.Tensor | None = None,
    max_iter: int = 100,
    tol: float = 1e-7,
) -> torch.Tensor:
    """Karcher (Fréchet / geodesic) mean of rotations on SO(3).

    Averages over the **second-to-last batched axis**: an input of shape
    ``(*B, N, 3, 3)`` returns ``(*B, 3, 3)`` — the mean of the ``N`` rotations for
    each leading batch element ``B`` (so one call can mean every joint at once,
    e.g. ``(J, T, 3, 3) -> (J, 3, 3)``).

    Iterates the Riemannian gradient descent
        ``M <- M · exp( mean_n log(M^T R_n) )``
    which converges to the unique minimizer of the sum of squared geodesic
    distances whenever the rotations lie within a geodesic ball of radius < pi/2
    (always true for a single joint's motion in a grasp).  Stops when the mean
    tangent step's norm falls below ``tol`` or ``max_iter`` is reached.

    Args:
        R:       ``(*B, N, 3, 3)`` rotation matrices to average over axis ``-3``.
        weights: optional ``(*B, N)`` non-negative weights (renormalized per
                 batch element); defaults to a uniform mean.
        max_iter, tol: iteration budget and tangent-step convergence threshold.

    Returns:
        ``(*B, 3, 3)`` mean rotation matrices.
    """
    if R.shape[-2:] != (3, 3) or R.dim() < 3:
        raise ValueError(f"expected (*B, N, 3, 3), got {tuple(R.shape)}")
    *B, N, _, _ = R.shape

    # Flatten the leading batch so the SO(3) maps see a plain (M, 3, 3) stack;
    # reshape results back to (*B, ...) on the way out.
    Rf = R.reshape(-1, N, 3, 3)                                  # (P, N, 3, 3)
    P = Rf.shape[0]

    if weights is not None:
        w = weights.reshape(P, N).to(Rf.dtype)
        w = w / w.sum(dim=1, keepdim=True).clamp_min(torch.finfo(Rf.dtype).eps)
    else:
        w = None

    M = Rf[:, 0].clone()                                         # (P, 3, 3) seed
    for _ in range(max_iter):
        # Tangent vectors of each R_n at the current mean M: log(M^T R_n).
        rel = torch.matmul(M.transpose(1, 2).unsqueeze(1), Rf)  # (P, N, 3, 3)
        tangent = so3_log_map(rel.reshape(-1, 3, 3)).reshape(P, N, 3)
        if w is None:
            step = tangent.mean(dim=1)                           # (P, 3)
        else:
            step = (tangent * w.unsqueeze(-1)).sum(dim=1)        # (P, 3)
        M = torch.matmul(M, so3_exp_map(step))                   # retract onto SO(3)
        if step.norm(dim=-1).max() < tol:
            break

    return M.reshape(*B, 3, 3)


def _to_joint_matrices(hand_pose: torch.Tensor) -> torch.Tensor:
    """MANO joint rotations → ``(T, J, 3, 3)`` matrices.

    Accepts the finger-joint rot6d block in any of these forms:
      - ``(T, J, 6)``  already split per joint,
      - ``(T, J*6)``   flat rot6d (e.g. ``x_hand[:, 9:]`` = 15 joints, 90 dims),
      - ``(T, J, 3, 3)`` already matrices (returned unchanged).
    """
    if hand_pose.dim() == 4 and hand_pose.shape[-2:] == (3, 3):
        return hand_pose
    if hand_pose.dim() == 2:                       # (T, J*6) → (T, J, 6)
        T, D = hand_pose.shape
        if D % 6 != 0:
            raise ValueError(f"flat rot6d dim {D} is not a multiple of 6")
        hand_pose = hand_pose.reshape(T, D // 6, 6)
    if hand_pose.dim() != 3 or hand_pose.shape[-1] != 6:
        raise ValueError(
            "expected (T, J, 6), (T, J*6) or (T, J, 3, 3); "
            f"got {tuple(hand_pose.shape)}")
    T, J, _ = hand_pose.shape
    return rotation_6d_to_matrix(hand_pose.reshape(T * J, 6)).reshape(T, J, 3, 3)


def normalize_joint_rotations(
    hand_pose: torch.Tensor,
    max_iter: int = 100,
    tol: float = 1e-7,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean-normalize a clip's per-joint rotation sequences.

    For each finger joint independently, computes the Karcher mean over the ``T``
    frames and the per-frame relative rotation ``Rel_t = M_j^T R_{t,j}``.

    Args:
        hand_pose: ``(T, J, 6)`` / ``(T, J*6)`` rot6d, or ``(T, J, 3, 3)`` — the
                   finger-joint rotations of one clip (e.g. GRAB
                   ``x_hand[:, 9:]`` = 15 joints).
        max_iter, tol: forwarded to :func:`karcher_mean`.

    Returns:
        ``(mean_R, rel_R)`` where
          - ``mean_R`` is ``(J, 3, 3)`` — each joint's Fréchet-mean rotation,
          - ``rel_R``  is ``(T, J, 3, 3)`` — the mean-relative rotations, with
            ``rel_R[t, j]`` identity when frame ``t`` sits at joint ``j``'s mean.
    """
    R = _to_joint_matrices(hand_pose)                           # (T, J, 3, 3)
    T, J = R.shape[:2]

    # Mean over frames per joint: reorder to (J, T, 3, 3) and average axis -3.
    mean_R = karcher_mean(R.permute(1, 0, 2, 3), max_iter=max_iter, tol=tol)  # (J,3,3)

    # Relative rotations Rel_{t,j} = M_j^T R_{t,j}, broadcast the mean over frames.
    rel_R = torch.matmul(mean_R.transpose(1, 2).unsqueeze(0), R)  # (T, J, 3, 3)
    return mean_R, rel_R


def geodesic_angle(R: torch.Tensor) -> torch.Tensor:
    """Rotation angle (radians) of each rotation matrix — ``||log(R)||``.

    ``(*B, 3, 3) -> (*B,)``.  Handy for summarizing relative rotations (e.g. the
    range of motion is ``geodesic_angle(rel_R).max()`` per joint).
    """
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"expected (*B, 3, 3), got {tuple(R.shape)}")
    *B, _, _ = R.shape
    return so3_log_map(R.reshape(-1, 3, 3)).norm(dim=-1).reshape(*B)
