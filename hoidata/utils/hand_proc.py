import numpy as np
import torch

from manopth.manolayer import ManoLayer

# manopth's ManoLayer already returns 21 joints (16 base + 5 fingertips), but
# ordered per-finger-chain as [wrist, thumb(3)+tip, index(3)+tip, middle(3)+tip,
# ring(3)+tip, pinky(3)+tip]. This repo's convention (see joint layout in
# dataloaders/action_dataset.py) wants [wrist, index(3), middle(3), pinky(3),
# ring(3), thumb(3), index_tip, middle_tip, pinky_tip, ring_tip, thumb_tip].
# This permutation maps the former layout to the latter.
_JOINT_REORDER = [0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3, 8, 12, 20, 16, 4]


class ManoOutput:
    """Dict-like MANO forward output: this repo's ``ManoLayer_C`` convention."""

    def __init__(self, vertices, joints_w_tip):
        self.vertices = vertices
        self.joints_w_tip = joints_w_tip


class ManoLayerWrapper(ManoLayer):
    """Wrapper for ManoLayer to support the calling convention of this repo:
    ``hand_layer(pose, betas, trans) -> ManoOutput`` with ``.vertices`` (m) and
    ``.joints_w_tip`` (m, 21 joints incl. fingertips, this repo's joint order).

    ``v_template`` optionally overrides the base MANO template with a subject's
    shaped hand mesh (778, 3) in **metres** — the same units as manopth's own
    ``th_v_template`` (e.g. GRAB's per-subject ``s{N}_{l,r}hand.ply``).  With a
    subject template installed, pose the layer with **zero betas**: the shape is
    already baked into the template, so betas would double-apply it.
    """

    def __init__(self, *args, v_template=None, **kwargs):
        super().__init__(*args, **kwargs)
        if v_template is not None:
            vt = np.asarray(v_template, dtype=np.float32)
            if vt.ndim == 2:
                vt = vt[None]                       # (778,3) -> (1,778,3)
            tmpl = torch.from_numpy(vt).to(self.th_v_template.dtype)
            if tmpl.shape != self.th_v_template.shape:
                raise ValueError(
                    f"v_template shape {tuple(tmpl.shape)} != expected "
                    f"{tuple(self.th_v_template.shape)}")
            self.th_v_template.data.copy_(tmpl)

    def forward(self, th_pose_coeffs, th_betas, th_trans, **kwargs):
        verts, joints = super().forward(
            th_pose_coeffs, th_betas=th_betas, th_trans=th_trans, **kwargs)
        verts /= 1000.0   # mm → m
        joints /= 1000.0  # mm → m
        joints_w_tip = joints[:, _JOINT_REORDER]
        return ManoOutput(vertices=verts, joints_w_tip=joints_w_tip)


def fix_mano_left_shapedirs(lhand_layer, rhand_layer):
    """Fix the sign bug in the official ``MANO_LEFT.pkl`` shape blend shapes.

    The released left-hand model has its first shape direction's x-component
    sign-flipped relative to the right hand, so non-zero betas deform the left
    hand wrongly (https://github.com/vchoutas/smplx/issues/48).  Datasets fit
    with the corrected model (e.g. HOT3D) need the same fix to reproduce their
    hand shapes.  Detection mirrors HOT3D's ``MANOHandModel``: if the left and
    right ``shapedirs[:, 0, :]`` are (almost) identical, negate the left one.
    Idempotent: a second call sees the already-fixed layer and does nothing.

    Irrelevant when posing with zero betas (GRAB / ARCTIC x-data).
    """
    l_dirs = lhand_layer.th_shapedirs
    r_dirs = rhand_layer.th_shapedirs.to(l_dirs.device)
    if torch.sum(torch.abs(l_dirs[:, 0, :] - r_dirs[:, 0, :])) < 1:
        l_dirs[:, 0, :] *= -1
    return lhand_layer


def shaped_mano_layer(layer, beta):
    """Copy of a MANO layer with the shape ``beta`` baked into its template.

    Returns a deep copy of ``layer`` whose ``th_v_template`` is the shaped hand
    ``v_template + shapedirs @ beta``.  Posing the copy with **zero betas** then
    reproduces posing ``layer`` with ``beta`` — so code that hard-codes zero
    betas (``transform_xdata_to_verts`` / ``_joints``) gets the right hand shape,
    the same mechanism as GRAB's per-subject templates.  For the left hand,
    apply ``fix_mano_left_shapedirs`` to ``layer`` first if the betas come from a
    dataset fit with the corrected MANO_LEFT model (e.g. HOT3D).
    """
    import copy
    out = copy.deepcopy(layer)
    dirs = out.th_shapedirs                                     # (778, 3, 10)
    b = torch.as_tensor(np.asarray(beta, dtype=np.float32).reshape(-1)[:dirs.shape[-1]],
                        dtype=dirs.dtype, device=dirs.device)
    out.th_v_template.data.copy_(
        out.th_v_template + torch.einsum("vdk,k->vd", dirs, b)[None])
    return out
