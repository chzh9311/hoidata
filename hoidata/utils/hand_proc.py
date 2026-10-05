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
