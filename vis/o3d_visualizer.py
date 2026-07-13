"""
This file implements a visualizer based on Open3D.
It loads data from a dataloader in dataloaders/ and visualizes the hand-object interaction sequence in 3D.
The visualization window shows the action in loop, and the user can rotate the camera to view the scene from different angles.
The visualizer can also save the visualization as a video file.
"""

import argparse
import os.path as osp
import time

import numpy as np
import trimesh
import open3d as o3d
from omegaconf import OmegaConf

from dataloaders.action_dataset import ObjectModel
from dataloaders.grab_dataset import MotionGRAB
from dataloaders.arctic_dataset import MotionARCTIC
from utils.hand_proc import ManoLayerWrapper
from utils.proc import process_obj_result, proc_torch_cuda, proc_numpy
from utils.proc_arctic import process_object
from utils.object_tensors import ObjectTensors


HAND_SIDE_COLORS = {"left": (0.9, 0.75, 0.65), "right": (0.75, 0.85, 0.95)}
OBJECT_COLOR = (0.6, 0.7, 0.9)


def _pose_arctic_object(cfg, item):
    """Articulated ARCTIC object mesh, posed per frame.

    ``item["obj_traj"]`` is (T,6) trans+rotvec and ``item["obj_angle"]`` is (T,)
    radians; ``process_object`` takes both directly (no rot6d conversion needed,
    unlike the GRAB path).
    """
    object_model = ObjectTensors(cfg.obj_root)
    object_model.to("cuda")

    obj_trans = item["obj_traj"][:, :3]
    obj_rotvec = item["obj_traj"][:, 3:6]
    obj_angle = item["obj_angle"].reshape(-1, 1)
    out = process_object(obj_angle, obj_rotvec, obj_trans, item["obj_name"], object_model)

    v_len = int(out["v_len"][0].item())
    f_len = int(out["f_len"][0].item())
    obj_verts = proc_numpy(out["v"][:, :v_len])
    obj_faces = proc_numpy(out["f"][0, :f_len])
    return obj_verts, obj_faces


def _pose_grab_object(cfg, ds, index, item):
    """Rigid GRAB object mesh, posed per frame.

    Uses the dataset's raw ``x_obj`` (trans+rot6d), not ``item["obj_traj"]``
    (trans+rotvec) — ``process_obj_result`` expects the rot6d representation.
    The rotation-sign convention here is best-effort: GRAB has no preprocessed
    data on disk yet to verify against empirically (see plan notes).
    """
    _, _, _, obj_path = ObjectModel(cfg.data_obj_pc_path)(item["obj_name"])
    mesh = trimesh.load(osp.join(cfg.obj_root, obj_path), process=False)

    x_obj = ds.x_obj[index][: item["nframes"]]
    obj_verts_t = process_obj_result(
        proc_torch_cuda(np.asarray(mesh.vertices, dtype=np.float32)),
        proc_torch_cuda(x_obj),
        "grab",
    )
    return proc_numpy(obj_verts_t), np.asarray(mesh.faces)


def load_sample(cfg, dataset_name: str, split: str = "test", index: int = 0) -> dict:
    """Load one full sequence from the GRAB/ARCTIC dataloader, with hand and
    object meshes already posed per frame.

    Returns a dict:
        hand_verts : (T, 778 or 1556, 3) float32, world-space MANO vertices
        hand_faces : dict {'left': (1538,3), 'right': (1538,3)} int64 — MANO's
                     left/right topologies have opposite face winding (the two
                     hands are mirrored, not just re-translated), so they are
                     kept separate rather than shared
        hand_sides : list[str], 'left'/'right' per active hand, matching the
                     778-vert blocks in hand_verts
        obj_verts  : (T, V, 3) float32, world-space object mesh vertices
        obj_faces  : (F, 3) int, static object mesh topology
        nframes    : int
        obj_name, action_name : str
    """
    # This repo's x-data convention stores a full 48-D axis-angle pose (3 global
    # + 45 per-joint, one 3-vector per finger joint), not manopth's default 6-D
    # PCA coefficient pose — use_pca must be off (ncomps=45) or ManoLayer slices
    # the finger pose down to 6 dims and treats them as PCA coefficients,
    # silently collapsing per-frame finger articulation into a near-constant
    # shape.
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

    dataset_cls = {"grab": MotionGRAB, "arctic": MotionARCTIC}[dataset_name]
    ds = dataset_cls(
        data_path=cfg.data_path,
        data_obj_pc_path=cfg.data_obj_pc_path,
        text_json=cfg.text_json,
        max_nframes=-1,
        task="evaluate",
        split=split,
        lhand_layer=lhand_layer,
        rhand_layer=rhand_layer,
    )
    item = ds[index]

    if dataset_name == "arctic":
        obj_verts, obj_faces = _pose_arctic_object(cfg, item)
    else:
        obj_verts, obj_faces = _pose_grab_object(cfg, ds, index, item)

    # Left/right MANO models are mirrored (not just re-translated): their vertex
    # templates are X-axis reflections of each other, so their face index arrays
    # use opposite winding order to keep normals pointing outward. Reusing one
    # side's faces for the other renders that hand inside-out.
    hand_faces = {
        "left": proc_numpy(lhand_layer.th_faces),
        "right": proc_numpy(rhand_layer.th_faces),
    }

    return {
        "hand_verts": item["hand_verts"],
        "hand_faces": hand_faces,
        "hand_sides": item["hand_sides"],
        "obj_verts": obj_verts,
        "obj_faces": obj_faces,
        "nframes": item["nframes"],
        "obj_name": item["obj_name"],
        "action_name": item["action_name"],
    }


def build_hand_mesh(hand_verts_frame, hand_sides, hand_faces) -> o3d.geometry.TriangleMesh:
    """One frame of hand_verts (778|1556, 3), sliced per active side and summed
    into a single mesh so it stays correct for one or two active hands.

    ``hand_faces`` is a dict {'left': ..., 'right': ...} — each side's own face
    winding must be used, since MANO's left/right topologies are mirrored.
    """
    combined = o3d.geometry.TriangleMesh()
    for i, side in enumerate(hand_sides):
        v = hand_verts_frame[i * 778 : (i + 1) * 778]
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(hand_faces[side])
        )
        mesh.paint_uniform_color(HAND_SIDE_COLORS[side])
        combined += mesh
    combined.compute_vertex_normals()
    return combined


def build_object_mesh(obj_verts_frame, obj_faces) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(obj_verts_frame), o3d.utility.Vector3iVector(obj_faces)
    )
    mesh.paint_uniform_color(OBJECT_COLOR)
    mesh.compute_vertex_normals()
    return mesh


class HOIVisualizer:
    """Loops a hand-object interaction sequence in an Open3D window (free orbit
    camera via mouse drag), and can export the same sequence to a video file."""

    def __init__(self, sample: dict, fps: int = 30):
        self.sample = sample
        self.fps = fps
        self.nframes = sample["nframes"]
        self._frame_idx = 0
        self._last_tick = 0.0

        self.hand_mesh = build_hand_mesh(
            sample["hand_verts"][0], sample["hand_sides"], sample["hand_faces"]
        )
        self.obj_mesh = build_object_mesh(sample["obj_verts"][0], sample["obj_faces"])

    def _set_frame(self, mesh_hand: o3d.geometry.TriangleMesh, mesh_obj: o3d.geometry.TriangleMesh, frame_idx: int):
        hv = self.sample["hand_verts"][frame_idx]
        ov = self.sample["obj_verts"][frame_idx]

        combined_hv = np.concatenate(
            [hv[i * 778 : (i + 1) * 778] for i in range(len(self.sample["hand_sides"]))], axis=0
        )
        mesh_hand.vertices = o3d.utility.Vector3dVector(combined_hv)
        mesh_obj.vertices = o3d.utility.Vector3dVector(ov)
        mesh_hand.compute_vertex_normals()
        mesh_obj.compute_vertex_normals()

    def _advance_frame(self):
        self._frame_idx = (self._frame_idx + 1) % self.nframes
        self._set_frame(self.hand_mesh, self.obj_mesh, self._frame_idx)

    def _animation_callback(self, vis):
        now = time.time()
        if now - self._last_tick < 1.0 / self.fps:
            return False
        self._last_tick = now
        self._advance_frame()
        vis.update_geometry(self.hand_mesh)
        vis.update_geometry(self.obj_mesh)
        return False

    def run_interactive(self):
        """Blocking. Opens a window, loops the sequence, and lets the user
        freely orbit the camera by dragging with the mouse. Requires an active
        display (a monitor with a running X/Wayland session)."""
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"HOI Visualizer — {self.sample['action_name']}")
        vis.add_geometry(self.hand_mesh)
        vis.add_geometry(self.obj_mesh)
        self._last_tick = time.time()
        vis.register_animation_callback(self._animation_callback)
        vis.run()
        vis.destroy_window()

    def render_video(self, out_path: str, fps: int = 30, orbit: bool = True,
                      width: int = 960, height: int = 720):
        """Deterministic, headless-safe export: renders every frame once via
        OffscreenRenderer (independent of wall-clock time and of any live
        interactive window) and writes them to a video file."""
        import imageio

        renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
        renderer.scene.set_background([1, 1, 1, 1])
        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultLit"

        hand_mesh = build_hand_mesh(
            self.sample["hand_verts"][0], self.sample["hand_sides"], self.sample["hand_faces"]
        )
        obj_mesh = build_object_mesh(self.sample["obj_verts"][0], self.sample["obj_faces"])
        renderer.scene.add_geometry("hand", hand_mesh, material)
        renderer.scene.add_geometry("obj", obj_mesh, material)

        # Frame the camera to the full scene (object + hand trajectories across
        # all frames), not just the object's initial pose — hands can start far
        # from the object before a grasp.
        all_pts = np.concatenate(
            [self.sample["obj_verts"].reshape(-1, 3), self.sample["hand_verts"].reshape(-1, 3)],
            axis=0,
        )
        bbox_min, bbox_max = all_pts.min(axis=0), all_pts.max(axis=0)
        center = (bbox_min + bbox_max) / 2.0
        radius = max(0.3, float(np.linalg.norm(bbox_max - bbox_min)))

        writer = imageio.get_writer(out_path, fps=fps)
        try:
            for t in range(self.nframes):
                self._set_frame(hand_mesh, obj_mesh, t)
                renderer.scene.remove_geometry("hand")
                renderer.scene.remove_geometry("obj")
                renderer.scene.add_geometry("hand", hand_mesh, material)
                renderer.scene.add_geometry("obj", obj_mesh, material)

                angle = 2 * np.pi * t / max(self.nframes, 1) if orbit else 0.0
                eye = center + radius * np.array([np.sin(angle), 0.4, np.cos(angle)])
                renderer.scene.camera.look_at(center, eye, [0, 1, 0])

                img = renderer.render_to_image()
                writer.append_data(np.asarray(img))
        finally:
            writer.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize a hand-object interaction sequence.")
    parser.add_argument("--config", required=True, help="Path to a dataset config yaml, e.g. configs/arctic.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--save", default=None, help="Optional output video path, e.g. out.mp4")
    parser.add_argument("--no-interactive", action="store_true", help="Skip the interactive window")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    sample = load_sample(cfg, cfg.name, split=args.split, index=args.index)
    viz = HOIVisualizer(sample, fps=args.fps)

    if args.save:
        viz.render_video(args.save, fps=args.fps)
    if not args.no_interactive:
        viz.run_interactive()


if __name__ == "__main__":
    main()
