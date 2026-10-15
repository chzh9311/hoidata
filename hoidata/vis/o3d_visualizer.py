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

from hoidata.dataloaders.action_dataset import (
    ObjectModel, segment_grasp_timeline, TA_HALF, MIN_NULL)
from hoidata.dataloaders.grab_dataset import (
    MotionGRAB, TransitionMotionGRAB, BimanualTransitionMotionGRAB,
    StableGraspGRAB, BimanualStableGraspGRAB,
    load_grab_sbj_info, SubjectManoLayers)
from hoidata.dataloaders.arctic_dataset import MotionARCTIC, StableGraspARCTIC
from hoidata.utils.hand_proc import ManoLayerWrapper
from hoidata.utils.proc import process_obj_result, proc_torch_cuda, proc_numpy
from hoidata.utils.proc_arctic import process_object
from hoidata.utils.object_tensors import ObjectTensors


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


def build_dataset(cfg, dataset_name: str, split: str = "test"):
    """Build the MANO hand layers and the full-sequence (``task="evaluate"``)
    dataloader once, so callers can pose individual sequences on demand without
    re-instantiating everything per index (see ``pose_sample``).

    Returns ``(ds, lhand_layer, rhand_layer)``.
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

    # GRAB ships a per-subject shaped MANO hand template (tools/subject_meshes);
    # install per-subject layers so the visualized hands match each subject's
    # actual hand shape instead of the shared mean template.  ``_getitem_full``
    # then picks the subject layer via ``_get_layers`` and poses with zero betas
    # (the shape is baked into the template).
    if dataset_name == "grab":
        import os.path as _osp
        subj_dir = cfg.get(
            "subject_meshes_root",
            _osp.join(_osp.dirname(cfg.root.rstrip("/")), "tools", "subject_meshes"),
        )
        sbj_info = load_grab_sbj_info(subj_dir)
        if sbj_info:
            ds.subject_layers = SubjectManoLayers(
                sbj_info, mano_root=cfg.mano_root, flat_hand=cfg.flat_hand,
                use_pca=use_pca,
                default_layers={"left": lhand_layer, "right": rhand_layer},
            )
            print(f"[grab] using per-subject hand templates for "
                  f"{len(sbj_info)} subjects ({sorted(sbj_info)})")
        else:
            print(f"[grab] no subject meshes under {subj_dir}; "
                  f"falling back to the shared mean template")

    return ds, lhand_layer, rhand_layer


def build_gigahands_dataset(cfg, split: str = "test"):
    """Build the MANO layers and the GigaHands dataset for the viewer.

    GigaHands stores its own MANO params (converted to this repo's x-data on the
    fly), so the dataset needs no preprocessed ``.npz``.  The subset filters
    (``object_number`` / ``object_rigidity``) and the object requirement are read
    from the config; the viewer only shows sequences that actually have an object
    trajectory, so ``require_object`` is forced on.

    Returns ``(ds, lhand_layer, rhand_layer)``.
    """
    from hoidata.dataloaders.gigahands_dataset import MotionGigaHands

    use_pca = cfg.get("use_pca", False)
    ncomps = 6 if use_pca else 45
    lhand_layer = ManoLayerWrapper(
        side="left", flat_hand_mean=cfg.get("flat_hand", False),
        mano_root=cfg.mano_root, use_pca=use_pca, ncomps=ncomps,
    ).cuda()
    rhand_layer = ManoLayerWrapper(
        side="right", flat_hand_mean=cfg.get("flat_hand", False),
        mano_root=cfg.mano_root, use_pca=use_pca, ncomps=ncomps,
    ).cuda()

    ds = MotionGigaHands(
        dataset_root=cfg.get("dataset_root", "data/GigaHands"),
        split=split,
        object_number=cfg.get("object_number", None),
        object_rigidity=cfg.get("object_rigidity", "rigid"),
        require_object=True,
        smooth_object=cfg.get("smooth_object", True),
        sequence_type_file=cfg.get("sequence_type_file", None),
        annotation_file=cfg.get("annotation_file", None),
        # The EasyMocap->manopth root-pivot correction in the dataset computes J0
        # with these layer settings; they MUST match the posing layers above or
        # the corrected translation won't line up with the rendered hand.
        mano_root=cfg.mano_root,
        flat_hand=cfg.get("flat_hand", False),
    )
    return ds, lhand_layer, rhand_layer


def _pose_hand_with_beta(x_hand, beta, layer):
    """(T, 99) x-data + (10,) beta -> (T, 778, 3) world-space MANO vertices.

    Unlike ``transform_xdata_to_verts`` (which assumes a zero shape), this uses
    the per-sequence GigaHands beta so the posed hand matches the fitted shape.
    """
    import torch
    from pytorch3d.transforms import matrix_to_axis_angle, rotation_6d_to_matrix

    xd = proc_torch_cuda(np.asarray(x_hand, dtype=np.float32))
    T = xd.shape[0]
    pose = matrix_to_axis_angle(
        rotation_6d_to_matrix(xd[:, 3:].reshape(-1, 6))).reshape(T, 48)
    betas = proc_torch_cuda(np.asarray(beta, dtype=np.float32)).reshape(1, -1)[:, :10]
    betas = betas.expand(T, 10).contiguous()
    out = layer(pose, betas, xd[:, :3])
    return proc_numpy(out.vertices).astype(np.float32)


def _pose_gigahands_object(cfg, item):
    """Load and pose the GigaHands object mesh per frame from the item's x_obj.

    ``x_obj`` is (T, 9) trans + rot6d holding the object's extrinsic matrix in
    GigaHands' convention (rotation ``R`` from ``_pytorch3d_quat_to_rotmat``); the
    mesh is placed as ``v @ R^T + t`` per frame — the same transform the reference
    loader applies (``base_v @ T[:3,:3].T + T[:3,3]``).

    Returns ``(obj_verts (T, V, 3), obj_faces (F, 3))``, or ``(None, None)`` when
    no mesh is available.
    """
    from pytorch3d.transforms import rotation_6d_to_matrix

    mesh_path = item.get("obj_mesh")
    x_obj = item.get("x_obj")
    if mesh_path is None or x_obj is None:
        return None, None

    mesh = trimesh.load(mesh_path, process=False, force="mesh")
    verts = np.asarray(mesh.vertices, dtype=np.float32)               # (V, 3)
    faces = np.asarray(mesh.faces)

    x_obj = np.asarray(x_obj, dtype=np.float32)[: item["nframes"]]
    trans = x_obj[:, :3]                                              # (T, 3)
    R = proc_numpy(rotation_6d_to_matrix(
        proc_torch_cuda(x_obj[:, 3:]).reshape(-1, 6)))                # (T, 3, 3)
    # v @ R^T + t   (== apply R^T, matching the reference object transform).
    obj_verts = (np.einsum("vj,tij->tvi", verts, R)
                 + trans[:, None, :]).astype(np.float32)              # (T, V, 3)
    return obj_verts, faces


def pose_gigahands_sample(cfg, ds, lhand_layer, rhand_layer, index: int = 0) -> dict:
    """Pose one GigaHands clip (hands + object) in the world frame for the viewer.

    Hands are built from ``x_lhand`` / ``x_rhand`` with the sequence's own MANO
    betas; only the tracked hands (``is_lhand`` / ``is_rhand``) are drawn.  The
    object mesh is loaded from ``object_meta`` and posed from ``x_obj``.
    """
    item = ds[index]
    nframes = int(item["nframes"])

    hand_sides = []
    hand_blocks = []
    if item["is_lhand"]:
        hand_sides.append("left")
        hand_blocks.append(_pose_hand_with_beta(
            item["x_lhand"], item["lhand_beta"], lhand_layer))
    if item["is_rhand"]:
        hand_sides.append("right")
        hand_blocks.append(_pose_hand_with_beta(
            item["x_rhand"], item["rhand_beta"], rhand_layer))
    # (T, 778*H, 3) matching the hand_sides order, or an empty (T,0,3) fallback.
    hand_verts = (np.concatenate(hand_blocks, axis=1)
                  if hand_blocks else np.zeros((nframes, 0, 3), np.float32))

    hand_faces = {
        "left": proc_numpy(lhand_layer.th_faces),
        "right": proc_numpy(rhand_layer.th_faces),
    }

    obj_verts, obj_faces = _pose_gigahands_object(cfg, item)
    if obj_verts is None:
        # No mesh: a degenerate single-point object so the serializer stays happy.
        obj_verts = np.zeros((nframes, 1, 3), np.float32)
        obj_faces = np.zeros((0, 3), np.int64)

    # Reference frame for the table: the first frame where the object is actually
    # tracked.  GigaHands leaves untracked frames at the origin, so a blind
    # frame-0 would place the object (and hence the table) at (0,0,0).
    obj_valid = item.get("obj_valid")
    table_frame = 0
    if obj_valid is not None:
        idxs = np.nonzero(np.asarray(obj_valid[:nframes]))[0]
        if len(idxs):
            table_frame = int(idxs[0])

    return {
        "hand_verts": hand_verts,
        "hand_faces": hand_faces,
        "hand_sides": hand_sides,
        "obj_verts": obj_verts,
        "obj_faces": obj_faces,
        "nframes": nframes,
        "obj_name": str(item.get("obj_name") or ""),
        "action_name": str(item.get("text") or ""),
        "labels": [],
        "subject": item["seq_name"],
        "grasps": {"left": [], "right": []},
        "table_frame": table_frame,
    }


def build_hot3d_dataset(cfg, split: str = "test"):
    """Build the MANO layers and the HOT3D dataset for the viewer.

    HOT3D x-data stores the absolute 45-D finger pose (PCA expanded with the
    MANO mean pose), so the layers use ``flat_hand_mean=True``.  HOT3D's hand
    shapes were fit with the corrected MANO_LEFT shape blend shapes, so the left
    layer gets the same fix (``fix_mano_left_shapedirs``) before posing with the
    recording's betas.

    Returns ``(ds, lhand_layer, rhand_layer)``.
    """
    from hoidata.dataloaders.hot3d_dataset import MotionHOT3DMultiObj
    from hoidata.utils.hand_proc import fix_mano_left_shapedirs

    use_pca = cfg.get("use_pca", False)
    ncomps = 6 if use_pca else 45
    lhand_layer = ManoLayerWrapper(
        side="left", flat_hand_mean=cfg.get("flat_hand", True),
        mano_root=cfg.mano_root, use_pca=use_pca, ncomps=ncomps,
    ).cuda()
    rhand_layer = ManoLayerWrapper(
        side="right", flat_hand_mean=cfg.get("flat_hand", True),
        mano_root=cfg.mano_root, use_pca=use_pca, ncomps=ncomps,
    ).cuda()
    fix_mano_left_shapedirs(lhand_layer, rhand_layer)

    ds = MotionHOT3DMultiObj(
        dataset_root=cfg.get("dataset_root", "data/hot3d_raw"),
        split=split,
        clip_len=cfg.get("clip_len", 300),
        clip_stride=cfg.get("clip_stride", None),
        min_clip_len=cfg.get("min_clip_len", 30),
        headsets=cfg.get("headsets", None),
        val_participants=cfg.get("val_participants", ["P0017"]),
        test_participants=cfg.get("test_participants", ["P0021"]),
        merge_tol_ms=cfg.get("merge_tol_ms", 5.0),
        mano_root=cfg.mano_root,
        canonical_z_up=cfg.get("canonical_z_up", True),
    )
    return ds, lhand_layer, rhand_layer


#: Decimated HOT3D object meshes, keyed by (glb path, max faces).
_HOT3D_MESH_CACHE = {}

#: Colors for HOT3D objects 1..K-1 (object 0 uses OBJECT_COLOR).
HOT3D_EXTRA_OBJECT_COLORS = [
    (0.95, 0.75, 0.35), (0.55, 0.80, 0.55), (0.85, 0.55, 0.80),
    (0.45, 0.80, 0.85), (0.90, 0.55, 0.45), (0.70, 0.70, 0.45),
]


def _load_hot3d_mesh(path, max_faces=4000):
    """Load a HOT3D ``.glb`` object mesh in metres, decimated for the viewer.

    The glb node transform scales the CAD model from mm to m; ``force="mesh"``
    bakes it in.  Meshes above ``max_faces`` (up to ~200k verts) are reduced by
    quadric decimation so a multi-object clip stays a reasonable payload
    (``max_faces <= 0`` keeps the full mesh).  Results are memoised.

    Returns ``(verts (V, 3) float32, faces (F, 3) int64)``.
    """
    key = (path, int(max_faces))
    if key in _HOT3D_MESH_CACHE:
        return _HOT3D_MESH_CACHE[key]
    mesh = trimesh.load(path, process=False, force="mesh")
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if max_faces > 0 and len(faces) > max_faces:
        o3d_mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(faces))
        # glTF splits vertices along UV seams; weld them so decimation does not
        # tear the surface apart.
        o3d_mesh = o3d_mesh.merge_close_vertices(1e-6)
        o3d_mesh = o3d_mesh.simplify_quadric_decimation(int(max_faces))
        o3d_mesh.remove_unreferenced_vertices()
        verts = np.asarray(o3d_mesh.vertices)
        faces = np.asarray(o3d_mesh.triangles, dtype=np.int64)
    _HOT3D_MESH_CACHE[key] = (verts.astype(np.float32), faces)
    return _HOT3D_MESH_CACHE[key]


def pose_hot3d_sample(cfg, ds, lhand_layer, rhand_layer, index: int = 0) -> dict:
    """Pose one HOT3D clip (hands + all objects) in the world frame for the viewer.

    Hands are built from ``x_lhand`` / ``x_rhand`` with the recording's MANO
    betas; only hands tracked somewhere in the clip are drawn.  Object 0 is the
    main object mesh (unless it is never tracked in the clip, then the first
    tracked object takes its place); the others are shipped as opaque
    ``extra_meshes``, each posed from its ``x_obj[:, k]`` as ``R @ v + t``.

    Untracked frames are not interpolated: per-frame visibility masks
    (``hand_visible`` / ``obj_visible`` / each extra's ``visible``, from the
    dataset's ``*_valid`` masks) make the viewer hide a hand / object on the
    frames where its pose is missing.  The table plane follows the recording's
    world up axis (Aria z-up → ``xoy``, Quest3 y-up → ``xoz``).
    """
    item = ds[index]
    nframes = int(item["nframes"])

    hand_sides = []
    hand_blocks = []
    hand_visible = {}
    for side, layer in (("left", lhand_layer), ("right", rhand_layer)):
        s = side[0]
        if not item[f"is_{s}hand"]:
            continue
        hand_sides.append(side)
        hand_blocks.append(_pose_hand_with_beta(
            item[f"x_{s}hand"], item[f"{s}hand_beta"], layer))
        hand_visible[side] = np.asarray(item[f"{s}hand_valid"][:nframes], bool)
    hand_verts = (np.concatenate(hand_blocks, axis=1)
                  if hand_blocks else np.zeros((nframes, 0, 3), np.float32))

    hand_faces = {
        "left": proc_numpy(lhand_layer.th_faces),
        "right": proc_numpy(rhand_layer.th_faces),
    }

    max_faces = int(cfg.get("viewer_max_obj_faces", 4000))
    x_obj = np.asarray(item["x_obj"], dtype=np.float32)[:nframes]       # (T, K, 9)
    obj_valid = np.asarray(item["obj_valid"], bool)[:nframes]           # (T, K)
    posed = []
    for k, mesh_path in enumerate(item["obj_meshes"]):
        verts, faces = _load_hot3d_mesh(mesh_path, max_faces)
        obj_verts = process_obj_result(
            proc_torch_cuda(verts), proc_torch_cuda(x_obj[:, k]), "hot3d")
        posed.append((proc_numpy(obj_verts).astype(np.float32), faces,
                      obj_valid[:, k]))

    # Main object: object 0, unless it is never tracked in this clip (the table
    # is placed under the main object, so it needs at least one tracked frame).
    tracked = [k for k in range(len(posed)) if posed[k][2].any()]
    main = 0 if (not tracked or 0 in tracked) else tracked[0]
    if posed:
        obj_verts, obj_faces, obj_visible = posed[main]
    else:
        obj_verts = np.zeros((nframes, 1, 3), np.float32)
        obj_faces = np.zeros((0, 3), np.int64)
        obj_visible = np.zeros(nframes, bool)
    extra_meshes = [
        {
            "name": f"obj{k}",
            "faces": faces,
            "verts": verts,
            "visible": visible,
            "color": HOT3D_EXTRA_OBJECT_COLORS[i % len(HOT3D_EXTRA_OBJECT_COLORS)],
            "opacity": 1.0,
        }
        for i, (k, (verts, faces, visible)) in enumerate(
            (k, p) for k, p in enumerate(posed) if k != main)
    ]

    # Table reference: the first frame with a tracked pose for the main object.
    idxs = np.nonzero(obj_visible)[0]
    table_frame = int(idxs[0]) if len(idxs) else 0

    return {
        "hand_verts": hand_verts,
        "hand_faces": hand_faces,
        "hand_sides": hand_sides,
        "hand_visible": hand_visible,
        "obj_verts": obj_verts,
        "obj_faces": obj_faces,
        "obj_visible": obj_visible,
        "obj_opacity": 1.0,
        "extra_meshes": extra_meshes,
        "nframes": nframes,
        "obj_name": ", ".join(item["obj_names"]),
        "action_name": item["seq_name"],
        "labels": [],
        "subject": item["recording"],
        "grasps": {"left": [], "right": []},
        "table_frame": table_frame,
        "table_plane": "xoz" if item["up_axis"] == "y" else "xoy",
    }


#: Timeline colors for the HOT3D single-object segment view: the inactive
#: transition frames before / after the core, and the object's active core
#: (named after the segmentation's ``seg_activity``: contact / moving).
HOT3D_SEGMENT_COLORS = {
    "transition": [0.95, 0.70, 0.20],   # amber — no object active
    "contact":    [0.20, 0.70, 0.75],   # teal  — only this object in contact
    "moving":     [0.20, 0.70, 0.75],   # teal  — only this object moving
}


def _mask_intervals(mask):
    """``[[start, end), ...]`` runs of True in a 1-D bool mask."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    edges = np.flatnonzero(m[1:] != m[:-1])
    return edges.reshape(-1, 2).tolist()


def build_hot3d_segment_dataset(cfg, split: str = "test"):
    """Build the MANO layers and the HOT3D single-object segment dataset.

    Runs any missing / stale preprocessing first (``obj.pkl`` / ``data.npz``,
    see ``preprocess_hot3d``), then loads ``MotionHOT3D`` with full-length
    segments (``max_nframes=-1``).  The layers are the shared defaults;
    ``MotionHOT3D`` derives per-participant shaped layers from them.

    Returns ``(ds, lhand_layer, rhand_layer)``.
    """
    from hoidata.dataloaders.hot3d_dataset import MotionHOT3D
    from hoidata.preprocess.preprocess_hot3d import run_missing_preprocessing

    run_missing_preprocessing(cfg)
    use_pca = cfg.get("use_pca", False)
    ncomps = 6 if use_pca else 45
    lhand_layer = ManoLayerWrapper(
        side="left", flat_hand_mean=cfg.get("flat_hand", True),
        mano_root=cfg.mano_root, use_pca=use_pca, ncomps=ncomps,
    ).cuda()
    rhand_layer = ManoLayerWrapper(
        side="right", flat_hand_mean=cfg.get("flat_hand", True),
        mano_root=cfg.mano_root, use_pca=use_pca, ncomps=ncomps,
    ).cuda()
    ds = MotionHOT3D(
        data_path=cfg.data_path,
        data_obj_pc_path=cfg.data_obj_pc_path,
        text_json=None,
        max_nframes=-1,
        task="evaluate",
        split=split,
        lhand_layer=lhand_layer,
        rhand_layer=rhand_layer,
        val_participants=cfg.get("val_participants", ["P0017"]),
        test_participants=cfg.get("test_participants", ["P0021"]),
    )
    return ds, lhand_layer, rhand_layer


def pose_hot3d_segment_sample(cfg, ds, lhand_layer, rhand_layer,
                              index: int = 0) -> dict:
    """Pose one HOT3D single-object segment (hands + its object) for the viewer.

    Hands come from ``MotionHOT3D._getitem_full`` (per-participant shaped
    layers); the object's ``.glb`` (decimated like the multi-object view) is
    posed from ``x_obj`` as ``R @ v + t``.  Untracked frames are hidden via the
    item's ``track_mask_*``.  The timeline's bottom track shows the segment's
    split into transition frames and its active core (contact / moving); the
    L / R grasp tracks show each hand's raw contact with the object.
    """
    item = ds[index]
    nframes = int(item["nframes"])

    hand_visible = {
        side: np.asarray(item[f"track_mask_{side[0]}hand"][:nframes], bool)
        for side in item["hand_sides"]
    }
    hand_faces = {
        "left": proc_numpy(lhand_layer.th_faces),
        "right": proc_numpy(rhand_layer.th_faces),
    }

    _, _, _, obj_path = ds.object_model(item["obj_name"])
    obj_root = cfg.get("obj_root", osp.join(cfg.dataset_root, "assets"))
    verts, obj_faces = _load_hot3d_mesh(
        osp.join(obj_root, obj_path), int(cfg.get("viewer_max_obj_faces", 4000)))
    obj_verts = proc_numpy(process_obj_result(
        proc_torch_cuda(verts), proc_torch_cuda(item["x_obj"][:nframes]),
        "hot3d")).astype(np.float32)
    obj_visible = np.asarray(item["track_mask_obj"][:nframes], bool)

    ms, me = int(item["move_start"]), int(item["move_end"])
    core = str(item.get("seg_activity", "moving"))
    segments = [{"start": s, "end": e, "type": t} for s, e, t in (
        (0, ms, "transition"), (ms, me, core), (me, nframes, "transition"))
        if e > s]
    grasps = {side: _mask_intervals(item[f"contact_{side[0]}hand"][:nframes])
              for side in ("left", "right")}

    idxs = np.nonzero(obj_visible)[0]
    return {
        "hand_verts": item["hand_verts"],
        "hand_faces": hand_faces,
        "hand_sides": item["hand_sides"],
        "hand_visible": hand_visible,
        "obj_verts": obj_verts,
        "obj_faces": obj_faces,
        "obj_visible": obj_visible,
        "nframes": nframes,
        "obj_name": str(item["obj_name"]),
        "action_name": item["seq_name"],
        "labels": [],
        "subject": str(item.get("subject", "")),
        "grasps": grasps,
        "segments": segments,
        "segment_colors": {k: HOT3D_SEGMENT_COLORS[k] for k in ("transition", core)},
        "table_frame": int(idxs[0]) if len(idxs) else 0,
        "table_plane": "xoz" if item["up_axis"] == "y" else "xoy",
    }


def pose_sample(cfg, dataset_name, ds, lhand_layer, rhand_layer, index: int = 0) -> dict:
    """Pose one full sequence from an already-built dataset, returning hand and
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
        # Per-sequence annotation intervals (ARCTIC full_motion only); empty list
        # when unavailable. Each: {start, end, hand: left|right|both, action}.
        "labels": item.get("labels", []),
        "subject": item.get("subject", ""),
        # Stable-grasp intervals per side: {"left": [[s,e],...], "right": [...]}.
        "grasps": item.get("grasps", {"left": [], "right": []}),
    }


def _fit_fixed_axis_motion(R_wrist_frame, t_wrist_frame):
    """Fit a fixed-axis / fixed-pivot model to wrist-frame object motion.

    Takes the object pose **already expressed in the wrist frame** — ``R`` (T,3,3)
    and ``t`` (T,3) — so the fit and any drawing share one frame by construction.

    Mirrors the estimator in ``scripts/dof_analysis.py``: the dominant rotation
    axis comes from the scatter of ``sin(theta) * u`` over frames, the per-frame
    signed angle from ``atan2`` of the trace-cosine and the axis-projected sine,
    and the pivot from the least-squares system ``(I - R_i) t = d_i``.

    Returns ``(R_rec, t_rec, axis, angle, resid_deg, pivot)`` where ``R_rec``
    (T,3,3) and ``t_rec`` (T,3) are the *reconstructed* wrist-frame object pose
    under the single-axis assumption, and ``pivot`` (3,) is the fitted fixed
    point on the axis (the perpendicular-part least-squares solution).
    """
    import torch
    from pytorch3d.transforms import axis_angle_to_matrix
    from pytorch3d.transforms.so3 import hat_inv

    R = torch.from_numpy(np.asarray(R_wrist_frame, dtype=np.float32))    # (T,3,3)
    t = torch.from_numpy(np.asarray(t_wrist_frame, dtype=np.float32))    # (T,3)

    sinaa = hat_inv((R - R.transpose(1, 2)) / 2)                      # sin(th)*u
    axis = torch.linalg.eigh(sinaa.transpose(0, 1) @ sinaa)[1][:, -1]    # (3,)
    cos_a = (torch.diagonal(R, dim1=1, dim2=2).sum(-1) - 1) / 2
    angle = torch.atan2(sinaa @ axis, cos_a)                             # (T,)
    R_rec = axis_angle_to_matrix(angle.unsqueeze(-1) * axis)             # (T,3,3)

    # Pivot: (I - R_i) t = t_i, rank-2 (axis component unobservable) → gelsd.
    # The system is homogeneous in t — the constant is already implied by the
    # model — so t must NOT be mean-centred here; centring destroys the very
    # signal the pivot is fitted from and biases the result.
    #
    # (I - R_i) maps onto the plane perpendicular to `axis`, so a fixed-pivot
    # model can ONLY produce translation perpendicular to the axis — it cannot
    # represent any motion along the axis.  Real grasps sit at a (roughly
    # constant) offset up the axis, which the pure model drops, leaving a
    # constant along-axis displacement between t_rec and t_true.  Split t into
    # its along-axis and perpendicular parts: fit the pivot on the perpendicular
    # part (which the model can represent) and add the mean along-axis offset
    # back so the reconstruction overlays the true object.
    I = torch.eye(3, dtype=R.dtype).expand_as(R)
    t_along = t @ axis                                                   # (T,) signed
    t_perp = t - t_along.unsqueeze(-1) * axis                            # (T,3)
    A = (I - R_rec).reshape(-1, 3)
    b = t_perp.reshape(-1)
    pivot = torch.linalg.lstsq(A, b, driver='gelsd').solution            # (3,)

    # Reconstructed translation: perpendicular part from the pivot model, plus
    # the constant along-axis offset the model cannot represent.
    t_rec = (torch.einsum('tij,j->ti', I - R_rec, pivot)
             + t_along.mean() * axis)                                    # (T,3)

    delta_R = R.transpose(1, 2) @ R_rec
    cos_d = (torch.diagonal(delta_R, dim1=1, dim2=2).sum(-1) - 1) / 2
    resid_deg = torch.rad2deg(torch.acos(cos_d.clamp(-1.0, 1.0)))        # (T,)

    return (proc_numpy(R_rec), proc_numpy(t_rec), proc_numpy(axis),
            proc_numpy(angle), proc_numpy(resid_deg), proc_numpy(pivot))


def _wrist_origin(x_hand, layer):
    """Posed wrist (root) joint position per frame, (T, 3).

    ``x_hand[:, :3]`` is MANO's global translation, which is *not* the wrist
    position — the mesh is built around the root joint.  Posing the layer and
    reading ``joints_w_tip[:, 0]`` gives the actual wrist.
    """
    import torch
    from pytorch3d.transforms import matrix_to_axis_angle, rotation_6d_to_matrix

    xd = proc_torch_cuda(np.asarray(x_hand, dtype=np.float32))
    T = xd.shape[0]
    pose = matrix_to_axis_angle(
        rotation_6d_to_matrix(xd[:, 3:].reshape(-1, 6))).reshape(T, 48)
    betas = torch.zeros(T, 10, device=xd.device)
    out = layer(pose, betas, xd[:, :3])
    return proc_numpy(out.joints_w_tip[:, 0]).astype(np.float32)


def pose_wrist_aligned_sample(cfg, ds, lhand_layer, rhand_layer, index: int = 0) -> dict:
    """Wrist-aligned original vs. fixed-axis-reconstructed object motion.

    Everything is drawn in the **wrist frame** of the transition hand: the hand
    is de-rotated/de-translated by its own per-frame root pose (so it sits still
    at the origin), and the object is shown twice in that same space —

      - ``obj_verts``  : the true object motion relative to the wrist;
      - extra mesh ``obj_recovered`` : the same mesh under the fitted fixed-axis /
        fixed-pivot model.

    Divergence between the two is exactly the model error reported as
    ``resid_angle`` by ``scripts/dof_analysis.py``.
    """
    from hoidata.utils.proc import transform_xdata_to_verts
    from pytorch3d.transforms import rotation_6d_to_matrix
    import torch

    item = ds[index]
    # Stable-grasp items expose is_lhand/is_rhand (one grasp = one hand); the
    # transition dataset exposes hand_sides instead. Accept either.
    if "hand_sides" in item:
        side = item["hand_sides"][0]
    else:
        side = "left" if item["is_lhand"] else "right"
    layer = lhand_layer if side == "left" else rhand_layer

    x_hand, x_object = item["x_hand"], item["x_object"]
    T = int(item["n_frames"])

    # World-space hand, then map into the wrist frame: v -> R_w^T (v - o_w).
    # The origin is the posed **wrist joint**, not x_hand[:, :3] — MANO applies
    # the latter as a global offset while the mesh is built around the root
    # joint, so subtracting it would leave the hand off-origin.
    hand_verts = transform_xdata_to_verts(x_hand, layer).astype(np.float32)  # (T,778,3)
    xh = torch.from_numpy(np.asarray(x_hand, dtype=np.float32))
    wrist_R = proc_numpy(rotation_6d_to_matrix(xh[:, 3:9]))                        # (T,3,3)
    wrist_o = _wrist_origin(x_hand, layer)                                   # (T,3)
    hand_local = np.einsum('tji,tvj->tvi', wrist_R, hand_verts - wrist_o[:, None])

    # Canonical object geometry (GRAB applies R^T — see process_obj_result).
    # Prefer the full .ply mesh; fall back to the sampled point cloud from
    # obj.pkl when the raw GRAB meshes aren't present (renders as points, which
    # is enough to compare the two trajectories).
    _, obj_pc, _, obj_path = ObjectModel(cfg.data_obj_pc_path)(item["obj_name"])
    mesh_path = osp.join(cfg.obj_root, obj_path)
    if osp.isfile(mesh_path):
        mesh = trimesh.load(mesh_path, process=False)
        verts0 = np.asarray(mesh.vertices, dtype=np.float32)                 # (V,3)
        obj_faces = np.asarray(mesh.faces)
    else:
        verts0 = np.asarray(obj_pc, dtype=np.float32)                        # (P,3)
        obj_faces = np.zeros((0, 3), dtype=np.int64)                         # points only

    # True wrist-frame object pose, using the same wrist origin as the hand.
    #
    # GRAB stores the object rotation un-inverted (rot6d of R_o), and the
    # rotation actually *applied* to the mesh at display time is R_o^T — see
    # process_obj_result's grab branch (einsum 'tij,ki->tkj' == verts @ R_o ==
    # applies R_o^T).  In world space a vertex lands at  R_o^T v0 + t_o.  Mapping
    # that into the wrist frame,  v_wrist = R_w^T (v_world - o_w),  gives an
    # applied rotation  R_true = R_w^T R_o^T = (R_o R_w)^T  and translation
    # t_true = R_w^T (t_o - o_w).  R_true is the object's actual orientation in
    # the wrist frame, so both the fixed-axis fit and the drawing use it
    # directly (plain multiply, no extra transpose).
    xo = torch.from_numpy(np.asarray(x_object, dtype=np.float32))
    obj_R = proc_numpy(rotation_6d_to_matrix(xo[:, 3:9]))                          # R_o (stored)
    obj_t = np.asarray(x_object[:, :3], dtype=np.float32)
    R_true = np.einsum('tji,tkj->tik', wrist_R, obj_R)                       # R_w^T R_o^T
    t_true = np.einsum('tji,tj->ti', wrist_R, obj_t - wrist_o)

    R_rec, t_rec, axis, angle, resid_deg, _pivot = _fit_fixed_axis_motion(R_true, t_true)

    # R_true/R_rec are already the applied wrist-frame rotation → direct multiply.
    obj_true = np.einsum('tij,kj->tki', R_true, verts0) + t_true[:, None]
    obj_rec = np.einsum('tij,kj->tki', R_rec, verts0) + t_rec[:, None]

    hand_faces = {
        "left": proc_numpy(lhand_layer.th_faces),
        "right": proc_numpy(rhand_layer.th_faces),
    }

    if hasattr(ds, "transition_label"):
        label = ds.transition_label(index)
    else:
        label = f"{item['obj_name']} {side} grasp #{index}"
    return {
        "hand_verts": hand_local.astype(np.float32),
        "hand_faces": hand_faces,
        "hand_sides": [side],
        "obj_verts": obj_true.astype(np.float32),
        "obj_faces": obj_faces,
        # Reconstructed object, same mesh/topology, drawn in the same space.
        "extra_meshes": [{
            "name": "obj_recovered",
            "verts": obj_rec.astype(np.float32),
            "faces": obj_faces,
            "color": [0.95, 0.55, 0.20],
            "opacity": 0.5,
        }],
        "nframes": T,
        "obj_name": item["obj_name"],
        "action_name": f"{label} | resid={resid_deg.mean():.1f}deg",
        "labels": [],
        "subject": "",
        "grasps": {"left": [], "right": []},
    }


def build_transition_dataset(cfg, split: str = "test", aug_p=None):
    """Build the MANO layers and a :class:`TransitionMotionGRAB` dataset.

    Mirrors :func:`build_dataset` but constructs the transition dataset (indexed
    per transition, not per sequence) with the config's transition window bounds.
    ``aug_p`` overrides the config's ``aug_p`` (the random-trim augmentation
    probability on "01"/"10" windows); pass ``0.0`` to view raw windows, or leave
    ``None`` to use the config value so the viewer shows augmented windows.
    Returns ``(ds, lhand_layer, rhand_layer)``.
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

    if aug_p is None:
        aug_p = cfg.get("aug_p", 0.0)

    ds = TransitionMotionGRAB(
        data_path=cfg.data_path,
        data_obj_pc_path=cfg.data_obj_pc_path,
        text_json=cfg.text_json,
        max_nframes=-1,
        task="evaluate",
        split=split,
        lhand_layer=lhand_layer,
        rhand_layer=rhand_layer,
        min_len=cfg.get("transition_min_len", 10),
        max_len=cfg.get("transition_max_len", 100),
        aug_p=aug_p,
    )
    return ds, lhand_layer, rhand_layer


def build_bimanual_transition_dataset(cfg, split: str = "test", aug_p=None):
    """Build the MANO layers and a :class:`BimanualTransitionMotionGRAB` dataset.

    Like :func:`build_transition_dataset` but the two hands are merged per sequence
    into a grasp-count timeline, so windows are 0↔1↔2 count transitions
    (types ``01/02/10/12/20/21``) rather than per-hand ``01/10/11``.
    Returns ``(ds, lhand_layer, rhand_layer)``.
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

    if aug_p is None:
        aug_p = cfg.get("aug_p", 0.0)

    ds = BimanualTransitionMotionGRAB(
        data_path=cfg.data_path,
        data_obj_pc_path=cfg.data_obj_pc_path,
        text_json=cfg.text_json,
        max_nframes=-1,
        task="evaluate",
        split=split,
        lhand_layer=lhand_layer,
        rhand_layer=rhand_layer,
        min_len=cfg.get("transition_min_len", 10),
        max_len=cfg.get("transition_max_len", 100),
        aug_p=aug_p,
    )
    return ds, lhand_layer, rhand_layer


def build_stable_grasp_dataset(cfg, dataset_name: str, split: str = "test"):
    """Build the MANO layers and a stable-grasp dataset (one item per grasp).

    Mirrors :func:`build_dataset` but constructs :class:`StableGraspGRAB` /
    :class:`StableGraspARCTIC`, indexed per stable-grasp CSV row rather than per
    sequence.  Returns ``(ds, lhand_layer, rhand_layer)``.
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

    dataset_cls = {"grab": StableGraspGRAB, "arctic": StableGraspARCTIC}[dataset_name]
    ds = dataset_cls(
        data_path=cfg.data_path,
        data_obj_pc_path=cfg.data_obj_pc_path,
        text_json=cfg.text_json,
        max_nframes=-1,
        task="evaluate",
        split=split,
        lhand_layer=lhand_layer,
        rhand_layer=rhand_layer,
        stable_grasp_csv=cfg.get("stable_grasp_csv"),
    )
    return ds, lhand_layer, rhand_layer


def pose_transition_sample(cfg, ds, lhand_layer, rhand_layer, index: int = 0) -> dict:
    """Pose one GRAB transition window from a :class:`TransitionMotionGRAB`.

    Unlike :func:`pose_sample` (which re-poses the whole sequence's object off
    ``ds.x_obj[index]``), a transition item already carries its own sliced
    ``x_hand``/``x_object`` windows, so both hand and object are posed directly
    from the item.  Returns the same dict shape the viewer expects (note the
    viewer's own key is ``nframes``, while the dataset item uses ``n_frames``).
    """
    from hoidata.utils.proc import transform_xdata_to_verts

    item = ds[index]
    side = item["hand_sides"][0]
    layer = lhand_layer if side == "left" else rhand_layer

    # Hand: raw x-data window → world-space MANO vertices for the single hand.
    hand_verts = transform_xdata_to_verts(item["x_hand"], layer).astype(np.float32)  # (Tt,778,3)

    # Object: rigid GRAB mesh posed per frame from the sliced x_object window.
    _, _, _, obj_path = ObjectModel(cfg.data_obj_pc_path)(item["obj_name"])
    mesh = trimesh.load(osp.join(cfg.obj_root, obj_path), process=False)
    obj_verts_t = process_obj_result(
        proc_torch_cuda(np.asarray(mesh.vertices, dtype=np.float32)),
        proc_torch_cuda(item["x_object"]),
        "grab",
    )
    obj_verts = proc_numpy(obj_verts_t)
    obj_faces = np.asarray(mesh.faces)

    hand_faces = {
        "left": proc_numpy(lhand_layer.th_faces),
        "right": proc_numpy(rhand_layer.th_faces),
    }

    return {
        "hand_verts": hand_verts,
        "hand_faces": hand_faces,
        "hand_sides": item["hand_sides"],
        "obj_verts": obj_verts,
        "obj_faces": obj_faces,
        "nframes": item["n_frames"],
        "obj_name": item["obj_name"],
        # Label carries object / hand / transition-type (see transition_label).
        "action_name": ds.transition_label(index),
        "labels": [],
        "subject": "",
        "grasps": {"left": [], "right": []},
    }


def pose_bimanual_transition_sample(
        cfg, ds, lhand_layer, rhand_layer, index: int = 0) -> dict:
    """Pose one bimanual transition window from a :class:`BimanualTransitionMotionGRAB`.

    The item exposes each posed hand separately as ``x_lhand`` / ``x_rhand``
    (each ``(Tt, D)`` or ``None``); ``item["hand_sides"]`` lists which are present.
    Each hand is posed independently and the world-space vertex blocks are
    concatenated along the vertex axis into ``(Tt, 778*H, 3)`` — the layout the
    viewer's ``build_hand_mesh`` expects (it slices 778-vertex blocks per side).
    The object is posed directly from the item's sliced ``x_object`` window,
    exactly like the unimanual poser.
    """
    from hoidata.utils.proc import transform_xdata_to_verts

    item = ds[index]
    sides = list(item["hand_sides"])

    hand_blocks = []
    for side in sides:
        layer = lhand_layer if side == "left" else rhand_layer
        xh = np.asarray(item["x_lhand" if side == "left" else "x_rhand"],
                        dtype=np.float32)                                  # (Tt, D)
        verts = transform_xdata_to_verts(xh, layer).astype(np.float32)     # (Tt,778,3)
        hand_blocks.append(verts)
    hand_verts = np.concatenate(hand_blocks, axis=1)  # (Tt, 778*H, 3)

    # Object: rigid GRAB mesh posed per frame from the sliced x_object window.
    obj_model = ObjectModel(cfg.data_obj_pc_path)
    _, obj_pc, _obj_pc_normal, obj_path = obj_model(item["obj_name"])
    mesh = trimesh.load(osp.join(cfg.obj_root, obj_path), process=False)
    mesh_verts_canon = np.asarray(mesh.vertices, dtype=np.float32)          # (Vo,3)
    obj_verts_t = process_obj_result(
        proc_torch_cuda(mesh_verts_canon),
        proc_torch_cuda(item["x_object"]),
        "grab",
    )
    obj_verts = proc_numpy(obj_verts_t)
    obj_faces = np.asarray(mesh.faces)

    # Per-frame, per-hand contact map over the window, colored on the full-res
    # mesh: each mesh vertex takes the contact of its nearest sampled point.  The
    # dataset returns per-hand maps (Tt, 2, O) = [left, right]; both are kept so
    # the viewer can color the two hands' contact distinctly (and pick the larger
    # where a vertex touches both).  The canonical-space nearest-sampled-point map
    # is frame-invariant (mesh and pc move rigidly together), so compute it once.
    contact_lr = ds._window_contact(
        item["seq_idx"], item["start_frame"], item["end_frame"], sides)     # (Tt,2,O)
    from scipy.spatial import cKDTree
    nn_idx = cKDTree(obj_pc.astype(np.float32)).query(mesh_verts_canon)[1]   # (Vo,)
    obj_contact = contact_lr[:, :, nn_idx].astype(np.float32)               # (Tt,2,Vo)

    hand_faces = {
        "left": proc_numpy(lhand_layer.th_faces),
        "right": proc_numpy(rhand_layer.th_faces),
    }

    return {
        "hand_verts": hand_verts,
        "hand_faces": hand_faces,
        "hand_sides": sides,
        "obj_verts": obj_verts,
        "obj_faces": obj_faces,
        "obj_contact": obj_contact,          # (Tt, 2, Vo) per-frame per-hand [L,R] contact
        "nframes": item["n_frames"],
        "obj_name": item["obj_name"],
        # Label carries object / driving hand(s) / transition-type.
        "action_name": ds.transition_label(index),
        "labels": [],
        "subject": "",
        "grasps": {"left": [], "right": []},
    }


def build_bimanual_stable_grasp_dataset(cfg, split: str = "test"):
    """Build the MANO layers and a :class:`BimanualStableGraspGRAB` dataset.

    Mirrors :func:`build_stable_grasp_dataset` but constructs the *bimanual*
    stable-grasp dataset (one item per constant-grasp-count segment, both hands'
    grasps merged).  GRAB only.  ``min_len`` comes from the config
    (``stable_grasp_min_len``, else ``transition_min_len``, else 10).
    Returns ``(ds, lhand_layer, rhand_layer)``.
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

    min_len = cfg.get("stable_grasp_min_len", cfg.get("transition_min_len", 10))
    ds = BimanualStableGraspGRAB(
        data_path=cfg.data_path,
        data_obj_pc_path=cfg.data_obj_pc_path,
        text_json=cfg.text_json,
        max_nframes=-1,
        task="evaluate",
        split=split,
        lhand_layer=lhand_layer,
        rhand_layer=rhand_layer,
        stable_grasp_csv=cfg.get("stable_grasp_csv"),
        min_len=min_len,
    )
    return ds, lhand_layer, rhand_layer


def pose_bimanual_stable_grasp_sample(
        cfg, ds, lhand_layer, rhand_layer, index: int = 0) -> dict:
    """Pose one bimanual stable-grasp segment from :class:`BimanualStableGraspGRAB`.

    Everything is drawn in the **world frame** (no wrist alignment): each active
    hand's raw ``x_lhand`` / ``x_rhand`` window is posed to world-space MANO
    vertices and the object is posed from the item's ``x_object`` window.  No
    fixed-axis / 1-DOF fit is done — this simply shows the true motion.  The
    per-hand vertex blocks are concatenated into ``(Tt, 778*H, 3)`` for the
    viewer's ``build_hand_mesh``.
    """
    from hoidata.utils.proc import transform_xdata_to_verts

    item = ds[index]
    # Which hands are grasping in this segment (bimanual "both" has both).
    sides = []
    if item["is_lhand"]:
        sides.append("left")
    if item["is_rhand"]:
        sides.append("right")

    hand_blocks = []
    for side in sides:
        layer = lhand_layer if side == "left" else rhand_layer
        xh = np.asarray(item["x_lhand" if side == "left" else "x_rhand"],
                        dtype=np.float32)                                  # (Tt, D)
        verts = transform_xdata_to_verts(xh, layer).astype(np.float32)     # (Tt,778,3)
        hand_blocks.append(verts)
    hand_verts = np.concatenate(hand_blocks, axis=1)  # (Tt, 778*H, 3)

    # Object: rigid GRAB mesh posed per frame from the item's x_object window.
    _, _, _, obj_path = ObjectModel(cfg.data_obj_pc_path)(item["obj_name"])
    mesh = trimesh.load(osp.join(cfg.obj_root, obj_path), process=False)
    obj_verts_t = process_obj_result(
        proc_torch_cuda(np.asarray(mesh.vertices, dtype=np.float32)),
        proc_torch_cuda(item["x_object"]),
        "grab",
    )
    obj_verts = proc_numpy(obj_verts_t)
    obj_faces = np.asarray(mesh.faces)

    hand_faces = {
        "left": proc_numpy(lhand_layer.th_faces),
        "right": proc_numpy(rhand_layer.th_faces),
    }

    label = (f"{item['obj_name']} {item['hand_side']} grasp #{index}"
             + (f" | {item['text']}" if item.get("text") else ""))
    return {
        "hand_verts": hand_verts,
        "hand_faces": hand_faces,
        "hand_sides": sides,
        "obj_verts": obj_verts,
        "obj_faces": obj_faces,
        "nframes": item["n_frames"],
        "obj_name": item["obj_name"],
        "action_name": label,
        "labels": [],
        "subject": "",
        "grasps": {"left": [], "right": []},
    }


# Grasp-timeline segmentation (segment_grasp_timeline / TA_HALF / MIN_NULL) lives
# with the dataset machinery (imported at the top) so the viewer and
# SegmentedMotionGRAB share one implementation.


def pose_segmented_sample(cfg, dataset_name, ds, lhand_layer, rhand_layer,
                          index: int = 0) -> dict:
    """Pose a full sequence and attach its null/transition/stable-grasp split.

    Delegates to :func:`pose_sample` (world-frame full sequence, hands + object),
    then adds a ``segments`` timeline computed from the item's stable-grasp
    intervals (see :func:`segment_grasp_timeline`) for the viewer's bottom track.
    """
    sample = pose_sample(cfg, dataset_name, ds, lhand_layer, rhand_layer, index)
    grasps = sample.get("grasps", {"left": [], "right": []})
    sample["segments"] = segment_grasp_timeline(grasps, sample["nframes"])
    return sample


def load_sample(cfg, dataset_name: str, split: str = "test", index: int = 0) -> dict:
    """Convenience wrapper: build the dataset and pose one sequence in one call.

    Equivalent to ``build_dataset(...)`` followed by ``pose_sample(...)``.
    Kept for the standalone o3d CLI and existing callers; a long-lived server
    should instead call ``build_dataset`` once and ``pose_sample`` per index.
    """
    ds, lhand_layer, rhand_layer = build_dataset(cfg, dataset_name, split)
    return pose_sample(cfg, dataset_name, ds, lhand_layer, rhand_layer, index)


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
