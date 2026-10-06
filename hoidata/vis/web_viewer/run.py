"""Local web app for interactive HOI (hand-object interaction) visualization.

Serves a single-page Three.js viewer that plays a posed ARCTIC/GRAB/GigaHands/
HOT3D sequence in a loop with play/pause, a draggable frame progress bar, a
sequence dropdown, and prev/next buttons — with free mouse-drag camera orbit,
matching the native
``vis/o3d_visualizer.py`` window.

The heavy lifting (dataset loading, MANO posing, object posing) is reused
directly from ``vis.o3d_visualizer`` — this module is a thin web adapter:

- the dataset is built ONCE at startup (``build_dataset``); the sequence list
  (index/object/action/nframes) is read straight off the dataset arrays with no
  posing, so startup is cheap;
- each sequence is posed on the GPU the first time it is requested
  (``pose_sample``) and then cached in memory.

Per-sequence geometry is shipped to the browser as one binary blob (float32
vertices + int32 faces, concatenated) plus a compact JSON manifest in the
``X-Meta`` response header, so the client reads it straight into typed arrays.

Usage:
    python -m vis.web_viewer.run --config configs/arctic.yaml --split test \
        [--host 127.0.0.1] [--port 8080] [--fps 30]
"""

import argparse
import json
import os
import os.path as osp
import shutil

import numpy as np
from flask import Flask, Response, jsonify, render_template, request
from omegaconf import OmegaConf

from hoidata.vis.o3d_visualizer import (
    build_dataset, pose_sample, build_transition_dataset, pose_transition_sample,
    build_bimanual_transition_dataset, pose_bimanual_transition_sample,
    build_stable_grasp_dataset, pose_wrist_aligned_sample,
    build_bimanual_stable_grasp_dataset, pose_bimanual_stable_grasp_sample,
    pose_segmented_sample, build_gigahands_dataset, pose_gigahands_sample,
    build_hot3d_dataset, pose_hot3d_sample)
from hoidata.preprocess.preprocess_arctic import parse_description_labels

HAND_SIDE_COLORS = {"left": [0.9, 0.75, 0.65], "right": [0.75, 0.85, 0.95]}
OBJECT_COLOR = [0.6, 0.7, 0.9]

# Table-top surface (drawn under the object at its first-frame resting height).
TABLE_SIZE = 0.5                  # side length in meters (0.5m x 0.5m square)
TABLE_COLOR = [0.45, 0.40, 0.35]  # muted brown

# Progress-bar / active-label colors by annotated hand type (0..1 RGB).
LABEL_COLORS = {
    "left":  [0.20, 0.45, 0.95],   # blue
    "right": [0.90, 0.25, 0.25],   # red
    "both":  [0.25, 0.75, 0.35],   # green
}

# Action-segmentation track colors (--segmented): gray = null grasp.
SEGMENT_COLORS = {
    "null":         [0.50, 0.50, 0.50],   # gray  — no hand in stable grasp
    "transition":   [0.95, 0.70, 0.20],   # amber — transition action (±5 frames)
    "stable_grasp": [0.25, 0.70, 0.40],   # green — one or both hands in stable grasp
}

app = Flask(__name__)

# Populated by main() before app.run().
STATE = {
    "cfg": None,
    "dataset_name": None,
    "split": None,
    "ds": None,
    "lhand_layer": None,
    "rhand_layer": None,
    "sequences": [],   # list of {index, obj_name, action_name, nframes}
    "cache": {},        # index -> (meta_dict, bytes)
    "fps": 30,
    "transition": False,  # True → GRAB transition-motion dataset
    "bimanual": False,   # True → GRAB bimanual (merged-hand) transition dataset
    "wrist_aligned": False,  # True → wrist-frame original vs. reconstructed object
    "stable_grasp": False,   # True → world-frame bimanual stable-grasp segments
    "segmented": False,      # True → full sequence + null/transition/SG action track
    "label_mode": False,     # True → action-label editing enabled (ARCTIC only)
    "label_dir": None,        # working-copy dir for editable description.txt files
    "seq_meta": [],           # index -> {subject, action_name, nframes} (label mode)
}


def _label_dir(cfg, dataset_name):
    """Working-copy dir for editable descriptions: ``data/{dataset_name}_description``.

    Derived from ``cfg.text_root``'s grandparent so it lands inside the repo ``data/``
    tree regardless of the current working directory
    (text_root == ``data/arctic_raw/description_text2hoi`` → ``data/arctic_description``).
    """
    data_root = osp.dirname(osp.dirname(cfg.text_root))
    return osp.join(data_root, f"{dataset_name}_description")


def _ensure_label_dir(cfg, dataset_name):
    """Create ``data/{dataset_name}_description`` as a copy of ``cfg.text_root`` if absent.

    Seeds the working copy once; a later run reuses the existing (edited) copy — the raw
    ``cfg.text_root`` is never written to by this feature.
    """
    dst = _label_dir(cfg, dataset_name)
    if not osp.isdir(dst):
        shutil.copytree(cfg.text_root, dst)
        print(f"[label] seeded working copy: {cfg.text_root} -> {dst}")
    else:
        print(f"[label] reusing working copy: {dst}")
    return dst


def _build_seq_meta(ds):
    """Cheap per-index identity map (no posing) for mapping index → description file.

    Only meaningful for the ARCTIC full_motion dataset, where ``subject`` +
    ``action_name`` (= sequence name) locate the sequence's ``description.txt``.
    """
    return [{
        "subject": str(ds.subject[i]),
        "action_name": str(ds.action_name[i]),
        "nframes": int(ds.nframes[i]),
    } for i in range(len(ds))]


def _description_path(index):
    """Working-dir ``description.txt`` for a sequence index."""
    m = STATE["seq_meta"][index]
    return osp.join(STATE["label_dir"], m["subject"], m["action_name"], "description.txt")


def _read_labels(index):
    """Parse the working-dir ``description.txt`` → 0-indexed ``[start, end)`` labels.

    Clamped to the sequence's frame count; degenerate intervals dropped; a missing file
    yields ``[]``.  This is the single source of truth for the action track in label mode.
    """
    path = _description_path(index)
    nframes = STATE["seq_meta"][index]["nframes"]
    if not osp.exists(path):
        return []
    out = []
    for l in parse_description_labels(path):
        s = max(0, int(l["start"]))
        e = min(nframes, int(l["end"]))
        if e > s:
            out.append({"start": s, "end": e, "hand": l["hand"],
                        "action": l["action"]})
    return out


def _write_description(path, labels):
    """Write labels (0-indexed ``[start, end)``) to a ``description.txt``.

    Byte-compatible with the raw ARCTIC files: one line per interval, sorted by
    (start, end), formatted ``"{start+1}-{end} {action}, {token}"`` where ``token`` is
    ``"both hands"`` for a both-hand interval else ``"{hand} hand"``.  The ``start+1``
    makes the start 1-indexed; the ``end`` is already exclusive (consecutive intervals
    share the boundary frame), matching ``parse_description_labels``.  Empty labels → an
    empty file; parent dirs are created as needed.
    """
    os.makedirs(osp.dirname(path), exist_ok=True)
    lines = []
    for l in sorted(labels, key=lambda d: (d["start"], d["end"])):
        sf = int(l["start"]) + 1
        ef = int(l["end"])
        hand = l["hand"]
        token = "both hands" if hand == "both" else f"{hand} hand"
        action = str(l["action"]).strip()
        lines.append(f"{sf}-{ef} {action}, {token}")
    with open(path, "w") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")


def _build_sequence_list(ds, transition=False, wrist_aligned=False,
                         stable_grasp=False):
    """Cheap per-index metadata for the dropdown — no MANO/GPU posing.

    Uses the ``_object_name`` hook (uniform across GRAB/ARCTIC) rather than a
    dataset-specific attribute: ARCTIC stores ``object_name`` while GRAB stores
    ``obj_name``, but both expose ``_object_name(index)``.

    For the transition dataset each index is a *transition* (not a sequence), so
    per-transition helpers (``transition_label`` / ``transition_nframes``) supply
    the label and length.  For the wrist-aligned and bimanual stable-grasp views
    each index is a flat per-grasp/per-segment row: the label carries the grasp
    ``hand_side`` and the length comes from its start/end frames.
    """
    seqs = []
    for i in range(len(ds)):
        if wrist_aligned or stable_grasp:
            sg = ds.stable_grasps
            side = sg["hand_side"][i]
            nframes = int(sg["end_frame"][i]) - int(sg["start_frame"][i])
            label = f"{ds._object_name(i)} {side} grasp #{i}"
            # Append the natural-language description (bimanual stable-grasp
            # view).  _build_sequence_list runs at startup with no posing, but
            # _sg_text is cheap (no GPU) — call it directly off the flat index so
            # the dropdown carries the same text the viewport shows.
            if stable_grasp and hasattr(ds, "_sg_text"):
                text = ds._sg_text(int(sg["seq_idx"][i]),
                                   bool(sg["is_lhand"][i]), bool(sg["is_rhand"][i]))
                if text:
                    label = f"{label} | {text}"
            seqs.append({
                "index": i,
                "obj_name": str(ds._object_name(i)),
                "action_name": label,
                "nframes": nframes,
            })
        elif transition:
            seqs.append({
                "index": i,
                "obj_name": str(ds._object_name(i)),
                "action_name": ds.transition_label(i),
                "nframes": int(ds.transition_nframes(i)),
            })
        else:
            seqs.append({
                "index": i,
                "obj_name": str(ds._object_name(i)),
                "action_name": str(ds.action_name[i]),
                "nframes": int(ds.nframes[i]),
            })
    return seqs


def _build_gigahands_sequence_list(ds):
    """Dropdown metadata for the GigaHands dataset (no MANO/GPU posing).

    Each index is a full clip; the label is the object name plus the caption, and
    the length comes from the cheap ``clip_meta`` (tracked-frame window).
    """
    seqs = []
    for i in range(len(ds)):
        m = ds.clip_meta(i)
        obj = m["obj_name"]
        label = f"{obj} | {m['text']}" if m["text"] else (obj or m["seq_name"])
        seqs.append({
            "index": i,
            "obj_name": str(obj),
            "action_name": label,
            "nframes": int(m["nframes"]),
        })
    return seqs


def _build_hot3d_sequence_list(ds):
    """Dropdown metadata for the HOT3D dataset (no pose parsing / MANO posing).

    Each index is one fixed-length clip of a recording; the label names the
    recording, headset, frame window and the recording's objects.
    """
    seqs = []
    for i in range(len(ds)):
        m = ds.clip_meta(i)
        window = m["seq_name"].split(":", 1)[1]
        objs = ", ".join(m["obj_names"])
        seqs.append({
            "index": i,
            "obj_name": objs,
            "action_name": f"{m['recording']} [{m['headset']}] {window} | {objs}",
            "nframes": int(m["nframes"]),
        })
    return seqs


def _serialize_sample(sample):
    """Pack a posed sample into (meta_dict, blob_bytes).

    The blob is a single little-endian byte string holding, in order:
      - object faces         (int32,   (Fo, 3))
      - per-side hand faces   (int32,   (1538, 3)) for each active side
      - object verts          (float32, (T, Vo, 3))
      - per-side hand verts    (float32, (T, 778, 3)) for each active side
      - optional per-frame visibility masks (int32, (T,)): ``obj_visible``,
        ``hand_visible_{side}``, ``extra_visible_{name}`` — 0 hides that mesh
        on the frame (e.g. HOT3D frames whose pose was not tracked)
    The manifest records dtype / offset (bytes) / shape for every array so the
    client can slice the ArrayBuffer without guessing strides.
    """
    parts = []       # list of np.ndarray (contiguous, correct dtype)
    manifest = {}
    offset = 0

    def _add(name, arr, dtype):
        nonlocal offset
        a = np.ascontiguousarray(arr, dtype=dtype)
        manifest[name] = {
            "dtype": "float32" if dtype == np.float32 else "int32",
            "offset": offset,
            "shape": list(a.shape),
        }
        parts.append(a)
        offset += a.nbytes

    nframes = int(sample["nframes"])
    hand_sides = list(sample["hand_sides"])
    obj_verts = np.asarray(sample["obj_verts"])[:nframes]      # (T, Vo, 3)
    obj_faces = np.asarray(sample["obj_faces"])                # (Fo, 3)
    hand_verts = np.asarray(sample["hand_verts"])[:nframes]    # (T, 778*H, 3)

    _add("obj_faces", obj_faces, np.int32)
    for side in hand_sides:
        _add(f"hand_faces_{side}", sample["hand_faces"][side], np.int32)

    # Optional extra meshes (e.g. a reconstructed object shown alongside the
    # original).  Each carries its own faces/verts/color and is drawn in the
    # same space as everything else.
    extras = sample.get("extra_meshes", []) or []
    for ex in extras:
        _add(f"extra_faces_{ex['name']}", ex["faces"], np.int32)

    _add("obj_verts", obj_verts, np.float32)
    for i, side in enumerate(hand_sides):
        _add(f"hand_verts_{side}", hand_verts[:, i * 778:(i + 1) * 778], np.float32)
    for ex in extras:
        _add(f"extra_verts_{ex['name']}", np.asarray(ex["verts"])[:nframes], np.float32)

    # Optional per-frame, per-hand object-vertex contact map (Tt, 2, Vo) =
    # [left, right]: drives a two-color heatmap of the object mesh in the client.
    has_contact = sample.get("obj_contact") is not None
    if has_contact:
        _add("obj_contact", np.asarray(sample["obj_contact"])[:nframes], np.float32)

    # Optional per-frame visibility masks (T,): the client hides a mesh on frames
    # where its mask is 0 (pose not tracked). Meshes without a mask always show.
    if sample.get("obj_visible") is not None:
        _add("obj_visible", np.asarray(sample["obj_visible"])[:nframes], np.int32)
    hand_visible = sample.get("hand_visible") or {}
    for side in hand_sides:
        if hand_visible.get(side) is not None:
            _add(f"hand_visible_{side}", np.asarray(hand_visible[side])[:nframes], np.int32)
    for ex in extras:
        if ex.get("visible") is not None:
            _add(f"extra_visible_{ex['name']}", np.asarray(ex["visible"])[:nframes], np.int32)

    # Annotation intervals (may be empty). Keys are plain ints/str → JSON-safe.
    labels = [
        {"start": int(l["start"]), "end": int(l["end"]),
         "hand": str(l["hand"]), "action": str(l.get("action", ""))}
        for l in sample.get("labels", [])
    ]

    # Stable-grasp intervals per side → JSON-safe [[start, end], ...].
    raw_grasps = sample.get("grasps", {}) or {}
    grasps = {
        side: [[int(s), int(e)] for s, e in raw_grasps.get(side, [])]
        for side in ("left", "right")
    }

    # Optional action-segmentation track (null / transition / stable_grasp).
    segments = [
        {"start": int(s["start"]), "end": int(s["end"]), "type": str(s["type"])}
        for s in sample.get("segments", [])
    ]

    # Table-top surface: a 0.5m x 0.5m square lying in a horizontal plane, placed
    # so the object rests on it in the reference frame — the "up" axis is the one
    # normal to that plane, and the table sits at the object's lowest coordinate
    # along it, centered on the object's centroid in the other two axes. The plane
    # is per-dataset (``cfg.table_plane``): "xoy" (up=z, default) for GRAB/ARCTIC,
    # "xoz" (up=y) for GigaHands. A sample may override it with its own
    # ``table_plane`` (HOT3D: Aria recordings are z-up, Quest3 y-up). Shipped as a
    # static (non-animated) quad.
    #
    # The reference frame is the first frame where the object is actually tracked
    # (``table_frame`` from the sample; defaults to 0). GigaHands leaves untracked
    # frames at the origin, so a blind frame-0 would drop the table at (0,0,0) —
    # away from the real scene and penetrating it.
    table_frame = int(sample.get("table_frame", 0) or 0)
    table_frame = max(0, min(table_frame, obj_verts.shape[0] - 1))
    ov0 = obj_verts[table_frame]                           # (Vo, 3)
    # up_axis is normal to the table plane; the other two axes span the square.
    table_plane = sample.get("table_plane") or STATE["cfg"].get("table_plane", "xoy")
    up_axis = 1 if str(table_plane).lower() == "xoz" else 2
    a0, a1 = [ax for ax in (0, 1, 2) if ax != up_axis]     # the two spanning axes
    up_val = float(ov0[:, up_axis].min())
    c0, c1 = float(ov0[:, a0].mean()), float(ov0[:, a1].mean())
    h = TABLE_SIZE / 2.0
    table_verts = np.zeros((4, 3), dtype=np.float32)
    table_verts[:, up_axis] = up_val
    for k, (s0, s1) in enumerate([(-h, -h), (h, -h), (h, h), (-h, h)]):
        table_verts[k, a0] = c0 + s0
        table_verts[k, a1] = c1 + s1
    table_faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    _add("table_verts", table_verts, np.float32)
    _add("table_faces", table_faces, np.int32)

    meta = {
        "nframes": nframes,
        "hand_sides": hand_sides,
        "obj_name": str(sample["obj_name"]),
        "action_name": str(sample["action_name"]),
        "subject": str(sample.get("subject", "")),
        "hand_colors": {s: HAND_SIDE_COLORS[s] for s in hand_sides},
        "obj_color": OBJECT_COLOR,
        # Optional main-object opacity; None lets the client pick (0.5 when
        # extras are overlaid, else opaque).
        "obj_opacity": sample.get("obj_opacity"),
        "has_contact": bool(has_contact),
        "label_colors": LABEL_COLORS,
        "labels": labels,
        "grasps": grasps,
        "segments": segments,
        "segment_colors": SEGMENT_COLORS,
        "table_color": TABLE_COLOR,
        "extras": [
            {"name": ex["name"], "color": list(ex["color"]),
             "opacity": float(ex.get("opacity", 0.5))}
            for ex in extras
        ],
        "fps": STATE["fps"],
        "arrays": manifest,
    }
    blob = b"".join(p.tobytes() for p in parts)
    return meta, blob


def _get_sequence_payload(index):
    """Return (meta, blob) for a sequence, posing + caching on first request."""
    if index in STATE["cache"]:
        return STATE["cache"][index]
    if STATE["segmented"]:
        sample = pose_segmented_sample(
            STATE["cfg"], STATE["dataset_name"], STATE["ds"],
            STATE["lhand_layer"], STATE["rhand_layer"], index,
        )
    elif STATE["stable_grasp"]:
        sample = pose_bimanual_stable_grasp_sample(
            STATE["cfg"], STATE["ds"],
            STATE["lhand_layer"], STATE["rhand_layer"], index,
        )
    elif STATE["wrist_aligned"]:
        sample = pose_wrist_aligned_sample(
            STATE["cfg"], STATE["ds"],
            STATE["lhand_layer"], STATE["rhand_layer"], index,
        )
    elif STATE["bimanual"]:
        sample = pose_bimanual_transition_sample(
            STATE["cfg"], STATE["ds"],
            STATE["lhand_layer"], STATE["rhand_layer"], index,
        )
    elif STATE["transition"]:
        sample = pose_transition_sample(
            STATE["cfg"], STATE["ds"],
            STATE["lhand_layer"], STATE["rhand_layer"], index,
        )
    elif STATE["dataset_name"] == "gigahands":
        sample = pose_gigahands_sample(
            STATE["cfg"], STATE["ds"],
            STATE["lhand_layer"], STATE["rhand_layer"], index,
        )
    elif STATE["dataset_name"] == "hot3d":
        sample = pose_hot3d_sample(
            STATE["cfg"], STATE["ds"],
            STATE["lhand_layer"], STATE["rhand_layer"], index,
        )
    else:
        sample = pose_sample(
            STATE["cfg"], STATE["dataset_name"], STATE["ds"],
            STATE["lhand_layer"], STATE["rhand_layer"], index,
        )
    payload = _serialize_sample(sample)
    if STATE["label_mode"]:
        # The editable working-copy file is authoritative in label mode; the baked-in
        # data.npz labels are just the initial seed (and match on first load, since the
        # working copy was seeded from the same text_root).
        meta, blob = payload
        meta = {**meta, "labels": _read_labels(index)}
        payload = (meta, blob)
    STATE["cache"][index] = payload
    return payload


@app.route("/")
def index():
    return render_template(
        "index.html",
        sequences=STATE["sequences"],
        dataset_name=STATE["dataset_name"],
        split=STATE["split"],
        fps=STATE["fps"],
        label_mode=STATE["label_mode"],
    )


@app.route("/api/sequences")
def api_sequences():
    return jsonify(STATE["sequences"])


@app.route("/api/labels/<int:index>", methods=["GET"])
def api_get_labels(index):
    if not STATE["label_mode"]:
        return jsonify({"error": "label mode is off"}), 403
    if index < 0 or index >= len(STATE["seq_meta"]):
        return jsonify({"error": "index out of range"}), 404
    return jsonify({"labels": _read_labels(index)})


@app.route("/api/labels/<int:index>", methods=["POST"])
def api_save_labels(index):
    """Full-replacement save of a sequence's action labels → working-dir description.txt.

    Body: ``{"labels": [{start, end, hand, action}, ...]}`` (0-indexed ``[start, end)``).
    Rewrites the file, invalidates the geometry cache so the next load re-reads it, and
    returns the canonical re-parsed labels so the client and file never diverge.
    """
    if not STATE["label_mode"]:
        return jsonify({"error": "label mode is off"}), 403
    if index < 0 or index >= len(STATE["seq_meta"]):
        return jsonify({"error": "index out of range"}), 404
    nframes = STATE["seq_meta"][index]["nframes"]
    body = request.get_json(force=True, silent=True) or {}
    clean = []
    for l in body.get("labels", []):
        try:
            s, e = int(l["start"]), int(l["end"])
        except (KeyError, TypeError, ValueError):
            return jsonify({"error": "start/end must be integers"}), 400
        hand = str(l.get("hand", "")).lower()
        action = str(l.get("action", "")).strip()
        if hand not in ("left", "right", "both"):
            return jsonify({"error": f"bad hand: {hand!r}"}), 400
        if not action:
            return jsonify({"error": "action must be non-empty"}), 400
        if not (0 <= s < e <= nframes):
            return jsonify({
                "error": f"require 0<=start<end<=nframes({nframes}); got [{s},{e})"
            }), 400
        clean.append({"start": s, "end": e, "hand": hand, "action": action})
    _write_description(_description_path(index), clean)
    STATE["cache"].pop(index, None)   # force re-read on next geometry load
    return jsonify({"ok": True, "labels": _read_labels(index)})


@app.route("/api/sequence/<int:index>")
def api_sequence(index):
    if index < 0 or index >= len(STATE["sequences"]):
        return jsonify({"error": "index out of range"}), 404
    meta, blob = _get_sequence_payload(index)
    resp = Response(blob, mimetype="application/octet-stream")
    resp.headers["X-Meta"] = json.dumps(meta, separators=(",", ":"))
    resp.headers["Access-Control-Expose-Headers"] = "X-Meta"
    return resp


def main():
    parser = argparse.ArgumentParser(description="Web viewer for HOI sequences.")
    parser.add_argument("--config", required=True, help="Dataset config yaml, e.g. configs/arctic.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--transition", action="store_true",
                        help="View GRAB transition motions (TransitionMotionGRAB) "
                             "instead of full sequences.")
    parser.add_argument("--bimanual", action="store_true",
                        help="View GRAB bimanual transition motions "
                             "(BimanualTransitionMotionGRAB): the two hands merged "
                             "per sequence into 0<->1<->2 grasp-count transitions "
                             "(types 01/02/10/12/20/21). GRAB config only.")
    parser.add_argument("--aug-p", type=float, default=None,
                        help="Override transition random-trim augmentation "
                             "probability (default: config aug_p; 0 = raw windows). "
                             "Only used with --transition / --bimanual.")
    parser.add_argument("--wrist-aligned", action="store_true",
                        help="Draw stable-grasp actions in the wrist frame: hand "
                             "fixed at the origin, with the true object motion and "
                             "the fixed-axis reconstruction overlaid in the same "
                             "space (one item per stable grasp).")
    parser.add_argument("--stable-grasp", action="store_true",
                        help="View GRAB bimanual stable-grasp segments "
                             "(BimanualStableGraspGRAB) in the WORLD frame: hands "
                             "and object shown as-is per segment, no wrist "
                             "alignment and no 1-DOF fit. GRAB config only.")
    parser.add_argument("--segmented", action="store_true",
                        help="View full GRAB sequences with an action-split track "
                             "at the bottom: null grasp (gray) / transition action "
                             "(+-5 frames around each grasp-count change) / stable "
                             "grasp. GRAB config only.")
    parser.add_argument("--label", action="store_true",
                        help="Enable action-label editing (ARCTIC full_motion only). "
                             "Edits persist to data/{dataset}_description/ and sync "
                             "live to the viewer; the raw descriptions are never touched.")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    dataset_name = cfg.name

    view_modes = [args.transition, args.bimanual, args.wrist_aligned,
                  args.stable_grasp, args.segmented]
    if sum(bool(m) for m in view_modes) > 1:
        parser.error("--transition / --bimanual / --wrist-aligned / --stable-grasp "
                     "/ --segmented are mutually exclusive.")

    if args.label:
        if dataset_name != "arctic" or not cfg.get("full_motion", False):
            parser.error("--label is only supported for the ARCTIC full_motion config.")
        if any(view_modes):
            parser.error("--label cannot be combined with --transition / "
                         "--bimanual / --wrist-aligned / --stable-grasp / --segmented.")

    if args.segmented:
        if dataset_name != "grab":
            parser.error("--segmented is only supported for the GRAB config.")
        print(f"Building GRAB dataset (split={args.split}) for the "
              f"segmented action view ...")
        ds, lhand_layer, rhand_layer = build_dataset(cfg, dataset_name, args.split)
    elif args.stable_grasp:
        if dataset_name != "grab":
            parser.error("--stable-grasp is only supported for the GRAB config.")
        print(f"Building GRAB bimanual stable-grasp dataset (split={args.split}) "
              f"for the world-frame view ...")
        ds, lhand_layer, rhand_layer = build_bimanual_stable_grasp_dataset(
            cfg, args.split)
    elif args.wrist_aligned:
        print(f"Building stable-grasp dataset ({dataset_name}, "
              f"split={args.split}) for the wrist-aligned view ...")
        ds, lhand_layer, rhand_layer = build_stable_grasp_dataset(
            cfg, dataset_name, args.split)
    elif args.bimanual:
        if dataset_name != "grab":
            parser.error("--bimanual is only supported for the GRAB config.")
        aug_p = args.aug_p if args.aug_p is not None else cfg.get("aug_p", 0.0)
        print(f"Building GRAB bimanual transition dataset (split={args.split}, "
              f"aug_p={aug_p}) ...")
        ds, lhand_layer, rhand_layer = build_bimanual_transition_dataset(
            cfg, args.split, aug_p=aug_p)
    elif args.transition:
        if dataset_name != "grab":
            parser.error("--transition is only supported for the GRAB config.")
        aug_p = args.aug_p if args.aug_p is not None else cfg.get("aug_p", 0.0)
        print(f"Building GRAB transition dataset (split={args.split}, "
              f"aug_p={aug_p}) ...")
        ds, lhand_layer, rhand_layer = build_transition_dataset(
            cfg, args.split, aug_p=aug_p)
    elif dataset_name in ("gigahands", "hot3d"):
        pretty = {"gigahands": "GigaHands", "hot3d": "HOT3D"}[dataset_name]
        if any(view_modes):
            parser.error("--transition / --bimanual / --wrist-aligned / "
                         "--stable-grasp / --segmented are not supported for the "
                         f"{pretty} config; use the plain world-frame view.")
        print(f"Building {pretty} dataset (split={args.split}) ...")
        build = (build_gigahands_dataset if dataset_name == "gigahands"
                 else build_hot3d_dataset)
        ds, lhand_layer, rhand_layer = build(cfg, args.split)
    else:
        print(f"Building dataset ({dataset_name}, split={args.split}) ...")
        ds, lhand_layer, rhand_layer = build_dataset(cfg, dataset_name, args.split)
    if dataset_name == "gigahands":
        sequences = _build_gigahands_sequence_list(ds)
    elif dataset_name == "hot3d":
        sequences = _build_hot3d_sequence_list(ds)
    else:
        sequences = _build_sequence_list(
            ds, transition=args.transition or args.bimanual,
            wrist_aligned=args.wrist_aligned, stable_grasp=args.stable_grasp)

    label_dir = _ensure_label_dir(cfg, dataset_name) if args.label else None
    seq_meta = _build_seq_meta(ds) if args.label else []

    STATE.update({
        "cfg": cfg,
        "dataset_name": dataset_name,
        "split": args.split,
        "ds": ds,
        "lhand_layer": lhand_layer,
        "rhand_layer": rhand_layer,
        "sequences": sequences,
        "cache": {},
        "fps": args.fps,
        "transition": args.transition,
        "bimanual": args.bimanual,
        "wrist_aligned": args.wrist_aligned,
        "stable_grasp": args.stable_grasp,
        "segmented": args.segmented,
        "label_mode": args.label,
        "label_dir": label_dir,
        "seq_meta": seq_meta,
    })

    kind = ("segmented sequences" if args.segmented
            else "bimanual stable grasps" if args.stable_grasp
            else "stable grasps" if args.wrist_aligned
            else "bimanual transitions" if args.bimanual
            else "transitions" if args.transition else "sequences")
    print(f"Loaded {len(sequences)} {kind}. Serving at "
          f"http://{args.host}:{args.port}/  ({kind} posed lazily on first view)")
    if args.label:
        print(f"[label] editing enabled — edits saved to {label_dir}")
    # threaded=False keeps GPU posing serialized (one CUDA context, no reentrancy).
    app.run(host=args.host, port=args.port, threaded=False, debug=False)


if __name__ == "__main__":
    main()
