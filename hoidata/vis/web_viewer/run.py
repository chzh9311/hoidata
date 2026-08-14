"""Local web app for interactive HOI (hand-object interaction) visualization.

Serves a single-page Three.js viewer that plays a posed ARCTIC/GRAB sequence in
a loop with play/pause, a draggable frame progress bar, a sequence dropdown, and
prev/next buttons — with free mouse-drag camera orbit, matching the native
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
    build_stable_grasp_dataset, pose_wrist_aligned_sample)
from hoidata.preprocess.preprocess_arctic import parse_description_labels

HAND_SIDE_COLORS = {"left": [0.9, 0.75, 0.65], "right": [0.75, 0.85, 0.95]}
OBJECT_COLOR = [0.6, 0.7, 0.9]

# Progress-bar / active-label colors by annotated hand type (0..1 RGB).
LABEL_COLORS = {
    "left":  [0.20, 0.45, 0.95],   # blue
    "right": [0.90, 0.25, 0.25],   # red
    "both":  [0.25, 0.75, 0.35],   # green
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


def _build_sequence_list(ds, transition=False, wrist_aligned=False):
    """Cheap per-index metadata for the dropdown — no MANO/GPU posing.

    Uses the ``_object_name`` hook (uniform across GRAB/ARCTIC) rather than a
    dataset-specific attribute: ARCTIC stores ``object_name`` while GRAB stores
    ``obj_name``, but both expose ``_object_name(index)``.

    For the transition dataset each index is a *transition* (not a sequence), so
    per-transition helpers (``transition_label`` / ``transition_nframes``) supply
    the label and length.  For the wrist-aligned view each index is a *stable
    grasp* (one CSV row): the length comes from the flat per-grasp index arrays.
    """
    seqs = []
    for i in range(len(ds)):
        if wrist_aligned:
            sg = ds.stable_grasps
            side = sg["hand_side"][i]
            nframes = int(sg["end_frame"][i]) - int(sg["start_frame"][i])
            seqs.append({
                "index": i,
                "obj_name": str(ds._object_name(i)),
                "action_name": f"{ds._object_name(i)} {side} grasp #{i}",
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


def _serialize_sample(sample):
    """Pack a posed sample into (meta_dict, blob_bytes).

    The blob is a single little-endian byte string holding, in order:
      - object faces         (int32,   (Fo, 3))
      - per-side hand faces   (int32,   (1538, 3)) for each active side
      - object verts          (float32, (T, Vo, 3))
      - per-side hand verts    (float32, (T, 778, 3)) for each active side
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

    meta = {
        "nframes": nframes,
        "hand_sides": hand_sides,
        "obj_name": str(sample["obj_name"]),
        "action_name": str(sample["action_name"]),
        "subject": str(sample.get("subject", "")),
        "hand_colors": {s: HAND_SIDE_COLORS[s] for s in hand_sides},
        "obj_color": OBJECT_COLOR,
        "has_contact": bool(has_contact),
        "label_colors": LABEL_COLORS,
        "labels": labels,
        "grasps": grasps,
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
    if STATE["wrist_aligned"]:
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
    parser.add_argument("--label", action="store_true",
                        help="Enable action-label editing (ARCTIC full_motion only). "
                             "Edits persist to data/{dataset}_description/ and sync "
                             "live to the viewer; the raw descriptions are never touched.")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    dataset_name = cfg.name

    if args.transition and args.bimanual:
        parser.error("--transition and --bimanual are mutually exclusive.")

    if args.label:
        if dataset_name != "arctic" or not cfg.get("full_motion", False):
            parser.error("--label is only supported for the ARCTIC full_motion config.")
        if args.transition or args.bimanual or args.wrist_aligned:
            parser.error("--label cannot be combined with "
                         "--transition / --bimanual / --wrist-aligned.")

    if args.wrist_aligned:
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
    else:
        print(f"Building dataset ({dataset_name}, split={args.split}) ...")
        ds, lhand_layer, rhand_layer = build_dataset(cfg, dataset_name, args.split)
    sequences = _build_sequence_list(
        ds, transition=args.transition or args.bimanual,
        wrist_aligned=args.wrist_aligned)

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
        "label_mode": args.label,
        "label_dir": label_dir,
        "seq_meta": seq_meta,
    })

    kind = ("stable grasps" if args.wrist_aligned
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
