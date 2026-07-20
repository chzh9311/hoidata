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

import numpy as np
from flask import Flask, Response, jsonify, render_template
from omegaconf import OmegaConf

from hoidata.vis.o3d_visualizer import (
    build_dataset, pose_sample, build_transition_dataset, pose_transition_sample)

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
}


def _build_sequence_list(ds, transition=False):
    """Cheap per-index metadata for the dropdown — no MANO/GPU posing.

    Uses the ``_object_name`` hook (uniform across GRAB/ARCTIC) rather than a
    dataset-specific attribute: ARCTIC stores ``object_name`` while GRAB stores
    ``obj_name``, but both expose ``_object_name(index)``.

    For the transition dataset each index is a *transition* (not a sequence), so
    per-transition helpers (``transition_label`` / ``transition_nframes``) supply
    the label and length.
    """
    seqs = []
    for i in range(len(ds)):
        if transition:
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

    _add("obj_verts", obj_verts, np.float32)
    for i, side in enumerate(hand_sides):
        _add(f"hand_verts_{side}", hand_verts[:, i * 778:(i + 1) * 778], np.float32)

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
        "label_colors": LABEL_COLORS,
        "labels": labels,
        "grasps": grasps,
        "fps": STATE["fps"],
        "arrays": manifest,
    }
    blob = b"".join(p.tobytes() for p in parts)
    return meta, blob


def _get_sequence_payload(index):
    """Return (meta, blob) for a sequence, posing + caching on first request."""
    if index in STATE["cache"]:
        return STATE["cache"][index]
    if STATE["transition"]:
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
    )


@app.route("/api/sequences")
def api_sequences():
    return jsonify(STATE["sequences"])


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
    parser.add_argument("--aug-p", type=float, default=None,
                        help="Override transition random-trim augmentation "
                             "probability (default: config aug_p; 0 = raw windows). "
                             "Only used with --transition.")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    dataset_name = cfg.name

    if args.transition:
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
    sequences = _build_sequence_list(ds, transition=args.transition)

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
    })

    kind = "transitions" if args.transition else "sequences"
    print(f"Loaded {len(sequences)} {kind}. Serving at "
          f"http://{args.host}:{args.port}/  ({kind} posed lazily on first view)")
    # threaded=False keeps GPU posing serialized (one CUDA context, no reentrancy).
    app.run(host=args.host, port=args.port, threaded=False, debug=False)


if __name__ == "__main__":
    main()
