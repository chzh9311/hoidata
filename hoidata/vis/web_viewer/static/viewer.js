import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ---------------------------------------------------------------- DOM handles
const viewport = document.getElementById('viewport');
const statusEl = document.getElementById('status');
const seqSelect = document.getElementById('seqSelect');
const prevBtn = document.getElementById('prevBtn');
const nextBtn = document.getElementById('nextBtn');
const seqLabel = document.getElementById('seqLabel');
const playBtn = document.getElementById('playBtn');
const frameReadout = document.getElementById('frameReadout');
const scrub = document.getElementById('scrub');
const segAction = document.getElementById('segAction');
const segLeft = document.getElementById('segLeft');
const segRight = document.getElementById('segRight');
const playheadLine = document.querySelector('#playhead .line');
const activeLabelEl = document.getElementById('activeLabel');

// Stable-grasp block colors (match the legend + hand-side convention).
const GRASP_COLORS = { left: [0.20, 0.45, 0.95], right: [0.90, 0.25, 0.25] };

const FPS = window.__FPS__ || 30;
const LABEL_MODE = !!window.__LABEL_MODE__;

// ---- Label-editor DOM (present only in --label mode) ----
const editor = document.getElementById('editor');
const edNew = document.getElementById('edNew');
const edStartInput = document.getElementById('edStart');
const edEndInput = document.getElementById('edEnd');
const edSetIn = document.getElementById('edSetIn');
const edSetOut = document.getElementById('edSetOut');
const edAction = document.getElementById('edAction');
const edHand = document.getElementById('edHand');
const edApply = document.getElementById('edApply');
const edDelete = document.getElementById('edDelete');
const edSave = document.getElementById('edSave');
const edStatus = document.getElementById('edStatus');

// ---------------------------------------------------------------- Three setup
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1b1d23);

const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
camera.position.set(0, 0.4, 1.2);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
viewport.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

scene.add(new THREE.HemisphereLight(0xffffff, 0x404050, 1.1));
const dir = new THREE.DirectionalLight(0xffffff, 1.4);
dir.position.set(1, 2, 1.5);
scene.add(dir);
const dir2 = new THREE.DirectionalLight(0xffffff, 0.6);
dir2.position.set(-1, -0.5, -1);
scene.add(dir2);

function resize() {
  const w = viewport.clientWidth, h = viewport.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);

// ---------------------------------------------------------------- scene state
// One entry per rendered mesh: { mesh, positions (Float32Array T*V*3), V,
//   baseColor (THREE.Color) }.
let meshes = [];
let nframes = 0;
let frameIdx = 0;
let playing = true;
let lastTick = 0;
let loading = false;

// Annotation state for the loaded sequence.
let labels = [];              // [{start, end, hand, action}]
let labelColors = {};          // {left|right|both -> [r,g,b] 0..1}
let grasps = { left: [], right: [] };  // stable-grasp intervals per side
let activeLabelIdx = -1;       // which label is highlighted right now (-1 none)
let segEls = [];               // DOM nodes for each action label's colored block

// Label-editing state (label mode only).
let curIndex = 0;              // index of the loaded sequence (for the save POST)
let editIdx = -1;              // label being edited (-1 = new / unselected)
let edStart = null;            // pending in-point (0-indexed frame) or null
let edEnd = null;              // pending out-point (exclusive frame) or null
let selBandEl = null;          // translucent band on the action lane (in/out preview)

function clearMeshes() {
  for (const m of meshes) {
    scene.remove(m.mesh);
    m.mesh.geometry.dispose();
    m.mesh.material.dispose();
  }
  meshes = [];
}

function makeMesh(positionsFrame0, faces, colorArr, V, opacity = 1.0, vertexColors = false) {
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(positionsFrame0.slice(), 3));
  if (vertexColors) {
    // Per-vertex color attribute (filled per frame from the contact heatmap).
    geom.setAttribute('color', new THREE.BufferAttribute(new Float32Array(V * 3), 3));
  }
  geom.setIndex(new THREE.BufferAttribute(faces, 1));
  geom.computeVertexNormals();
  // With vertexColors the material color is a global multiplier — keep it white
  // so the per-vertex heatmap shows untinted; otherwise use the flat base color.
  const color = vertexColors
    ? new THREE.Color(1, 1, 1)
    : new THREE.Color(colorArr[0], colorArr[1], colorArr[2]);
  const transparent = opacity < 1.0;
  const material = new THREE.MeshStandardMaterial({
    color, roughness: 0.75, metalness: 0.0, side: THREE.DoubleSide,
    transparent, opacity, vertexColors,
    // Don't write depth for translucent meshes so two overlapping objects
    // (true vs. reconstructed) blend instead of one hiding the other.
    depthWrite: !transparent,
  });
  const mesh = new THREE.Mesh(geom, material);
  scene.add(mesh);
  return mesh;
}

// Two-hand contact heatmap. Each hand has its own hot color; a vertex is colored
// by whichever hand's contact value is larger (ties → left), blending from the
// object's base color (no contact) toward that hand's hot color with the
// winning value. Writes into out[o..o+2] (0..1 floats).
const CONTACT_HOT = {
  left:  new THREE.Color(0.20, 0.55, 1.00),   // blue  → left-hand contact
  right: new THREE.Color(1.00, 0.30, 0.15),   // red   → right-hand contact
};
const _contactTmp = new THREE.Color();
function writeContactColor(out, o, vLeft, vRight, baseColor) {
  const l = Math.max(0, Math.min(1, vLeft));
  const r = Math.max(0, Math.min(1, vRight));
  const useLeft = l >= r;                       // ties favor left
  const c = useLeft ? l : r;
  _contactTmp.copy(baseColor).lerp(useLeft ? CONTACT_HOT.left : CONTACT_HOT.right, c);
  out[o] = _contactTmp.r; out[o + 1] = _contactTmp.g; out[o + 2] = _contactTmp.b;
}

// ---------------------------------------------------------------- decode blob
// meta.arrays[name] = { dtype:'float32'|'int32', offset, shape }
function viewArray(buffer, entry) {
  const count = entry.shape.reduce((a, b) => a * b, 1);
  if (entry.dtype === 'float32') return new Float32Array(buffer, entry.offset, count);
  return new Int32Array(buffer, entry.offset, count);
}

function buildScene(meta, buffer) {
  clearMeshes();
  nframes = meta.nframes;
  labels = meta.labels || [];
  labelColors = meta.label_colors || {};
  grasps = meta.grasps || { left: [], right: [] };
  activeLabelIdx = -1;

  const arrays = meta.arrays;
  // When extra objects are overlaid in the same space (wrist-aligned view:
  // true vs. reconstructed), draw the objects semitransparent so both read.
  const hasExtras = (meta.extras || []).length > 0;
  const objOpacity = hasExtras ? 0.5 : 1.0;
  const objFaces = new Uint32Array(viewArray(buffer, arrays.obj_faces)); // copy to Uint32
  const objVertsEntry = arrays.obj_verts;                                // (T, Vo, 3)
  const Vo = objVertsEntry.shape[1];
  const objVerts = viewArray(buffer, objVertsEntry);
  // Optional per-frame, per-hand contact heatmap (T, 2, Vo) = [left, right].
  const hasContact = !!meta.has_contact && !!arrays.obj_contact;
  const objContact = hasContact ? viewArray(buffer, arrays.obj_contact) : null;
  meshes.push({
    mesh: makeMesh(objVerts.subarray(0, Vo * 3), objFaces, meta.obj_color, Vo,
                   objOpacity, hasContact),
    positions: objVerts, V: Vo,
    baseColor: new THREE.Color(...meta.obj_color),
    contact: objContact,          // (T*2*Vo) flat [left,right] per frame, or null
  });

  for (const side of meta.hand_sides) {
    const facesEntry = arrays['hand_faces_' + side];
    const vertsEntry = arrays['hand_verts_' + side];       // (T, 778, 3)
    const V = vertsEntry.shape[1];
    const faces = new Uint32Array(viewArray(buffer, facesEntry));
    const verts = viewArray(buffer, vertsEntry);
    const col = meta.hand_colors[side];
    meshes.push({
      mesh: makeMesh(verts.subarray(0, V * 3), faces, col, V),
      positions: verts, V,
      baseColor: new THREE.Color(...col),
    });
  }

  // Extra meshes (e.g. reconstructed object) — same space, own color.
  for (const ex of (meta.extras || [])) {
    const facesEntry = arrays['extra_faces_' + ex.name];
    const vertsEntry = arrays['extra_verts_' + ex.name];
    if (!facesEntry || !vertsEntry) continue;
    const V = vertsEntry.shape[1];
    const faces = new Uint32Array(viewArray(buffer, facesEntry));
    const verts = viewArray(buffer, vertsEntry);
    const exOpacity = (ex.opacity != null) ? ex.opacity : 0.5;
    meshes.push({
      mesh: makeMesh(verts.subarray(0, V * 3), faces, ex.color, V, exOpacity),
      positions: verts, V,
      baseColor: new THREE.Color(...ex.color),
    });
  }

  renderTracks();
  if (LABEL_MODE) { rebindActionSegClicks(); edClear(); }
  frameSceneToBBox();
  setFrame(0);
}

function rgbCss(c) {
  return `rgb(${(c[0] * 255) | 0},${(c[1] * 255) | 0},${(c[2] * 255) | 0})`;
}

// Build one colored block for an interval within a track lane, positioned by
// frame fraction over the full sequence. Returns the DOM node.
function makeSegment(start, end, color, text, tooltip) {
  const seg = document.createElement('div');
  seg.className = 'seg';
  seg.style.left = (start / nframes) * 100 + '%';
  seg.style.width = Math.max(((end - start) / nframes) * 100, 0.4) + '%';
  seg.style.background = rgbCss(color);
  if (tooltip) seg.title = tooltip;
  if (text) {
    const txt = document.createElement('span');
    txt.className = 'segtext';
    txt.textContent = text;
    seg.appendChild(txt);
  }
  return seg;
}

// Fill all three tracks: action labels (with text), left-hand stable grasps,
// and right-hand stable grasps (colored blocks, no text).
function renderTracks() {
  segAction.innerHTML = '';
  segLeft.innerHTML = '';
  segRight.innerHTML = '';
  segEls = [];
  if (nframes <= 0) return;

  // Top track: action labels (kept as the highlightable segEls for meshes).
  for (const lab of labels) {
    const c = labelColors[lab.hand] || [0.6, 0.6, 0.6];
    const seg = makeSegment(lab.start, lab.end, c, lab.action || '',
      `${lab.start}-${lab.end}  ${lab.action} (${lab.hand})`);
    segAction.appendChild(seg);
    segEls.push(seg);
  }

  // Middle / bottom tracks: stable grasps per side (no text).
  for (const [side, container] of [['left', segLeft], ['right', segRight]]) {
    const c = GRASP_COLORS[side];
    for (const [s, e] of (grasps[side] || [])) {
      container.appendChild(
        makeSegment(s, e, c, '', `${side} grasp  ${s}-${e}`));
    }
  }
}

// ---------------------------------------------------------------- label editor
// All editing behavior is gated behind LABEL_MODE; in plain view mode these are
// inert (the editor panel stays hidden and the hooks below are never called).

// Mark each action block as editable (cursor + hit-testing). Selection itself is
// handled in the #scrub pointerdown handler, which owns pointer capture and must
// intercept the press before it becomes a scrub (a per-segment click never fires
// once #scrub captures the pointer). segEls[i] corresponds to labels[i].
function rebindActionSegClicks() {
  segEls.forEach((el) => el.classList.add('editable'));
}

// Translucent band on the action lane showing the pending [in, out) range.
function renderSelectionBand() {
  if (!selBandEl) return;
  if (edStart == null || edEnd == null || edEnd <= edStart || nframes <= 0) {
    selBandEl.style.display = 'none';
    return;
  }
  selBandEl.style.display = 'block';
  selBandEl.style.left = (edStart / nframes) * 100 + '%';
  selBandEl.style.width = ((edEnd - edStart) / nframes) * 100 + '%';
}

// Reset the editor to a fresh (unselected) interval.
function edClear() {
  editIdx = -1; edStart = null; edEnd = null;
  edStartInput.value = ''; edEndInput.value = '';
  edAction.value = ''; edHand.value = 'both';
  renderSelectionBand();
}

// Populate the editor from an existing label.
function edLoadFrom(i) {
  const l = labels[i];
  if (!l) return;
  editIdx = i; edStart = l.start; edEnd = l.end;
  edStartInput.value = l.start; edEndInput.value = l.end;
  edAction.value = l.action || ''; edHand.value = l.hand;
  edStatus.textContent = `editing #${i}`;
  renderSelectionBand();
}

// Read the current In/Out numeric inputs into edStart/edEnd (null if blank).
function edSyncFromInputs() {
  const s = parseInt(edStartInput.value, 10);
  const e = parseInt(edEndInput.value, 10);
  edStart = Number.isInteger(s) ? s : null;
  edEnd = Number.isInteger(e) ? e : null;
  renderSelectionBand();
}

function wireEditor() {
  editor.hidden = false;

  // Selection band lives inside the action lane (parent of #segAction).
  selBandEl = document.createElement('div');
  selBandEl.id = 'selBand';
  selBandEl.style.display = 'none';
  segAction.parentElement.appendChild(selBandEl);

  edNew.onclick = () => { edClear(); edStatus.textContent = 'new interval'; };

  edSetIn.onclick = () => {
    edStart = frameIdx; edStartInput.value = frameIdx; renderSelectionBand();
  };
  // Out-point is exclusive: the playhead frame is the first EXCLUDED frame, so
  // the interval is [in, playhead). This matches the file's [start, end) form.
  edSetOut.onclick = () => {
    edEnd = frameIdx; edEndInput.value = frameIdx; renderSelectionBand();
  };
  edStartInput.addEventListener('change', edSyncFromInputs);
  edEndInput.addEventListener('change', edSyncFromInputs);

  // Add or update the interval in the local labels array (staged, not yet saved).
  edApply.onclick = () => {
    edSyncFromInputs();
    const s = edStart, e = edEnd;
    const hand = edHand.value, action = edAction.value.trim();
    if (!(Number.isInteger(s) && Number.isInteger(e) && 0 <= s && s < e && e <= nframes)) {
      edStatus.textContent = `bad range (need 0<=in<out<=${nframes})`;
      return;
    }
    if (!action) { edStatus.textContent = 'action required'; return; }
    const rec = { start: s, end: e, hand, action };
    if (editIdx >= 0) labels[editIdx] = rec; else labels.push(rec);
    labels.sort((a, b) => (a.start - b.start) || (a.end - b.end));
    renderTracks(); rebindActionSegClicks(); applyLabelHighlight(frameIdx);
    edStatus.textContent = 'staged (unsaved)';
    edClear();
  };

  edDelete.onclick = () => {
    if (editIdx < 0) { edStatus.textContent = 'nothing selected'; return; }
    labels.splice(editIdx, 1);
    renderTracks(); rebindActionSegClicks(); applyLabelHighlight(frameIdx);
    edClear();
    edStatus.textContent = 'deleted (unsaved)';
  };

  // Save the whole track: full-replacement POST; adopt the canonical re-parse.
  edSave.onclick = async () => {
    edStatus.textContent = 'saving…';
    try {
      const resp = await fetch(`/api/labels/${curIndex}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ labels }),
      });
      const j = await resp.json();
      if (!resp.ok || !j.ok) throw new Error(j.error || `HTTP ${resp.status}`);
      labels = j.labels;
      renderTracks(); rebindActionSegClicks(); applyLabelHighlight(frameIdx);
      edStatus.textContent = 'saved ✓';
    } catch (err) {
      edStatus.textContent = 'save failed: ' + err.message;
      console.error(err);
    }
  };
}

// Move the playhead line to frame t's horizontal position within the lanes.
function updatePlayhead(t) {
  const frac = nframes > 1 ? t / (nframes - 1) : 0;
  playheadLine.style.left = (frac * 100) + '%';
}

// The label whose interval contains frame t (last one wins on overlap), or -1.
function labelAtFrame(t) {
  let found = -1;
  for (let i = 0; i < labels.length; i++) {
    if (t >= labels[i].start && t < labels[i].end) found = i;
  }
  return found;
}

// Recolor meshes to signal "this frame is annotated": when inside a label the
// meshes take the label's hand-type color; outside, they revert to base colors.
function applyLabelHighlight(t) {
  const idx = labelAtFrame(t);
  if (idx === activeLabelIdx) return;   // no change → skip the material churn

  // Un-emphasise the previously active segment block.
  if (activeLabelIdx >= 0 && segEls[activeLabelIdx]) {
    segEls[activeLabelIdx].classList.remove('active');
  }
  activeLabelIdx = idx;

  if (idx < 0) {
    // Contact meshes use vertex colors; keep their material color white (a
    // neutral multiplier) so the per-vertex heatmap is shown untinted.
    for (const m of meshes) {
      if (m.contact) m.mesh.material.color.setRGB(1, 1, 1);
      else m.mesh.material.color.copy(m.baseColor);
    }
    activeLabelEl.textContent = '';
    return;
  }
  if (segEls[idx]) segEls[idx].classList.add('active');
  const lab = labels[idx];
  const c = labelColors[lab.hand] || [0.6, 0.6, 0.6];
  const hi = new THREE.Color(c[0], c[1], c[2]);
  for (const m of meshes) {
    if (m.contact) continue;   // leave the contact heatmap unmodified
    // Blend base color toward the label color so shapes stay readable.
    m.mesh.material.color.copy(m.baseColor).lerp(hi, 0.6);
  }
  activeLabelEl.textContent = `${lab.action || '—'} · ${lab.hand}`;
}

// Frame the camera to the full-sequence bounding box (all frames, all meshes),
// matching o3d render_video's framing so hands starting far from the object
// still fit.
function frameSceneToBBox() {
  let mn = [Infinity, Infinity, Infinity];
  let mx = [-Infinity, -Infinity, -Infinity];
  for (const m of meshes) {
    const p = m.positions;
    for (let i = 0; i < p.length; i += 3) {
      for (let k = 0; k < 3; k++) {
        const v = p[i + k];
        if (v < mn[k]) mn[k] = v;
        if (v > mx[k]) mx[k] = v;
      }
    }
  }
  const center = new THREE.Vector3(
    (mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, (mn[2] + mx[2]) / 2);
  const diag = Math.hypot(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]) || 0.4;
  const radius = Math.max(0.3, diag);

  controls.target.copy(center);
  camera.position.set(center.x, center.y + radius * 0.35, center.z + radius * 1.2);
  camera.near = Math.max(radius / 500, 0.001);
  camera.far = radius * 50;
  camera.updateProjectionMatrix();
  controls.update();
}

// Copy per-frame vertex slice into each geometry and refresh normals.
function setFrame(t) {
  frameIdx = t;
  for (const m of meshes) {
    const stride = m.V * 3;
    const src = m.positions.subarray(t * stride, (t + 1) * stride);
    const attr = m.mesh.geometry.getAttribute('position');
    attr.array.set(src);
    attr.needsUpdate = true;
    m.mesh.geometry.computeVertexNormals();
    // Refresh the two-hand contact heatmap for this frame, if present. The flat
    // contact array is (T, 2, Vo): per frame a left block then a right block.
    if (m.contact) {
      const cattr = m.mesh.geometry.getAttribute('color');
      const lBase = t * 2 * m.V;       // left  block for frame t
      const rBase = lBase + m.V;       // right block for frame t
      for (let i = 0; i < m.V; i++) {
        writeContactColor(cattr.array, i * 3,
          m.contact[lBase + i], m.contact[rBase + i], m.baseColor);
      }
      cattr.needsUpdate = true;
    }
  }
  applyLabelHighlight(t);
  updatePlayhead(t);
  frameReadout.textContent = `${t + 1} / ${nframes}`;
}

// ---------------------------------------------------------------- playback
function animate(now) {
  requestAnimationFrame(animate);
  controls.update();

  if (playing && nframes > 0 && !loading) {
    if (now - lastTick >= 1000 / FPS) {
      lastTick = now;
      setFrame((frameIdx + 1) % nframes);
    }
  }
  renderer.render(scene, camera);
}

function setPlaying(p) {
  playing = p;
  // YouTube-style glyphs: ⏸ while playing, ▶ (triangle) while paused.
  playBtn.textContent = p ? '⏸' : '▶';
  playBtn.title = p ? 'Pause (space)' : 'Play (space)';
}

// ---------------------------------------------------------------- data loading
async function loadSequence(index) {
  loading = true;
  curIndex = index;
  statusEl.textContent = 'loading sequence…';
  statusEl.classList.add('loading');
  try {
    const resp = await fetch(`/api/sequence/${index}`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const meta = JSON.parse(resp.headers.get('X-Meta'));
    const buffer = await resp.arrayBuffer();
    buildScene(meta, buffer);
    seqLabel.textContent = `${meta.obj_name} · ${meta.action_name} · ${meta.nframes}f`;
    statusEl.textContent = `${meta.obj_name} / ${meta.action_name}`;
    statusEl.classList.remove('loading');
    setPlaying(true);
    lastTick = performance.now();
  } catch (err) {
    statusEl.textContent = 'load failed: ' + err.message;
    console.error(err);
  } finally {
    loading = false;
  }
}

// ---------------------------------------------------------------- UI wiring
playBtn.addEventListener('click', () => setPlaying(!playing));

// Spacebar toggles play/pause (unless typing in the sequence dropdown, whose
// own keyboard behaviour we leave alone).
// Inputs that consume the spacebar for typing — don't toggle playback for them.
const spaceGuardEls = new Set([seqSelect]);
if (LABEL_MODE) {
  spaceGuardEls.add(edAction);
  spaceGuardEls.add(edStartInput);
  spaceGuardEls.add(edEndInput);
}
window.addEventListener('keydown', (e) => {
  if (e.code !== 'Space' && e.key !== ' ') return;
  if (spaceGuardEls.has(document.activeElement)) return;
  e.preventDefault();           // stop the page from scrolling
  if (nframes > 0) setPlaying(!playing);
});

// Scrubbing on the custom segmented bar: click or drag to seek. Playback pauses
// while dragging and resumes on release if it was playing.
let scrubbing = false;
let wasPlayingBeforeScrub = false;

function frameFromEvent(e) {
  // Measure against a lane's geometry (segAction fills the lane via inset:0),
  // NOT the whole #scrub box — #scrub includes the left track-name gutter, while
  // the segments and playhead live only within the lane. Using #scrub here would
  // offset the mapping by the gutter width and mis-track the pointer.
  const rect = segAction.getBoundingClientRect();
  let frac = (e.clientX - rect.left) / rect.width;
  frac = Math.max(0, Math.min(1, frac));
  return Math.round(frac * Math.max(nframes - 1, 0));
}

scrub.addEventListener('pointerdown', (e) => {
  if (nframes <= 0) return;
  // In label mode, a press that lands on an editable action block selects that
  // label for editing instead of scrubbing. #scrub owns pointerdown + pointer
  // capture, so a segment-level handler can't win on its own — we must bail here
  // (a plain .onclick would never fire once #scrub captures the pointer).
  if (LABEL_MODE) {
    const seg = e.target.closest('.seg.editable');
    if (seg && segEls.includes(seg)) {
      edLoadFrom(segEls.indexOf(seg));
      return;   // don't start scrubbing
    }
  }
  scrubbing = true;
  wasPlayingBeforeScrub = playing;
  setPlaying(false);
  scrub.setPointerCapture(e.pointerId);
  setFrame(frameFromEvent(e));
});
scrub.addEventListener('pointermove', (e) => {
  if (scrubbing) setFrame(frameFromEvent(e));
});
function endScrub(e) {
  if (!scrubbing) return;
  scrubbing = false;
  try { scrub.releasePointerCapture(e.pointerId); } catch (_) {}
  if (wasPlayingBeforeScrub) { setPlaying(true); lastTick = performance.now(); }
}
scrub.addEventListener('pointerup', endScrub);
scrub.addEventListener('pointercancel', endScrub);

function selectAndLoad(index) {
  const clamped = Math.max(0, Math.min(seqSelect.options.length - 1, index));
  seqSelect.value = String(clamped);
  loadSequence(clamped);
}

seqSelect.addEventListener('change', () => loadSequence(parseInt(seqSelect.value, 10)));
prevBtn.addEventListener('click', () => selectAndLoad(parseInt(seqSelect.value, 10) - 1));
nextBtn.addEventListener('click', () => selectAndLoad(parseInt(seqSelect.value, 10) + 1));

// ---------------------------------------------------------------- boot
resize();
if (LABEL_MODE) wireEditor();
requestAnimationFrame(animate);
if (seqSelect.options.length > 0) {
  loadSequence(parseInt(seqSelect.value, 10));
} else {
  statusEl.textContent = 'no sequences in this split';
}
