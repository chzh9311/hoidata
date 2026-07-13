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
const progress = document.getElementById('progress');
const frameReadout = document.getElementById('frameReadout');

const FPS = window.__FPS__ || 30;

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
// One entry per rendered mesh: { mesh, positions (Float32Array T*V*3), V }.
let meshes = [];
let nframes = 0;
let frameIdx = 0;
let playing = true;
let lastTick = 0;
let loading = false;

function clearMeshes() {
  for (const m of meshes) {
    scene.remove(m.mesh);
    m.mesh.geometry.dispose();
    m.mesh.material.dispose();
  }
  meshes = [];
}

function makeMesh(positionsFrame0, faces, colorArr, V) {
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(positionsFrame0.slice(), 3));
  geom.setIndex(new THREE.BufferAttribute(faces, 1));
  geom.computeVertexNormals();
  const color = new THREE.Color(colorArr[0], colorArr[1], colorArr[2]);
  const material = new THREE.MeshStandardMaterial({
    color, roughness: 0.75, metalness: 0.0, side: THREE.DoubleSide,
  });
  const mesh = new THREE.Mesh(geom, material);
  scene.add(mesh);
  return mesh;
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

  const arrays = meta.arrays;
  const objFaces = new Uint32Array(viewArray(buffer, arrays.obj_faces)); // copy to Uint32
  const objVertsEntry = arrays.obj_verts;                                // (T, Vo, 3)
  const Vo = objVertsEntry.shape[1];
  const objVerts = viewArray(buffer, objVertsEntry);
  meshes.push({
    mesh: makeMesh(objVerts.subarray(0, Vo * 3), objFaces, meta.obj_color, Vo),
    positions: objVerts, V: Vo,
  });

  for (const side of meta.hand_sides) {
    const facesEntry = arrays['hand_faces_' + side];
    const vertsEntry = arrays['hand_verts_' + side];       // (T, 778, 3)
    const V = vertsEntry.shape[1];
    const faces = new Uint32Array(viewArray(buffer, facesEntry));
    const verts = viewArray(buffer, vertsEntry);
    meshes.push({
      mesh: makeMesh(verts.subarray(0, V * 3), faces, meta.hand_colors[side], V),
      positions: verts, V,
    });
  }

  frameSceneToBBox();
  setFrame(0);
  progress.max = String(Math.max(nframes - 1, 0));
  progress.value = '0';
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
  }
  progress.value = String(t);
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
  playBtn.textContent = p ? 'Pause' : 'Play';
}

// ---------------------------------------------------------------- data loading
async function loadSequence(index) {
  loading = true;
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

// Scrubbing: pause while dragging, resume on release if it was playing.
let wasPlayingBeforeScrub = false;
progress.addEventListener('pointerdown', () => {
  wasPlayingBeforeScrub = playing;
  setPlaying(false);
});
progress.addEventListener('input', () => {
  if (nframes > 0) setFrame(parseInt(progress.value, 10));
});
progress.addEventListener('pointerup', () => {
  if (wasPlayingBeforeScrub) setPlaying(true);
});
progress.addEventListener('change', () => {
  if (wasPlayingBeforeScrub) setPlaying(true);
});

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
requestAnimationFrame(animate);
if (seqSelect.options.length > 0) {
  loadSequence(parseInt(seqSelect.value, 10));
} else {
  statusEl.textContent = 'no sequences in this split';
}
