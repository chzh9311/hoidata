## HOIData Analyzer & Loaders

Self-implemented dataset preprocessing, dataloaders, and viewers for HOI MoCap datasets.

Current implemented datasets:

* [GRAB](https://github.com/otaheri/GRAB)
* [ARCTIC](https://github.com/zc-alexfan/arctic)
* [HOT3D](https://github.com/facebookresearch/hot3d) (Aria and Quest3 recordings)
* [GiGaHands](https://github.com/brown-ivl/GigaHands)

## Installation

Tested with Python 3.10, PyTorch 2.5.1 (CUDA 12.4) and PyTorch3D 0.7.8 on Linux
with an NVIDIA GPU. The dataloaders, preprocessing and viewers pose MANO hands
on the GPU, so a CUDA-capable GPU is required.

### 1. Create the environment

```bash
conda create -n hoi_common python=3.10 -y
conda activate hoi_common
```

### 2. Install PyTorch (CUDA 12.4)

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
```

### 3. Install PyTorch3D

PyTorch3D is built from source, so the CUDA Toolkit must match PyTorch's CUDA
version (12.4). Check with `nvcc --version` first.

```bash
pip install fvcore iopath
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

### 4. Install MANO dependencies

[manopth](https://github.com/hassony2/manopth) loads the MANO model files,
which need [chumpy](https://github.com/mattloper/chumpy). Install chumpy from
GitHub: the PyPI release (0.70) fails to import with NumPy ≥ 1.24.

```bash
pip install --no-build-isolation "git+https://github.com/mattloper/chumpy"
pip install "git+https://github.com/hassony2/manopth"
```

### 5. Install the remaining dependencies and this package

```bash
pip install numpy scipy scikit-learn pandas tqdm matplotlib imageio easydict \
    trimesh open3d omegaconf hydra-core lightning flask
pip install -e .
```

### 6. Download the MANO models

Register at [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de/) and download
**Models & Code** (MANO v1.2). Unpack it so that `MANO_LEFT.pkl` and
`MANO_RIGHT.pkl` are in one folder, e.g. `mano_v1_2/models/`, then point every
config to it:

```yaml
# configs/{grab,arctic,gigahands,hot3d}.yaml
mano_root: "/path/to/mano_v1_2/models"
```

### 7. Check the installation

```bash
python -c "import torch, pytorch3d, manopth, hoidata; print(torch.__version__, torch.cuda.is_available())"
```

This should print `2.5.1+cu124 True`.

## Dataset setup

Raw datasets are expected under `data/` (paths are set in `configs/*.yaml`).
Symlinks to wherever the data is stored work fine, e.g.
`ln -s /storage/GRAB data/grab_raw`. Preprocessed outputs are written to
`data/output/<dataset>/`.

| Dataset | Location | Expected contents |
| --- | --- | --- |
| GRAB | `data/grab_raw/` | `grab/s1 … s10/*.npz`, `tools/object_meshes/contact_meshes/`, `tools/subject_meshes/` |
| ARCTIC | `data/arctic_raw/` | `raw_seqs/`, `meta/object_vtemplates/`, `description_text2hoi/` |
| GigaHands | `data/GigaHands/` | `annotations_v2.jsonl`, `sequence_type.json`, `hand_poses/`, `object_poses/`, `object_meta/` |
| HOT3D | `data/hot3d_raw/` | `assets/` (`instance.json`, `*.glb`) and the `P00xx_<hash>/` recording folders |

**HOT3D** only needs the ground-truth MoCap files, not the images (which are
over 500 GB). Follow the [HOT3D download instructions](https://github.com/facebookresearch/hot3d#step-2-sign-up-and-get-the-download-links-file)
to get the download-link JSON files, then with the downloader in
`hot3d/data_downloader/`:

* download the object library (`Hot3DAssets_download_urls.json`, all sequences)
  into `data/hot3d_raw/assets/`;
* download the Aria and Quest recordings
  (`Hot3DAria_download_urls.json`, `Hot3DQuest_download_urls.json`,
  `--sequence_name all`) with only the `ground_truth` and `hand_data` data
  types. Run the downloader without `--data_types` to list the type indices.

Only the 13 participants of the official training set ship with ground truth;
the official test participants (P0004, P0005, P0006, P0008, P0016, P0020) have
no poses and are skipped by the loaders.

## Preprocessing

GRAB and ARCTIC are preprocessed into a single `data.npz` (plus object point
clouds and text files) before training:

```bash
python -m hoidata.preprocess.preprocess_grab      # configs/grab.yaml
python -m hoidata.preprocess.preprocess_arctic    # configs/arctic.yaml
```

For HOT3D, the single-object segments used by `MotionHOT3D` are built with the
command below. The DataModule's `prepare_data` and the viewer's
`--single-object` mode also run it automatically when the output is missing or
the `seg_*` settings changed.

```bash
python -m hoidata.preprocess.preprocess_hot3d --config configs/hot3d.yaml
```

GigaHands and the multi-object HOT3D loader (`MotionHOT3DMultiObj`) read the
raw files directly and need no preprocessing.

## Web viewer

```bash
python -m hoidata.vis.web_viewer.run --config configs/grab.yaml --split test
```

Then open <http://127.0.0.1:8080/> (`--port` changes the port). Add
`--single-object` with `configs/hot3d.yaml` to browse the HOT3D single-object
segments; run with `--help` for the other view modes.
