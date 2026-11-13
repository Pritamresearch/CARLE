# CARLE Research Lab Notebook

## Phase S — Setup and Environment Initialization
**Date:** 2026-11-13  
**Hardware Detected:** Intel CPU (16 logical processors), NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM), Windows 11.  
**Tools:** Git 2.50.1, Python 3.12.10, ffmpeg N-111115 (with `libx265` and `ffv1` codecs enabled in `C:/MediaToolkit/ffmpeg.exe`).

### Activities & Commands:
1. **Repository & Directory Structure Initialization:**
   - Initialized Git repository (`git init`).
   - Created folder hierarchy: `carle/`, `baselines/`, `experiments/`, `configs/`, `results/`, `results/setup/`, `tests/`.
   - Cloned original legacy repository into `legacy/` (`git clone https://github.com/Pritamresearch/carle-v2x.git legacy/`).
   - Documented full file-by-file inventory in `legacy/INVENTORY.md`.

2. **Environment & Dependencies:**
   - Created virtual environment `.venv` with Python 3.12.10.
   - Installed core scientific stack, deep learning framework, compression libraries:
     `numpy`, `numba`, `torch`, `torchvision`, `ultralytics`, `opencv-python`, `pillow`, `zstandard`, `lz4`, `python-snappy`, `cramjam`, `constriction`, `pandas`, `scipy`, `matplotlib`, `pyyaml`, `tqdm`, `pytest`, `nuscenes-devkit`, `shapely`, `scikit-learn`.
   - Pinned and exported all packages to `requirements.txt`.
   - Trivial pytest test suite created (`tests/test_trivial.py`) and verified (`pytest -q tests` passed).

3. **Dataset Discovery & Linking:**
   - Discovered datasets across `C:/PhD`:
     - **nuScenes mini (`v1.0-mini`)**: `C:/PhD/TS BEV/TS BEV CODE/dataset/nuScenes` (5.01 GB, 10 scenes, 4,848 sample files, 26,358 sweep files).
     - **Cityscapes**: `C:/PhD/NetSegNav` (`gtFine`: 20,000 files, `leftImg8bit`: 5,000 images, 11.55 GB).
     - **IDD 20K II**: `C:/PhD/Dataset/IDD 20K II` (8.42 GB, 10,098 images, 8,089 masks).
     - **Field-test Driving Videos**: `C:/PhD/Semantic Depth/realtime/data/videos` (Sikkim, Bolpur, Bhubaneswar, Dhaka, Malaysia).
   - Created directory junctions in `data/`:
     - `data/nuscenes` -> `C:/PhD/TS BEV/TS BEV CODE/dataset/nuScenes`
     - `data/cityscapes` -> `C:/PhD/NetSegNav`
     - `data/idd` -> `C:/PhD/Dataset/IDD 20K II`
     - `data/field_videos` -> `C:/PhD/Semantic Depth/realtime/data/videos`
   - Configured canonical paths in `configs/paths.yaml`. Added `data/` and `results/**/cache/` to `.gitignore`.

4. **Smoke Test Execution:**
   - Executed `experiments/smoke_test.py` with YOLO11n-seg across 5 frames each for nuScenes, Cityscapes, IDD, and Sikkim field-test video.
   - All 20 visual overlays rendered and saved to `results/setup/`.
   - Median per-frame latency:
     - nuScenes: 91.9 ms
     - Cityscapes: 98.0 ms
     - IDD: 59.2 ms
     - Field-test Video: 39.6 ms

### Surprises & Observations:
- In `legacy/core/xor_rle.py`, `compressed_bytes` is estimated by formula (`len(rle) * 6`), not actual serialized byte payload.
- In `legacy/pipelines/yolo11n.py`, the compression ratio denominator was calculated against the JPEG file size on disk rather than uncompressed pixel or log2(C) bit representations.
- Cityscapes dataset contains `gtFine` and `leftImg8bit` (train/val/test), but `leftImg8bit_sequence` is absent locally. Pre-computed Cityscapes `gtFinePanopticParts` pairwise compression exists in `legacy/results/compression_output.csv`.
