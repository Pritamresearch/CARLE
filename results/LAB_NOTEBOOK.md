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


## Phase 0 — Audit and Paper Reproduction
**Date:** 2026-11-13  
**Auditor:** Research Engineer Agent  
**Environment Record:** `results/phase0/env.json` (Intel i7-13620H, 16 threads, RTX 3050 Laptop GPU, 16 GB RAM).  
**CPU-Only Smoke Test Latency (4 threads, 10 warmup):**
- nuScenes: 76.2 ms
- Cityscapes: 65.9 ms
- IDD: 44.0 ms
- Field-test Video: 40.4 ms

### Deliverables Generated:
- `results/phase0/AUDIT.md`: Detailed audit covering method vs code mismatch, size accounting, latency conflation, Cityscapes pairing flaw, nuScenes 2 Hz vs 12 Hz sampling, field video matching, and table-by-table resolution.
- `results/phase0/reproduction.csv`: Quantitative side-by-side comparison of paper values vs reproduced metrics.
- `results/phase0/cityscapes_pairs/`: 5 visual comparison overlays demonstrating real visual difference (39% - 54%) between paired Aachen snippets.
- `legacy_local/`: Created read-only directory junction to `C:\PhD\CARLE`.

### Core Audit Discoveries:
1. **6,270.9× Headline CR Origin**: Computed as the arithmetic mean of per-frame ratios against camera JPEG image files on disk (`OriginalSize2`), rather than raw uncompressed label map bytes. True aggregate sum-ratio vs Raw 8-bit is 265.1×; true sum-ratio vs 2-bit log2(C) is 66.3×; real 7-byte binary serialization is 56.8×.
2. **Method vs Code Mismatch**: `legacy/` uses `np.bitwise_xor(cur, prev)` and `decode_rle.py` never XORs with $F_t$, producing corrupted class IDs (e.g. 2 ^ 3 = 1) and failing exact reconstruction (`False`). In contrast, `legacy_local/CARLE CODE/main_PIDNET.py` correctly implements Eq. 2 ($D = F_{t+1}$ where changed, else 0) and serializes 7-byte packets with exact reconstruction (`True`).
3. **Class-0 vs Null-symbol Collision**: Solved in `main_PIDNET.py` by mapping classes to positive integers 1..4 and reserving 0 as NULL. `legacy/` confounded Background=0 with No-change=0.
4. **Cityscapes 0.007% Delta Flaw**: Non-adjacent snippets from Aachen paired; 1,448 of 1,487 rows had exactly 2 RLE entries and 39 rows had 1 entry because only the first small run was captured, omitting the remaining 1.5 million pixels.
5. **nuScenes Sampling Rate**: Evaluated on keyframes (`samples`) at 2.00 Hz (500 ms gap), not streaming video rate (11.86 Hz, 84 ms gap).
6. **Frame Counts & Pairs**: nuScenes 2,424 total frames - 60 starts = 2,364 pairs. Field videos 1,073 total frames - 4 starts = 1,069 pairs. 547 pairs corresponds to 2× subsampling.
7. **DVC / UCVC in Table 3**: No code exists in either repository; marked `NOT_RUN` and flagged for removal.
