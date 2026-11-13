# Legacy Codebase Inventory (`carle-v2x`)

This inventory documents all scripts, modules, inputs, outputs, and corresponding paper artifacts from the original repository cloned into `legacy/`.

---

## 1. Directory Structure

```
legacy/
├── setup_models.py           # Model download and CoreML export utility
├── core/
│   ├── segmentor.py          # YOLO11 segmentation wrapper & priority class mapping
│   ├── nuscenes_loader.py    # nuScenes dataset loading and frame iterator
│   ├── xor_rle.py            # Bitwise XOR and Run-Length Encoding logic
│   └── decode_rle.py         # Reconstruction of masks from RLE entries
├── pipelines/
│   ├── yolo11n.py            # End-to-end evaluation pipeline with YOLO11n-seg
│   ├── yolo11l.py            # End-to-end evaluation pipeline with YOLO11l-seg
│   └── lraspp.py             # End-to-end evaluation pipeline with LRASPP-MobileNetV3
├── analysis/
│   ├── benchmark.py          # Side-by-side timing benchmark across models
│   ├── metrics.py            # Compression ratio, bits/frame, latency, and mIoU calculator
│   ├── scene_stats.py        # Per-scene and per-camera aggregations
│   └── visualize.py          # Visual mask overlay generation onto raw frames
└── results/
    ├── compression_output.csv # Baseline comparison on Cityscapes gtFinePanopticParts (1,488 rows)
    ├── model_summary.csv      # High-level comparison across backbones (Table 5/6)
    ├── yolo11n_nuscenes.csv   # Full 2,424-frame nuScenes output with YOLO11n-seg
    ├── yolo11l_nuscenes.csv   # Full 2,424-frame nuScenes output with YOLO11l-seg
    └── lraspp_nuscenes.csv    # Full 2,424-frame nuScenes output with LRASPP-MobileNetV3
```

---

## 2. File-by-File Inventory

### `setup_models.py`
- **Purpose**: Downloads pretrained PyTorch weights for YOLO11n-seg, YOLO11l-seg, and LRASPP-MobileNetV3, and exports them to Apple CoreML format (`.mlpackage`) targeted at Apple Neural Engine.
- **Inputs**: PyTorch / Torchvision model weights, Ultralytics YOLO models.
- **Outputs**: `models/yolo11n-seg.mlpackage`, `models/yolo11l-seg.mlpackage`, `models/lraspp_mobilenet.mlpackage`.
- **Paper Link**: Pre-requisite setup for all model benchmarking (Tables 1, 3, 5, 6).

---

### `core/xor_rle.py`
- **Purpose**: Implements frame-to-frame bitwise XOR differencing and sequential run-length encoding.
- **Key Functions**:
  - `xor_masks(cur, prev)`: Returns `np.bitwise_xor(cur, prev)`.
  - `rle_encode(flat)`: Flattens array, detects value changes via `np.diff`, returns triples `[start, length, value]`.
  - `xor_and_compress(cur_mask, prev_mask)`: Calls `xor_masks` and `rle_encode`. Calculates `compressed_bytes = len(rle) * 6` (assuming 3 integers × 2 bytes/int16).
- **Inputs**: Two 2D `uint8` priority masks (`cur_mask`, `prev_mask`).
- **Outputs**: RLE triple list, total pixel count, changed pixel count, original byte estimate, compressed byte estimate.
- **Paper Link**: Foundation of CARLE compression mechanism; computes values for Tables 1, 3, 5, 6.
- **Critical Audit Notes**:
  - `np.bitwise_xor` on priority labels (e.g. pedestrian=2, vehicle=3) produces synthetic class IDs (e.g. 2 ^ 3 = 1).
  - Assumes 2-byte integers for `start`, which overflows 65,535 for HD/FHD resolutions (e.g. 900×1600 = 1,440,000 pixels).
  - `compressed_bytes` is a formula estimate (`len(rle) * 6`), not an actual serialized bitstream.

---

### `core/segmentor.py`
- **Purpose**: Wraps Ultralytics YOLO segmentation and maps COCO 80 classes to CARLE 4 priority classes:
  - Class 0: Background / Ignore
  - Class 2: Pedestrian (COCO person)
  - Class 3: Vehicle (COCO car, motorcycle, bus, truck)
  - Class 4: Animal (COCO cat, dog, horse, sheep, cow, elephant)
- **Inputs**: Camera frame image file path (`img_path`).
- **Outputs**: 2D `uint8` numpy mask of shape `(H, W)` with priority values in `{0, 2, 3, 4}`.
- **Paper Link**: Semantic segmentation stage for YOLO pipelines (Tables 1, 5, 6).

---

### `core/nuscenes_loader.py`
- **Purpose**: Iterates over nuScenes dataset frames across scenes and cameras.
- **Key Functions**:
  - `load_nusc(version="v1.0-mini")`: Initializes nuScenes devkit.
  - `iter_cam_front(nusc)`: Yields front camera frames.
  - `iter_all_cameras(nusc)`: Yields all 6 camera channels (`CAM_FRONT`, `CAM_FRONT_LEFT`, `CAM_FRONT_RIGHT`, `CAM_BACK`, `CAM_BACK_LEFT`, `CAM_BACK_RIGHT`).
- **Inputs**: `data/nuscenes` dataset directory.
- **Outputs**: Frame paths, timestamps, scene names, camera IDs, sample tokens.
- **Paper Link**: Feeds the nuScenes evaluations (Tables 1, 5, 6; 2,424 total frames).

---

### `core/decode_rle.py`
- **Purpose**: Reconstructs a 2D mask from RLE code triples `[start, length, value]`.
- **Inputs**: `rle_code` list or string representation from CSV, `height`, `width`.
- **Outputs**: Reconstructed 2D `uint8` mask of shape `(H, W)`.
- **Paper Link**: Receiver-side reconstruction.
- **Critical Audit Notes**:
  - Reconstructs only the *differential* XOR mask, not the reconstructed current frame (it does not perform XOR with the previous frame).

---

### `pipelines/yolo11n.py`, `pipelines/yolo11l.py`, `pipelines/lraspp.py`
- **Purpose**: End-to-end execution of segmentation, temporal XOR, RLE compression, and CSV metrics logging for all 2,424 frames of nuScenes mini.
- **Inputs**: nuScenes mini dataset via `nuscenes_loader.py`.
- **Outputs**: `results/yolo11n_nuscenes.csv`, `results/yolo11l_nuscenes.csv`, `results/lraspp_nuscenes.csv`.
- **Paper Link**: Generates raw data for Table 5 and Table 6.
- **Critical Audit Notes**:
  - In `yolo11n.py`: `OriginalSize2(Bytes)` is set to `Path(img_path).stat().st_size`, which is the raw JPEG camera image size (~100–150 KB), NOT the raw uncompressed label map size (1.44 MB) nor the log2(C) bit budget.
  - Time measured with `time.perf_counter()` spans both segmentation inference and compression sequentially.

---

### `analysis/benchmark.py`
- **Purpose**: Comparative per-frame latency measurement across models on selected camera frames.
- **Inputs**: nuScenes mini frames, model checkpoints.
- **Outputs**: Console timing comparisons and average latency per model.
- **Paper Link**: Latency measurements reported in Table 5.

---

### `analysis/metrics.py`
- **Purpose**: Calculates aggregate compression ratio (CR), bits per frame (bpf), bandwidth reduction percentage, latency stats, and mIoU / pixel accuracy.
- **Inputs**: Pipeline output CSV files (`results/*.csv`).
- **Outputs**: Computed metrics, summary statistics, memory usage tracking via `tracemalloc`.
- **Paper Link**: Defines and computes metrics presented in Tables 1, 3, 5, 6.

---

### `analysis/scene_stats.py`
- **Purpose**: Aggregates timing and compression metrics grouped by scene and by camera.
- **Inputs**: Pipeline CSV outputs.
- **Outputs**: Tabular breakdown of frames, latency (min/avg/max), CR, and RLE entries by scene/camera.
- **Paper Link**: Corresponds to per-scene and per-camera analyses discussed in paper text.

---

### `analysis/visualize.py`
- **Purpose**: Renders color overlays of decoded priority masks onto original driving frames and draws bounding boxes around detected regions.
- **Inputs**: nuScenes image frames, CSV with RLE codes.
- **Outputs**: Annotated visual output images in `output/visuals/`.
- **Paper Link**: Corresponds to qualitative figures in the paper showing detected road/vehicle/pedestrian overlays.

---

## 3. Results File Inventory

| File | Rows | Contents | Source Script | Paper Target |
|---|---|---|---|---|
| `results/compression_output.csv` | 1,488 | Cityscapes `gtFinePanopticParts` pairwise compression | Legacy Cityscapes script (pre-computed) | Table 1 / Table 3 |
| `results/model_summary.csv` | 4 | Summary rows comparing YOLO11n-seg, YOLO11l-seg, and LRASPP | Summary of pipeline CSVs | Table 5 / Table 6 |
| `results/yolo11n_nuscenes.csv` | 2,424 | Per-frame metrics for all 6 cameras across 10 nuScenes scenes | `pipelines/yolo11n.py` | Table 5 / Table 6 |
| `results/yolo11l_nuscenes.csv` | 2,424 | Per-frame metrics for YOLO11l-seg on nuScenes | `pipelines/yolo11l.py` | Table 5 / Table 6 |
| `results/lraspp_nuscenes.csv` | 2,424 | Per-frame metrics for LRASPP on nuScenes | `pipelines/lraspp.py` | Table 5 / Table 6 |
