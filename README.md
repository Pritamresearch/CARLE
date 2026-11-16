# CARLE V2: Real-Time Multi-Camera Semantic-Map Sharing for V2X Collaborative Perception

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-brightgreen.svg)]()
[![C++17](https://img.shields.io/badge/C%2B%2B-17-orange.svg)]()
[![CUDA 12.6](https://img.shields.io/badge/CUDA-12.6-green.svg)]()

This repository contains the official implementation, evaluation suite, and experimental results for **CARLE V2**, an empirical study on **real-time, multi-camera semantic label-map compression for Vehicle-to-Everything (V2X) collaborative autonomous driving**.

---

## 1. Overview & Research Motivation

Connected and Automated Vehicles (CAVs) increasingly rely on **cooperative perception** via V2X networks to see through occlusions and extend sensing ranges beyond direct line-of-sight. However:
1. **Raw Sensor Sharing Bottleneck**: Transmitting raw LiDAR point clouds or multi-camera RGB video consumes tens to hundreds of Megabits per second (Mb/s) per vehicle, far exceeding dedicated V2X radio allocations (e.g., DSRC / C-V2X PC5 channels at 10–20 MHz).
2. **Semantic Abstraction**: Transmitting high-level semantic occupancy and segmentation maps drastically compresses scene information into actionable spatial primitives (roads, dynamic vehicles, pedestrians).
3. **General-Purpose Codec Inefficiency**: Standard lossless codecs (PNG, WebP, JPEG-XL) are designed for natural photographic RGB imagery or generic graphics. They fail to exploit the geometric and topological properties of driving segmentation maps (piecewise-constant regions, smooth contour boundaries, class transitions).

This benchmark establishes rigorous comparisons between state-of-the-art **contour-chain segmentation codecs (Yang et al. 2026, InterDigital)**, **context-adaptive arithmetic range coding (SemCM)**, and standard industry codecs (**JPEG-XL, WebP, PNG**) across **31,822 cached frames** from nuScenes (keyframes & sweeps) and diverse real-world field-test sequences.

---

## 2. System Architecture

```mermaid
flowchart TD
    subgraph SENSING["1. Multi-Camera Sensing (6 Surround Cameras)"]
        CAM["CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT\nCAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT"]
    end

    subgraph SEG["2. Real-Time Neural Semantic Segmentation (GPU)"]
        PID["PIDNet-S / M / L (Cityscapes Weights)\nBatched FP16 Inference (Batch=6)"]
        RES["Resolution: Full (900x1600) vs Half (450x800)"]
        ONTO["Class Projection: K=4, K=8, K=19"]
        PID --> RES --> ONTO
    end

    subgraph CODECS["3. Semantic-Map Lossless Compression Suite"]
        YANG["Yang et al. 2026\nContour-Chain + Golomb-Rice + Arithmetic Coding"]
        SEMCM["SemCM (a) [Frame-Independent]\n2-Fold Pretrained Prior (w=16384)"]
        JXL["JPEG-XL (Effort 9, Lossless)"]
        WEBP["WebP Lossless (Method 4)"]
        PNG["PNG (OpenCV Level 9)"]
    end

    subgraph V2X["4. Low-Bandwidth V2X Transmission"]
        V2X_OUT["Dedicated V2X / C-V2X Channel (10 Hz Multi-Camera Stream)"]
    end

    CAM --> PID
    ONTO --> YANG
    ONTO --> SEMCM
    ONTO --> JXL
    ONTO --> WEBP
    ONTO --> PNG
    YANG --> V2X_OUT
```

---

## 3. Evaluated Datasets & Class Ontologies

### Datasets (All 31,822 Cached Frames)
- **nuScenes Keyframes (2 Hz)**: 2,424 annotated multi-camera keyframe sets.
- **nuScenes Sweeps (12 Hz)**: 6,000 contiguous multi-camera sweep frames across 10 diverse scenes (all 6 cameras, 100 frames/camera).
- **Field-Test Driving Video (Native & 10 Hz)**:
  - *Unflagged Benchmark (3,673 native frames / 1,399 10 Hz frames)*: Bolpur, Malaysia, Sikkim, Dhaka Day, Dhaka Night.
  - *Flagged Domain-Shift (1,811 native frames / 604 10 Hz frames)*: Bhubaneswar Night (isolated due to extreme headlight flare on the ego-vehicle hood).

### Label Granularities ($K$)
- **$K = 4$**:
  1. Road $\{0\}$
  2. Vehicle $\{13, 14, 15, 16\}$ (Car, Truck, Bus, Train)
  3. Vulnerable Road User (VRU) $\{11, 12, 17, 18\}$ (Pedestrian, Rider, Motorcycle, Bicycle)
  4. Other Static $\{1..10\}$ (Sidewalk, Building, Wall, Fence, Pole, Sign, Vegetation, Sky)
- **$K = 8$**:
  1. Road, 2. Sidewalk, 3. Static Structure, 4. Vegetation/Terrain, 5. Sky, 6. Vehicle, 7. Two-Wheeler, 8. Person/Rider
- **$K = 19$**: Full Cityscapes ontology (trainIds $0..18$).

---

## 4. Key Experimental Results

### Table 1: End-to-End Codec Master Benchmark (Unflagged Datasets)
Measured across all cached frames with 95% bootstrap confidence intervals clustered by scene/clip (1,000 resamples). Bitrates assume a 10 Hz stream transmission rate.

#### Granularity $K = 4$
| Dataset | $N$ Frames | Codec | Bits/Frame | 95% CI Lower | 95% CI Upper | 1-Cam 10 Hz | 6-Cam 10 Hz | Gain vs SemCM (a) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **nuScenes Keyframes (2 Hz)** | 2,424 | **Yang et al. 2026** | **5,980.1** | 5,408.5 | 6,577.4 | **59.8 kb/s** | **358.8 kb/s** | **+45.97%** |
| | | SemCM (a) [Frame-Indep] | 11,068.4 | 10,124.0 | 11,962.1 | 110.7 kb/s | 664.1 kb/s | 0.00% (Ref) |
| | | JPEG-XL (Effort 9) | 19,608.8 | 18,129.7 | 21,026.4 | 196.1 kb/s | 1,176.5 kb/s | -77.16% |
| | | WebP Lossless | 21,869.3 | 19,912.5 | 23,922.0 | 218.7 kb/s | 1,312.2 kb/s | -97.58% |
| | | PNG (OpenCV level 9) | 49,391.2 | 46,519.1 | 52,311.0 | 493.9 kb/s | 2,963.5 kb/s | -346.23% |
| **nuScenes Sweeps (12 Hz)** | 6,000 | **Yang et al. 2026** | **5,884.5** | 5,265.8 | 6,531.2 | **58.9 kb/s** | **353.1 kb/s** | **+45.94%** |
| | | SemCM (a) [Frame-Indep] | 10,884.4 | 9,856.6 | 11,897.8 | 108.8 kb/s | 653.1 kb/s | 0.00% (Ref) |
| | | JPEG-XL (Effort 9) | 19,405.2 | 17,836.9 | 20,973.4 | 194.1 kb/s | 1,164.3 kb/s | -78.28% |
| | | WebP Lossless | 21,426.9 | 19,165.0 | 23,605.8 | 214.3 kb/s | 1,285.6 kb/s | -96.86% |
| | | PNG (OpenCV level 9) | 48,803.0 | 45,785.2 | 52,098.5 | 488.0 kb/s | 2,928.2 kb/s | -348.38% |
| **Field Native** | 3,673 | **Yang et al. 2026** | **3,289.7** | 1,858.1 | 6,111.9 | **32.9 kb/s** | **197.4 kb/s** | **+44.28%** |
| | | SemCM (a) [Frame-Indep] | 5,904.4 | 3,673.3 | 10,342.8 | 59.0 kb/s | 354.3 kb/s | 0.00% (Ref) |
| | | JPEG-XL (Effort 9) | 9,347.6 | 5,404.5 | 16,967.9 | 93.5 kb/s | 560.9 kb/s | -58.32% |
| | | WebP Lossless | 12,668.3 | 7,210.7 | 22,892.8 | 126.7 kb/s | 760.1 kb/s | -114.56% |
| | | PNG (OpenCV level 9) | 25,483.8 | 16,012.4 | 43,520.5 | 254.8 kb/s | 1,529.0 kb/s | -331.60% |
| **Field 10 Hz** | 1,399 | **Yang et al. 2026** | **3,704.8** | 1,752.1 | 6,233.5 | **37.1 kb/s** | **222.3 kb/s** | **+43.50%** |
| | | SemCM (a) [Frame-Indep] | 6,557.0 | 3,490.8 | 10,617.9 | 65.6 kb/s | 393.4 kb/s | 0.00% (Ref) |
| | | JPEG-XL (Effort 9) | 10,250.7 | 5,302.9 | 17,046.2 | 102.5 kb/s | 615.0 kb/s | -56.33% |
| | | WebP Lossless | 13,874.1 | 7,094.8 | 23,074.8 | 138.7 kb/s | 832.5 kb/s | -111.59% |
| | | PNG (OpenCV level 9) | 27,872.2 | 15,808.8 | 43,776.2 | 278.7 kb/s | 1,672.3 kb/s | -325.09% |

#### Granularity $K = 8$
| Dataset | $N$ Frames | Codec | Bits/Frame | 95% CI Lower | 95% CI Upper | 1-Cam 10 Hz | 6-Cam 10 Hz | Gain vs SemCM (a) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **nuScenes Keyframes (2 Hz)** | 2,424 | **Yang et al. 2026** | **12,400.1** | 11,466.7 | 13,382.6 | **124.0 kb/s** | **744.0 kb/s** | **+53.64%** |
| | | SemCM (a) [Frame-Indep] | 26,749.9 | 24,675.0 | 28,841.4 | 267.5 kb/s | 1,605.0 kb/s | 0.00% (Ref) |
| | | JPEG-XL (Effort 9) | 35,170.8 | 32,608.2 | 37,564.3 | 351.7 kb/s | 2,110.3 kb/s | -31.48% |
| | | WebP Lossless | 45,336.5 | 42,020.5 | 48,653.6 | 453.4 kb/s | 2,720.2 kb/s | -69.44% |
| | | PNG (OpenCV level 9) | 82,105.4 | 77,357.2 | 86,331.2 | 821.1 kb/s | 4,926.3 kb/s | -206.86% |
| **nuScenes Sweeps (12 Hz)** | 6,000 | **Yang et al. 2026** | **12,175.0** | 11,156.0 | 13,233.1 | **121.8 kb/s** | **730.5 kb/s** | **+53.31%** |
| | | SemCM (a) [Frame-Indep] | 26,078.1 | 23,912.6 | 28,197.2 | 260.8 kb/s | 1,564.7 kb/s | 0.00% (Ref) |
| | | JPEG-XL (Effort 9) | 34,577.6 | 31,881.6 | 37,136.4 | 345.8 kb/s | 2,074.7 kb/s | -32.59% |
| | | WebP Lossless | 44,284.4 | 40,509.4 | 48,030.3 | 442.8 kb/s | 2,657.1 kb/s | -69.81% |
| | | PNG (OpenCV level 9) | 80,571.1 | 75,036.8 | 85,650.2 | 805.7 kb/s | 4,834.3 kb/s | -208.96% |
| **Field Native** | 3,673 | **Yang et al. 2026** | **6,005.0** | 3,916.9 | 9,615.7 | **60.1 kb/s** | **360.3 kb/s** | **+53.55%** |
| | | SemCM (a) [Frame-Indep] | 12,928.1 | 8,809.5 | 20,344.1 | 129.3 kb/s | 775.7 kb/s | 0.00% (Ref) |
| | | JPEG-XL (Effort 9) | 15,970.4 | 10,444.0 | 26,315.1 | 159.7 kb/s | 958.2 kb/s | -23.53% |
| | | WebP Lossless | 23,131.2 | 15,517.2 | 36,708.3 | 231.3 kb/s | 1,387.9 kb/s | -78.92% |
| | | PNG (OpenCV level 9) | 38,502.2 | 25,432.4 | 60,332.1 | 385.0 kb/s | 2,310.1 kb/s | -197.82% |
| **Field 10 Hz** | 1,399 | **Yang et al. 2026** | **6,527.0** | 3,814.4 | 9,772.9 | **65.3 kb/s** | **391.6 kb/s** | **+53.46%** |
| | | SemCM (a) [Frame-Indep] | 14,025.9 | 8,954.2 | 20,687.5 | 140.3 kb/s | 841.6 kb/s | 0.00% (Ref) |
| | | JPEG-XL (Effort 9) | 17,498.0 | 10,470.8 | 26,809.2 | 175.0 kb/s | 1,049.9 kb/s | -24.75% |
| | | WebP Lossless | 25,135.8 | 15,893.1 | 37,348.9 | 251.4 kb/s | 1,508.2 kb/s | -79.21% |
| | | PNG (OpenCV level 9) | 41,761.1 | 26,894.5 | 61,496.1 | 417.6 kb/s | 2,505.7 kb/s | -197.74% |

---

### Table 2: Edge Hardware Latency & Multi-Camera Throughput
Evaluated on **Intel Core i7-13620H** and **NVIDIA GeForce RTX 3050 6GB Laptop GPU** (FP16 mode):

| Stage | Model / Codec | Resolution | Batch Size | Latency | Throughput | 10 Hz Real-Time Feasibility |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Segmentation** | PIDNet-S | Full ($900 \times 1600$) | 1 | 59.57 ms | 16.8 FPS | Single-camera only |
| **Segmentation** | PIDNet-S | Full ($900 \times 1600$) | 6 | 311.26 ms total | 19.3 FPS | Exceeds 100 ms 6-cam budget |
| **Segmentation** | **PIDNet-S** | **Half ($450 \times 800$)** | **6** | **90.99 ms total** | **65.9 FPS** | **$\le 100\text{ ms}$ (All 6 cameras real-time)** |
| **Segmentation** | PIDNet-M | Half ($450 \times 800$) | 6 | 228.03 ms total | 26.3 FPS | Exceeds 100 ms budget |
| **Segmentation** | PIDNet-L | Half ($450 \times 800$) | 6 | 320.85 ms total | 18.7 FPS | Exceeds 100 ms budget |
| **Compression** | **Yang et al. 2026** | Full ($K=4$) | 1 (1 core) | **7.11 ms** | **140.6 FPS** | **$\le 10\text{ ms}$ per camera** |
| **Compression** | **Yang et al. 2026** | Full ($K=8$) | 1 (1 core) | **9.31 ms** | **107.4 FPS** | **$\le 10\text{ ms}$ per camera** |
| **Decompression** | **Yang et al. 2026** | Full ($K=4$) | 1 (1 core) | **2.10 ms** | **476.2 FPS** | Ultra-fast receiver decode |
| **Decompression** | **Yang et al. 2026** | Full ($K=8$) | 1 (1 core) | **3.04 ms** | **328.9 FPS** | Ultra-fast receiver decode |

---

## 5. Repository Structure

```text
CARLE V2/
├── baselines/
│   └── yang2026/                       # Yang et al. 2026 (InterDigital)
│       ├── source/                     # Production C++ source
│       ├── source_instrumented/        # Instrumented build with std::chrono & dead-code counters
│       ├── Makefile, CMakeLists.txt
│       └── README.md
├── carle/
│   ├── env.py                          # Environment detection
│   └── rigorous_codecs.py              # SemCM Numba range coder & context utilities
├── configs/
│   └── fieldtest.yaml                  # Field video parameters & ground truth metadata
├── experiments/
│   ├── p0_verify_yang.py               # Phase 0: 31,822 frame lossless round-trip proof
│   ├── prepare_instrumented_yang.py    # Generates instrumented source
│   ├── run_p0_dead_code_sample.py      # Stratified 600-frame counter & timing verification
│   ├── exp_11_pidnet_caching.py        # PIDNet segmentation caching
│   ├── exp_14_unified_freeze_benchmark.py # Unified 2-fold freeze runner
│   ├── exp_16_pidnet_sweep.py          # Latency, FPS, granularity & flicker study
│   └── exp_16_run_all_frames_p1.py     # Master all-frame benchmark runner
├── results/
│   ├── phase0/
│   │   └── env.json                    # Hardware identity (i7-13620H, RTX 3050 Laptop)
│   └── phase1/                         # Summary tables & benchmark CSVs
│       ├── p1_master_summary_k4.csv    # Master summary table (K=4)
│       ├── p1_master_summary_k8.csv    # Master summary table (K=8)
│       ├── p1_flagged_summary_k4.csv   # Flagged Bhubaneswar summary (K=4)
│       ├── p1_flagged_summary_k8.csv   # Flagged Bhubaneswar summary (K=8)
│       ├── pidnet_latency_benchmark.json
│       ├── pidnet_granularity_flicker_study.json
│       └── p0_dead_code_and_timing_results.json
└── README.md
```

---

## 6. Reproduction Guide

### Prerequisites
- Python $\ge$ 3.10 (tested on 3.12)
- MinGW-w64 GCC or Linux GCC supporting C++17
- NVIDIA GPU with CUDA 12.x and PyTorch with FP16 support

### 1. Compile the Yang et al. 2026 Codec
```bash
cd baselines/yang2026
make
# Or with MinGW GCC:
g++ -std=c++17 -O3 -static source/commonlib/*.cpp source/arithmetic_coder/*.cpp source/arithmetic_coder/third_party/*.cpp source/encoder/*.cpp -I source/commonlib -I source/arithmetic_coder -I source/arithmetic_coder/third_party -I source/encoder -o encoder.exe
g++ -std=c++17 -O3 -static source/commonlib/*.cpp source/arithmetic_coder/*.cpp source/arithmetic_coder/third_party/*.cpp source/decoder/*.cpp -I source/commonlib -I source/arithmetic_coder -I source/arithmetic_coder/third_party -I source/decoder -o decoder.exe
```

### 2. Verify Lossless Reconstruction & Dead-Code Counters
```bash
python experiments/run_p0_dead_code_sample.py
```

### 3. Run PIDNet Latency & Flicker Profiling
```bash
python experiments/exp_16_pidnet_sweep.py
```

### 4. Run the Master All-Frame Benchmark
```bash
python experiments/exp_16_run_all_frames_p1.py
```

---

## 7. Citation & References

```bibtex
@article{yang2026lossless,
  title={Lossless Segmentation-Map Compression via Contour-Chain Coding and Context-Adaptive Golomb-Rice Arithmetic Models},
  author={Yang, et al.},
  journal={arXiv preprint arXiv:2603.03073},
  year={2026}
}

@inproceedings{xu2023pidnet,
  title={PIDNet: A Real-time Semantic Segmentation Network Inspired by PID Controllers},
  author={Xu, Jiacong and Xiong, Zixiang and Bhattacharyya, Saurabh P.},
  booktitle={CVPR},
  year={2023}
}
```
