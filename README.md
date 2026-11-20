# Frame-Independent Coding of Semantic Segmentation Maps for V2X: Bit Cost, Temporal Redundancy, Packet Loss and Sidelink Capacity

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-brightgreen.svg)]()
[![C++17](https://img.shields.io/badge/C%2B%2B-17-orange.svg)]()
[![CUDA 12.4](https://img.shields.io/badge/CUDA-12.4-green.svg)]()
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6-red.svg)]()

Official code, benchmark suite, and reproduction scripts for the manuscript:
> **"Frame-Independent Coding of Semantic Segmentation Maps for V2X: Bit Cost, Temporal Redundancy, Packet Loss and Sidelink Capacity"**  
> Pritam Chakraborty and Anjan Bandyopadhyay  
> *School of Computer Engineering, Kalinga Institute of Industrial Technology, Bhubaneswar, India*  
> Email: `pritam23@ieee.org`  
> Repository: [https://github.com/Pritamresearch/Frame-Independent-Coding-of-Semantic-Segmentation-Maps-for-V2X](https://github.com/Pritamresearch/Frame-Independent-Coding-of-Semantic-Segmentation-Maps-for-V2X)

---

## 1. Overview & Core Findings

Cooperative vehicle-to-everything (V2X) communication allows connected automated vehicles to share camera perception beyond line-of-sight. Sharing raw video exceeds sidelink capacity (often constrained to 10–20 MHz channels), while sharing semantic segmentation label maps enables interoperable perception across vehicle models without requiring matched neural decoders at receivers.

This repository implements the complete processing, compression, and transmission evaluation chain across **31,822 multi-camera frames** (nuScenes keyframes and 12 Hz sweeps, plus South and Southeast Asian field driving clips).

### Key Empirical Findings

1. **Intra vs. Inter-Frame Compression**:
   - Semantic segmentation maps feature region boundaries that change frame-to-frame due to neural inference flicker and ego-motion. One-third of label changes revert within a single frame.
   - For pixel-context coding (SemCM), the best temporal context saves only **8.2%** (K=4) and **5.2%** (K=8) on held-out scenes.
   - In contrast, frame-independent context-adaptive chain coding (Yang et al., 2026) requires **2.65 kb** per 450×800 frame at K=4—which is **36% to 45% fewer bits** than temporal pixel coding without any temporal context, **2.7× fewer bits than JPEG-XL**, and **4.6× fewer than JPEG-LS**.

2. **Resilience to Packet Loss**:
   - On a 10% loss channel, frame-independent coding maintains **99.4% pixel accuracy** and adds only **9.2 ms** excess age of information (AoI).
   - Inter-frame prediction with periodic intra refresh every 10 frames degrades to **95.9% accuracy** and adds **177.8 ms** excess AoI due to error propagation.

3. **Packet Slicing Trade-offs**:
   - Slicing frames into 4 horizontal packets bounds individual packets below typical MTU thresholds (1,200 B) and reduces the 95th-percentile frame-average AoI under random loss (from 186.7 ms to 145.1 ms).
   - However, slicing provides no benefit under time-correlated fading bursts (50–100 ms) and incurs a 62% sidelink capacity overhead if each slice occupies separate transport blocks.

4. **Failure Modes & Cost Artifacts**:
   - Under severe domain shift (such as nighttime lighting), neural segmentation can collapse to single-class outputs (48 bits in chain coding). A low bit cost can therefore indicate perception failure rather than high compression efficiency.

5. **5G NR-V2X Sidelink Capacity**:
   - Under 3GPP Release 16 sidelink assumptions (20 MHz channel, 30 kHz subcarrier spacing, MCS 16, usable resource share ρ = 0.6), one channel supports **305 to 367 four-class streams** at 10–12 Hz with per-frame transport block sizing. Aggregating 6 surround cameras per vehicle supports approximately **500 camera streams** (~83 vehicles).

---

## 2. System Architecture

```text
[6 Surround Cameras] 
       │
       ▼ (T_pre = 41.9 ms)
[Preprocessing: Resize to 450x800 & Normalise]
       │
       ▼ (T_seg = 97.1 ms, Batched FP16 on GPU)
[PIDNet-S Neural Semantic Segmentation]
       │
       ▼ (Class Projection: K=4, K=8, K=19)
[Discrete Semantic Label Map]
       │
       ▼ (T_enc = 3.0 ms/frame, CPU)
[Context-Adaptive Extended Chain Coder (Yang et al. 2026)]
       │
       ▼ (T_air = 2.1 ms @ 10 Mb/s, 12-byte header)
[5G NR-V2X Sidelink Broadcast]
       │
       ▼ (T_dec = 1.1 ms/frame, CPU)
[Lossless Exact Map Reconstruction at Receiver]
```

---

## 3. Class Projection Mappings

In accordance with Table 1 of the paper, Cityscapes 19-class training IDs are projected into $K$-class representations:

- **$K = 4$**:
  - `Class 1` (Road): ID 0
  - `Class 2` (Vehicle): IDs 13, 14, 15, 16 (Car, Truck, Bus, Train)
  - `Class 3` (Vulnerable Road User): IDs 11, 12, 17, 18 (Pedestrian, Rider, Motorcycle, Bicycle)
  - `Class 4` (Background / Other): IDs 1–10 (Sidewalk, Building, Wall, Fence, Pole, Traffic Light, Sign, Vegetation, Terrain, Sky)

- **$K = 8$**:
  - `Class 1`: Road (0)
  - `Class 2`: Sidewalk (1)
  - `Class 3`: Static Structure (2, 3, 4, 5, 6, 7)
  - `Class 4`: Vegetation & Terrain (8, 9)
  - `Class 5`: Sky (10)
  - `Class 6`: Vehicle & Two-Wheeler (13, 14, 15, 16, 17, 18)
  - `Class 7`: Pedestrian (11)
  - `Class 8`: Rider (12)

- **$K = 19$**: Full native Cityscapes ontology (IDs 0–18).

---

## 4. Evaluated Codecs

| Codec Family | Implementation | Mode | Description |
| :--- | :--- | :--- | :--- |
| **Chain Coding** | Yang et al. (2026) | Intra | Context-adaptive extended chain coding of class boundaries with arithmetic entropy coding. |
| **SemCM** | In-house (Numba) | Intra / Inter | Causal 4-neighbour pixel context arithmetic range coder with offline/adaptive priors. |
| **JPEG-XL** | `libjxl` 0.8+ | Lossless | Modular mode at effort 9. |
| **WebP** | `libwebp` 1.3+ | Lossless | Predictive entropy coding (level 9). |
| **JPEG-LS** | CharLS 2.4.5 | Lossless | Near-lossless/lossless LOCO-I prediction. |
| **JBIG1 / JBIG2** | JBIG-KIT / jbig2enc | Bilevel | Bit-plane decomposition of label indices. |

---

## 5. Summary Benchmark Results

### Coded Frame Size (kb / frame at 450×800)

| Granularity | Codec | nuScenes (2 Hz) | nuScenes (12 Hz) | Field Clips | Field (10 Hz subs.) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **K = 4** | **Yang et al. (Chain)** | **2.61** | **2.65** | **2.91** | **3.28** |
| | SemCM (Pixel context) | 4.52 | 4.55 | 5.35 | 5.95 |
| | JPEG-XL (Effort 9) | 6.94 | 7.03 | 7.65 | 8.40 |
| | WebP Lossless | 9.13 | 9.22 | 10.93 | 12.15 |
| | JPEG-LS | — | 12.07 | — | — |
| | JBIG2 (bit planes) | — | 7.37 | — | — |
| **K = 8** | **Yang et al. (Chain)** | **5.49** | **5.44** | **4.92** | **5.25** |
| | SemCM (Pixel context) | 10.81 | 10.64 | 10.38 | 11.00 |
| | JPEG-XL (Effort 9) | 12.80 | 12.74 | 12.25 | 12.95 |
| | WebP Lossless | 18.98 | 18.76 | 18.25 | 19.25 |
| | JPEG-LS | — | 24.34 | — | — |
| | JBIG2 (bit planes) | — | 14.52 | — | — |

---

## 6. Repository Layout

```text
├── baselines/
│   └── yang2026/                    # Yang et al. (2026) C++ chain coder
│       ├── source/                  # Production encoder & decoder
│       ├── source_instrumented/     # Verified instrumentation build
│       └── LICENSE                  # BSD 3-Clause licence
├── carle/
│   ├── classes.py                   # Centralized K=4 and K=8 class mapping definitions
│   ├── inference.py                 # Factorized FP16 batched PIDNet-S inference
│   └── rigorous_codecs.py           # SemCM arithmetic coder implementation
├── configs/
│   └── fieldtest.yaml               # Video metadata and configurations
├── experiments/
│   ├── exp_17_cache_pidnet_half.py  # PIDNet-S half-resolution caching
│   ├── exp_18_cool_timing_pass.py   # Latency measurement under cool machine conditions
│   ├── exp_19_p1b_all_frames.py     # Master all-frame half-resolution benchmark
│   ├── exp_20_p2_slices.py          # Slice-based MTU packetization
│   └── exp_21_p3_packet_loss.py     # Packet loss resilience & AoI simulation
├── results/
│   ├── phase1/                      # Timing & all-frame benchmark tables
│   ├── phase2/                      # Slicing & overhead distributions
│   ├── phase3/                      # Packet loss, burst fading, and AoI traces
│   └── review_round3/X6/            # Label accuracy & ground-truth verification
├── tests/
│   ├── test_class_mappings.py       # Unit tests for projection consistency
│   └── test_rigorous_codecs.py      # Codec round-trip verification
├── LICENSE                          # MIT Licence
├── NOTICE                           # Third-party code and dataset notices
└── README.md                        # Documentation
```

---

## 7. Setup & Reproduction Guide

### Environment Setup

1. **Python Environment**:
   ```bash
   python -m venv .venv
   # Windows:
   .\.venv\Scripts\activate
   # Linux:
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Compile the Yang et al. Codec**:
   ```bash
   cd baselines/yang2026
   make
   # Or using MinGW-w64 on Windows:
   g++ -std=c++17 -O3 -static source/commonlib/*.cpp source/arithmetic_coder/*.cpp source/arithmetic_coder/third_party/*.cpp source/encoder/*.cpp -I source/commonlib -I source/arithmetic_coder -I source/arithmetic_coder/third_party -I source/encoder -o encoder.exe
   g++ -std=c++17 -O3 -static source/commonlib/*.cpp source/arithmetic_coder/*.cpp source/arithmetic_coder/third_party/*.cpp source/decoder/*.cpp -I source/commonlib -I source/arithmetic_coder -I source/arithmetic_coder/third_party -I source/decoder -o decoder.exe
   cd ../..
   ```

3. **Run Test Suite**:
   ```bash
   pytest -q tests
   ```

### Executing Evaluation Steps

- **Half-Resolution Cache & Benchmark**:
  ```bash
  python experiments/exp_17_cache_pidnet_half.py
  python experiments/exp_19_p1b_all_frames.py
  ```

- **Cool-Machine Hardware Timing Verification**:
  ```bash
  python experiments/exp_18_cool_timing_pass.py
  ```

- **Packet Slicing & Loss Resilience**:
  ```bash
  python experiments/exp_20_p2_slices.py
  python experiments/exp_21_p3_packet_loss.py
  ```

---

## 8. Third-Party Code & Licensing

- **Repository Code**: Released under the [MIT License](LICENSE). Copyright (c) 2026 Pritam Chakraborty, Anjan Bandyopadhyay.
- **Third-Party Components** (see [NOTICE](NOTICE) for complete details):
  - `baselines/yang2026/`: Context-adaptive extended chain coder copyright (c) 2026 InterDigital, licensed under BSD 3-Clause.
  - `third_party/idd_public_code/`: AutoNUE toolkit, copyright (c) AutoNUE, MIT License.
  - `PIDNet`: Model architecture and pretrained weights from Xu et al. (CVPR 2023).
  - Datasets (nuScenes, Cityscapes, IDD) are used strictly under their respective research licenses and are not redistributed.

---

## 9. Citation

If you use this benchmark, code, or experimental data, please cite:

```bibtex
@article{chakraborty2026frame,
  title={Frame-Independent Coding of Semantic Segmentation Maps for {V2X}: Bit Cost, Temporal Redundancy, Packet Loss and Sidelink Capacity},
  author={Chakraborty, Pritam and Bandyopadhyay, Anjan},
  journal={Vehicular Communications},
  year={2026},
  note={Preprint and code: \url{https://github.com/Pritamresearch/Frame-Independent-Coding-of-Semantic-Segmentation-Maps-for-V2X}}
}
```
