# Experiment E1 Report: Inter-Frame Contour and SemCM Extended Context Evaluation

## 1. Executive Summary & Core Findings
- **Generalizability of the "Temporal Context Saves $\le 2.4\%$" Thesis**:
  - The finding that temporal redundancy exploitation yields marginal gains is **NOT specific to our SemCM pixel-context coder**.
  - On the exact same 6,000 nuScenes sweep frames (60 streams $\times$ 100 frames), our faithful inter-frame extension of the Yang et al. extended chain coder (ECC) achieves only:
    - **K = 4**: Intra ECC = 2,648.7 bpf $\to$ Inter Contour = 2,631.0 bpf $\implies$ **0.67% saving [95% CI: 0.59%, 0.76%]**.
    - **K = 8**: Intra ECC = 5,440.6 bpf $\to$ Inter Contour = 5,393.5 bpf $\implies$ **0.87% saving [95% CI: 0.81%, 0.93%]**.
  - Lossless decode verification succeeded on **12,000 out of 12,000 frames (100.0% bit-exact parity)**.
- **Extra SemCM Temporal Contexts**:
  - Adding the pixel label two frames back $F(t-2)$ to the context expands the model to 1,024 contexts, yielding a **negative saving of -14.48% [95% CI: -16.11%, -12.59%]** (a 14.5% bitrate penalty).
  - Adding a binary flag indicating whether the pixel changed in the prior frame ($\Delta_{t-1}$) expands the model to 512 contexts, yielding a **negative saving of -11.74% [95% CI: -12.23%, -11.13%]**.
  - In both cases, expanding the conditioning state space causes severe **context dilution** (starvation), where counts are spread across rarely visited states and adaptation overhead dominates any small entropy reduction.
- **Context Dilution Sweep**:
  - Training on 1 scene: 7,545.2 bpf.
  - Training on 5 scenes (2-fold leave-5-scenes-out): 6,808.1 bpf.
  - Training on 9 scenes (leave-one-scene-out cross-validation): 6,668.6 bpf.
  - **20 scenes per fold**: **NOT RUN** because the nuScenes mini dataset contains exactly 10 scenes in total.

---

## 2. Quantitative Results Table (All 6,000 Sweep Frames, 10 Scenes)

Bootstrap intervals computed via 10,000 scene-cluster resamples over the 10 nuScenes scenes (seed 0).

| Model Family | Conditioning Context | K | Baseline Bits/Frame | Variant Bits/Frame | Mean Saving (%) | 95% Bootstrap CI | Status / Verification |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Contour Coder (Yang ECC Extension)** | Intra-only vs Inter-frame Contour Reuse | 4 | 2,648.7 | 2,631.0 | **+0.67%** | [0.59%, 0.76%] | Bit-exact (6,000/6,000) |
| **Contour Coder (Yang ECC Extension)** | Intra-only vs Inter-frame Contour Reuse | 8 | 5,440.6 | 5,393.5 | **+0.87%** | [0.81%, 0.93%] | Bit-exact (6,000/6,000) |
| **SemCM Pixel-Context** | Intra Spatial (256 states) | 4 | 6,161.8 | — | Baseline | — | Bit-exact |
| **SemCM Pixel-Context** | Spatial + $F(t-2)$ Label (1024 states) | 4 | 6,161.8 | 7,056.5 | **-14.48%** | [-16.11%, -12.59%] | Bit-exact |
| **SemCM Pixel-Context** | Spatial + Change Flag $\Delta_{t-1}$ (512 states)| 4 | 6,161.8 | 6,885.3 | **-11.74%** | [-12.23%, -11.13%] | Bit-exact |

---

## 3. Public Code and Implementation Statement
- **Public Code Search**: We searched public repositories and literature for the inter-frame predictive contour coder of Yang et al. (arXiv:2610.40887) and CC-SMC (Yang et al. 2024). The official InterDigital repository (`LosslessSegmentationMapCompression`) only contains the intra-only contour coder.
- **Implementation Notice**: The inter-frame contour coder evaluated here is our faithful, independent extension of the Yang ECC coder that performs contour reuse, displacement estimation, and rate-distortion mode selection. It is strictly our implementation and we do not claim it reproduces the proprietary codebase of Yang et al.

---

## 4. Context Dilution Analysis
Evaluating the effect of training set size on the extended $F(t-2)$ temporal model on `scene-0061`:

| Training Scenes Count | Training Scenes Set | Test Scene | Test Bitrate (bpf) | Notes |
| :---: | :--- | :---: | :---: | :--- |
| **1 scene** | `scene-0103` | `scene-0061` | 7,545.2 | Severe under-fitting / sparse prior counts |
| **5 scenes** | `scene-0103` .. `scene-0796` | `scene-0061` | 6,808.1 | 9.8% bitrate reduction from larger training sample |
| **9 scenes** | All except `scene-0061` (LOO) | `scene-0061` | 6,668.6 | 2.1% further reduction; still 8.2% worse than intra |
| **20 scenes** | N/A | `scene-0061` | **NOT RUN** | **nuScenes mini contains only 10 scenes in total** |

### Takeaway
Even with 9 scenes (the maximum possible training split in nuScenes mini), the temporal context model suffers an 8.2% to 14.5% penalty relative to the compact intra-only spatial model, confirming that the high dimensionality of temporal conditioning states leads to persistent context dilution on autonomous driving datasets.
