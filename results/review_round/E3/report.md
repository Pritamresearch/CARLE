# Experiment E3 Report: Comprehensive Lossless Baseline Codec Evaluation

## 1. Executive Summary & Headline Statement
- **Paper Statement Verification**:
  - The paper statement that **"chain coding is 2.7 times smaller than JPEG-XL"** is **CONFIRMED**:
    At $K = 4$, Yang et al. ECC achieves 2,648.7 bits/frame compared to 7,030.6 bits/frame for JPEG-XL Effort 9, yielding an empirical ratio of **2.656 [95% CI: 2.580, 2.734]** (approximately $2.7\times$).
- **Comparison across Codec Paradigms**:
  - **Chain/Contour coding** dramatically outperforms general-purpose image formats and run-length coders on discrete label maps:
    - Extended Chain Coding (Yang et al.) is **$2.66\times$ more compact than JPEG-XL e9** at $K=4$ and **$2.34\times$ more compact** at $K=8$.
    - Extended Chain Coding is **$3.48\times$ more compact than WebP Lossless** at $K=4$ and **$3.45\times$ more compact** at $K=8$.
    - Extended Chain Coding is **$4.41\times$ more compact than Simple RLE + Arithmetic Coding** at $K=4$ and **$4.51\times$ more compact** at $K=8$.
  - Our SemCM pixel-context arithmetic coder represents an intermediate operating point ($1.72\times$ and $1.96\times$ larger than ECC at $K=4$ and $K=8$), but significantly outperforms JPEG-XL and WebP while supporting zero-overhead byte-level packet slicing.
- **Decode Verification**:
  Lossless decode verification was confirmed bit-exact on **12,000 out of 12,000 frames (100.0%)** for the newly evaluated RLE + AC baseline.

---

## 2. Lossless Baseline Comparison Table (All 6,000 nuScenes Sweep Frames)

Evaluated across all 60 camera streams $\times$ 100 frames across 10 scenes at half resolution (450 $\times$ 800).
Ratios are computed relative to the Yang et al. 2026 ECC baseline, with 10,000 scene-cluster bootstrap 95% confidence intervals (seed 0).

| Class Count ($K$) | Codec / Paradigm | Status | Mean Bits/Frame (bpf) | Effective 1-Cam 10 Hz (kb/s) | Ratio to Yang ECC | 95% Scene-Cluster Bootstrap CI |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **K = 4** | **Yang et al. 2026 (ECC)** | **AVAILABLE** | **2,648.7** | **26.49** | **1.000** | [1.000, 1.000] |
| $K = 4$ | SemCM (a) [Pixel-Context Intra] | AVAILABLE | 4,554.1 | 45.54 | 1.722 | [1.662, 1.789] |
| $K = 4$ | JPEG-XL Effort 9 (FLIF successor)| AVAILABLE | 7,030.6 | 70.31 | 2.656 | [2.580, 2.734] |
| $K = 4$ | WebP Lossless | AVAILABLE | 9,220.8 | 92.21 | 3.483 | [3.395, 3.561] |
| $K = 4$ | Simple Run-Length + AC (RLE+AC) | AVAILABLE | 11,672.2 | 116.72 | 4.407 | [4.296, 4.504] |
| $K = 4$ | JBIG2 (Generic Region Coding) | NOT RUN | — | — | — | No JBIG2 library/binary on system |
| $K = 4$ | SCM (Yang et al. prior work) | NOT RUN | — | — | — | No public codebase released |
| $K = 4$ | Learned Lossless Label Coder | NOT RUN | — | — | — | No public trained weights/code |
| **K = 8** | **Yang et al. 2026 (ECC)** | **AVAILABLE** | **5,440.6** | **54.41** | **1.000** | [1.000, 1.000] |
| $K = 8$ | SemCM (a) [Pixel-Context Intra] | AVAILABLE | 10,640.5 | 106.41 | 1.957 | [1.909, 2.008] |
| $K = 8$ | JPEG-XL Effort 9 (FLIF successor)| AVAILABLE | 12,744.4 | 127.44 | 2.344 | [2.250, 2.457] |
| $K = 8$ | WebP Lossless | AVAILABLE | 18,759.9 | 187.60 | 3.451 | [3.366, 3.543] |
| $K = 8$ | Simple Run-Length + AC (RLE+AC) | AVAILABLE | 24,537.5 | 245.38 | 4.512 | [4.424, 4.597] |
| $K = 8$ | JBIG2 (Generic Region Coding) | NOT RUN | — | — | — | No JBIG2 library/binary on system |
| $K = 8$ | SCM (Yang et al. prior work) | NOT RUN | — | — | — | No public codebase released |
| $K = 8$ | Learned Lossless Label Coder | NOT RUN | — | — | — | No public trained weights/code |

---

## 3. Detailed Audit of Skipped/Unrun Baselines
In accordance with Hard Rule 1:
1. **JBIG2**: Neither the standalone CLI (`jbig2`, `jbig2enc`) nor Python bindings (`pyjbig2`, `jbig2dec`) are installed in the workspace environment.
2. **SCM (Semantic Map Coder)**: SCM was Yang et al.'s earlier contour format before the 2026 ECC paper; no public standalone repository exists for SCM.
3. **Learned Lossless Coders**: Deep learning lossless image compression frameworks (e.g. L3C, Bit-Swap, CompressAI) do not provide pre-trained models or categorical entropy models adapted to 4-class or 8-class discrete semantic label maps.
