# Experiment E4 Report: 3GPP NR-V2X Sidelink Capacity Sensitivity

## 1. Executive Summary & Headline Statement
- **Baseline Reproduction**:
  The baseline configuration ($\rho = 0.60$, MCS 16 with TBS = 309 Bytes, Header = 12 Bytes, $f = 10$ Hz) evaluates on all 6,000 nuScenes sweep frames and **strictly reproduces 367 camera streams (61 vehicles)** for the chain coder [95% CI: 339, 401] and **249 camera streams (41 vehicles)** for SemCM (a) [95% CI: 227, 274].
- **Which Factor Moves the Result Most?**:
  **Frame rate ($f$) and channel congestion limit ($\rho$) dominate capacity scaling, while packet header overhead has minimal impact.** Halving the transmission rate from 10 Hz to 5 Hz produces an exact $+100.0\%$ doubling in capacity (from 367 to 734 streams). Among wireless channel and PHY parameters, the channel congestion ratio ($\rho$) exerts a strictly proportional linear influence ($\pm 50.0\%$ swing from 183 streams at $\rho=0.30$ to 550 streams at $\rho=0.90$). MCS selection shifts capacity across a 61.0% dynamic range (from 223 streams at robust QPSK MCS 10 up to 447 streams at 64QAM MCS 20), whereas tripling the packet header size from 12 B to 40 B reduces capacity by only $-5.4\%$ (from 367 to 347 streams), because the 309-byte transport block quantization absorbs small header increases for most frames.

---

## 2. One-At-A-Time (OAT) Sensitivity Table Around Baseline

Evaluated across all 6,000 nuScenes sweep frames (10 scenes). Baseline is highlighted: $\rho = 0.60$, MCS 16 (TBS = 309 B), Header = 12 B, Frame Rate = 10 Hz.

### Yang et al. ECC (Chain Coder, Whole-Frame S=1)

| Parameter Swept | Parameter Value | TBS (Bytes) | Subchannels/Frame ($E[\lceil B/TBS \rceil]$) | Supported Cameras ($N_{\text{max}}$) | 95% Scene-Cluster Bootstrap CI | Delta vs Baseline (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | — | **309 B** | **1.632** | **367** | **[339, 401]** | **0.0%** |
| $\rho$ (CBR Limit) | 0.30 | 309 B | 1.632 | 183 | [169, 200] | -50.1% |
| $\rho$ (CBR Limit) | 0.45 | 309 B | 1.632 | 275 | [253, 301] | -25.1% |
| $\rho$ (CBR Limit) | **0.60 (Base)** | 309 B | 1.632 | **367** | [339, 401] | **0.0%** |
| $\rho$ (CBR Limit) | 0.75 | 309 B | 1.632 | 458 | [422, 501] | +24.8% |
| $\rho$ (CBR Limit) | 0.90 | 309 B | 1.632 | 550 | [508, 602] | +49.9% |
| MCS Index | 10 (QPSK R=658) | 157 B | 2.684 | 223 | [200, 251] | -39.2% |
| MCS Index | 13 (16QAM R=490)| 233 B | 1.993 | 301 | [273, 335] | -18.0% |
| MCS Index | **16 (16QAM R=658)**| 309 B | 1.632 | **367** | [339, 401] | **0.0%** |
| MCS Index | 19 (64QAM R=517)| 372 B | 1.438 | 417 | [386, 454] | +13.6% |
| MCS Index | 20 (64QAM R=567)| 405 B | 1.339 | 447 | [414, 487] | +21.8% |
| Header Overhead | **12 Bytes (Base)**| 309 B | 1.632 | **367** | [339, 401] | **0.0%** |
| Header Overhead | 24 Bytes | 309 B | 1.674 | 358 | [331, 393] | -2.5% |
| Header Overhead | 40 Bytes | 309 B | 1.728 | 347 | [320, 381] | -5.4% |
| Frame Rate ($f$) | 5.0 Hz | 309 B | 1.632 | 734 | [676, 804] | +100.0% |
| Frame Rate ($f$) | **10.0 Hz (Base)**| 309 B | 1.632 | **367** | [339, 401] | **0.0%** |

---

### SemCM (a) (Pixel-Context Arithmetic Coder, Whole-Frame S=1)

| Parameter Swept | Parameter Value | TBS (Bytes) | Subchannels/Frame ($E[\lceil B/TBS \rceil]$) | Supported Cameras ($N_{\text{max}}$) | 95% Scene-Cluster Bootstrap CI | Delta vs Baseline (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | — | **309 B** | **2.402** | **249** | **[227, 274]** | **0.0%** |
| $\rho$ (CBR Limit) | 0.30 | 309 B | 2.402 | 124 | [113, 137] | -50.2% |
| $\rho$ (CBR Limit) | 0.45 | 309 B | 2.402 | 187 | [170, 205] | -24.9% |
| $\rho$ (CBR Limit) | **0.60 (Base)** | 309 B | 2.402 | **249** | [227, 274] | **0.0%** |
| $\rho$ (CBR Limit) | 0.75 | 309 B | 2.402 | 312 | [284, 342] | +25.3% |
| $\rho$ (CBR Limit) | 0.90 | 309 B | 2.402 | 374 | [341, 411] | +50.2% |
| MCS Index | 10 (QPSK R=658) | 157 B | 4.148 | 144 | [131, 160] | -42.2% |
| MCS Index | 13 (16QAM R=490)| 233 B | 2.978 | 201 | [183, 222] | -19.3% |
| MCS Index | **16 (16QAM R=658)**| 309 B | 2.402 | **249** | [227, 274] | **0.0%** |
| MCS Index | 19 (64QAM R=517)| 372 B | 2.062 | 290 | [264, 320] | +16.5% |
| MCS Index | 20 (64QAM R=567)| 405 B | 1.944 | 308 | [280, 339] | +23.7% |
| Header Overhead | **12 Bytes (Base)**| 309 B | 2.402 | **249** | [227, 274] | **0.0%** |
| Header Overhead | 24 Bytes | 309 B | 2.441 | 245 | [224, 270] | -1.6% |
| Header Overhead | 40 Bytes | 309 B | 2.493 | 240 | [219, 264] | -3.6% |
| Frame Rate ($f$) | 5.0 Hz | 309 B | 2.402 | 499 | [455, 548] | +100.4% |
| Frame Rate ($f$) | **10.0 Hz (Base)**| 309 B | 2.402 | **249** | [227, 274] | **0.0%** |

---

## 3. Full Grid Reference
The complete $5 \times 5 \times 3 \times 2 = 150$ grid combinations for both codecs (300 rows total) are saved in [capacity_grid.csv](file:///c:/PhD/CARLE%20V2/results/review_round/E4/capacity_grid.csv).
