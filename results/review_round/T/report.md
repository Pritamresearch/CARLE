# Provenance and Latency Budget Decomposition: $T_{\text{proc}} = 103.4$ ms

## 1. Executive Summary

In Phase 3 and Phase 4 of the CARLE V2 evaluation, the end-to-end multi-camera processing latency $T_{\text{proc}}$ was established as **103.4 ms** for $K = 4$ and **108.7 ms** for $K = 8$.

This report details the exact provenance, empirical measurement source files, hardware configurations, batch sizes, and constituent components that comprise this latency budget.

---

## 2. Latency Budget Decomposition Table

The table below presents the exact breakdown of $T_{\text{proc}}$ across the five core processing stages, extracted directly from the verified Phase 1 and Phase 4 benchmark files:
- `results/phase1/cool_timing_summary.json`
- `results/phase1/pidnet_latency_benchmark.json`
- `results/phase4/p4_latency_budget_detailed.json`

| Stage | Component Description | Latency (K=4) | Latency (K=8) | Hardware Platform | Batch Size | Timed Runs | Source File | Status / Notes |
| :--- | :--- | :---: | :---: | :--- | :---: | :---: | :--- | :--- |
| **Stage 1** | **Camera Ingestion & Preprocessing** | *Excluded / Subsumed* | *Excluded / Subsumed* | Intel Core i7-13620H | 6 | 10 | `p4_latency_budget_detailed.json` | Python/OpenCV baseline measured 42.31 ms; excluded from the 103.4 ms operating point under zero-copy DMA pipeline assumption. |
| **Stage 2** | **Neural Perception (PIDNet-S FP16)** | **97.13 ms** | **97.13 ms** | NVIDIA GeForce RTX 3050 Laptop GPU (Driver 610.91, CUDA 12.4) | 6 | 15 (10 timed + 5 warmup) | `cool_timing_summary.json` | Median GPU execution for all 6 surround cameras ($450 \times 800$ half res). Yields 16.19 ms/cam (61.8 FPS throughput). |
| **Stage 2b** | **Class Projection ($19 \to K$)** | **Included in Stage 2** | **Included in Stage 2** | NVIDIA GeForce RTX 3050 Laptop GPU | 6 | 15 | `cool_timing_summary.json` | Bilinear interpolation of logits + argmax + lookup table executed as part of GPU inference graph. Standalone CPU line item: **NOT FOUND**. |
| **Stage 3** | **Chain Coding (Yang et al. ECC)** | **3.03 ms** | **4.97 ms** | Intel Core i7-13620H (P-cores, up to 4.90 GHz) | 1 frame (4 slices) | 15 | `cool_timing_summary.json` | Median single-frame encode time on dedicated core for half-resolution map. Assumes ideal 6-core parallel encode across cameras. |
| **Stage 4** | **Packetisation (Headers & Slicing)** | **NOT FOUND** | **NOT FOUND** | Intel Core i7-13620H | 24 pkts | — | — | Zero-copy header attachment (12 B per packet $\times$ 24 packets = 288 B). Negligible (<0.05 ms); not logged as separate float metric. |
| **Stage 5** | **V2X Air-Time Transmission** | **2.15 ms** | **4.94 ms** | Dedicated NR-V2X Sidelink (10 Mb/s PHY throughput) | 24 pkts | Analytical | `p4_latency_budget_detailed.json` | $K=4$: 2,691 B total payload (21,528 bits / 10 Mbps = 2.15 ms).<br>$K=8$: 6,178 B total payload (49,424 bits / 10 Mbps = 4.94 ms). |
| **Stage 6** | **Receiver Decoding (Yang ECC)** | **1.07 ms** | **1.62 ms** | Intel Core i7-13620H | 1 frame | 15 | `cool_timing_summary.json` | Median decode time per frame on receiver CPU. |
| **Total** | **End-to-End Latency ($T_{\text{proc}}$)** | **103.38 ms** $\approx$ **103.4 ms** | **108.66 ms** $\approx$ **108.7 ms** | **Heterogeneous Edge (RTX 3050 + i7-13620H)** | **6 surround cams** | — | **Reconciled Phase 1 & 4** | **Exceeds 100 ms target by 3.4% ($K=4$) and 8.7% ($K=8$). Meets 10 Hz throughput under pipelining.** |

---

## 3. Provenance Verification & Observations

1. **Summation Integrity**:
   $$\text{Sum}(K=4) = 97.13\text{ ms (GPU)} + 3.03\text{ ms (Encode)} + 2.15\text{ ms (Air-Time)} + 1.07\text{ ms (Decode)} = 103.38\text{ ms} \to \mathbf{103.4\text{ ms}}$$
   $$\text{Sum}(K=8) = 97.13\text{ ms (GPU)} + 4.97\text{ ms (Encode)} + 4.94\text{ ms (Air-Time)} + 1.62\text{ ms (Decode)} = 108.66\text{ ms} \to \mathbf{108.7\text{ ms}}$$
2. **Missing Components (NOT FOUND)**:
   - **Standalone Class Projection**: The projection from 19 Cityscapes logits to $K=4$ and $K=8$ classes is integrated directly into the PyTorch inference loop on the GPU (bilinear interpolation followed by argmax). No isolated CPU-side projection kernel timing is recorded in `cool_timing_summary.json`.
   - **Packetisation Latency**: The overhead of prefixing each slice with a 12-byte header (`frame_id`, `slice_id`, `pkt_idx`, `pkt_count`, `crc32`) is an in-memory pointer offset operation and is not reported as an independent timing metric in any Phase 1 or Phase 4 summary file.
3. **Operating Point Realism**:
   - The 103.4 ms operating point assumes that camera ingestion/preprocessing is either performed via zero-copy hardware DMA or pipelined into the previous frame's acquisition interval. If preprocessing (measured at 42.31 ms in Python) is included sequentially without DMA acceleration, total latency increases to 145.7 ms.
   - The compression stage assumes ideal multi-core parallelism (each of the 6 camera frames encoded concurrently on an independent P-core in 3.03 ms). When implemented with Python `ProcessPoolExecutor`, inter-process serialization overhead inflated wall-clock compression time to 372.74 ms.
4. **Paper Verdict**:
   - **End-to-End Latency**: **NO** for strict 100 ms budget (exceeds budget by 3.4% at $K=4$ and 8.7% at $K=8$).
   - **Throughput**: **YES** for 10 Hz pipelined throughput (bottleneck stage is GPU perception at 97.13 ms < 100 ms, achieving 10.3 Hz pipelined frame rate).
