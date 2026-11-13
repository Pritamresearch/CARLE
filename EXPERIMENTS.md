# CARLE journal-upgrade experiments — instructions for Claude Code

> How to use: put this file in the root of the `carle-v2x` repo as `EXPERIMENTS.md`, then start Claude Code in that folder and say:
> **"Read EXPERIMENTS.md and execute Phase S (setup). My datasets are in: <folders>. Stop at the end of each phase and report."**
> Run one phase per session. Fill in the `<<...>>` placeholders first.

---

## 0. Your role and ground rules (read before doing anything)

You are a research engineer re-running and extending the experiments for the paper
"Semantic-Domain Differential Compression for Real-Time V2X Communication (CARLE)".
The goal is a Q1 journal version (IEEE TVT / T-ITS / IoT-J), with an optional later
Nature-Communications-level extension (Phase 5–6).

Non-negotiable rules:

1. **Never invent, estimate, or copy numbers from other papers into result tables.** Every number
   in `results/` must be produced by code in this repo, from a logged run, with a saved config.
   If something cannot be run (missing data, missing hardware), write `NOT_RUN` and the reason.
2. **Every experiment is a script + YAML config + output CSV + log.** No numbers typed by hand.
   Layout:
   ```
   carle/            # library code (encoders, decoders, metrics)
   baselines/        # baseline codecs
   experiments/     # one script per experiment, e.g. exp_01_bit_accounting.py
   configs/          # YAML per experiment
   results/<exp_id>/ # results.csv, summary.json, log.txt, env.json, figures/
   tests/            # pytest
   ```
3. **Losslessness is tested, not assumed.** For every codec and every frame: decode and assert
   exact equality with the input label map (`np.array_equal`). Count failures; any failure is a bug.
4. **Fair comparisons.** All codecs get the *same input* (same label maps, same class set) and
   timing is measured the *same way*: encode-only and decode-only times, separately from
   segmentation inference. Report inference time in its own column.
5. **Aggregate correctly.** Compression ratio = Σ raw_bits / Σ sent_bits over the dataset
   (not the mean of per-frame ratios). Also report the per-frame median and the 5th and 95th percentiles of
   bits/frame, and 95% bootstrap CIs (resample by *scene/sequence*, not by frame).
6. **Record the environment** in `env.json`: CPU model, cores, RAM, OS, Python and library versions,
   GPU/NPU if used, number of threads. Pin threads (`OMP_NUM_THREADS`, etc.) for timing.
   Timing: warm-up 10 runs, then median of ≥5 repeats per frame using `time.perf_counter_ns`.
7. **Use a fixed seed** (`seed: 0`, plus seeds 1–4 for anything stochastic) and log it.
8. **Stop and ask me** when: a dataset path is missing, a result contradicts the paper by >10%,
   a lossless check fails, or a design decision is ambiguous. Don't silently pick.
9. Keep a running `results/LAB_NOTEBOOK.md`: date, what was run, command, outcome, surprises.
10. Write pytest tests for every encoder/decoder (random maps, all-zero diff, all-changed diff,
    single pixel, max-length runs, non-square images). `pytest -q` must pass before any phase ends.

Environment:
- Python 3.11, numpy, numba (for fast loops), opencv-python, pillow, zstandard, lz4,
  python-snappy, `constriction` (rANS/arithmetic coding), pandas, matplotlib, scipy, pyyaml,
  tqdm, pytest, ultralytics (YOLO11-seg), torch/torchvision, nuscenes-devkit.
- ffmpeg with libx265 and FFV1 (for lossless video baselines).
- Install with pip `--break-system-packages` if needed; record versions.

Data locations (fill in):
- nuScenes full or mini: `<<NUSCENES_ROOT>>` (version `<<v1.0-mini | v1.0-trainval>>`)
- Field-test videos: `<<FIELD_VIDEOS_DIR>>`
- Cityscapes: `<<CITYSCAPES_ROOT>>` (need `gtFine`, and `leftImg8bit_sequence` if available)
- Optional later: OPV2V `<<OPV2V_ROOT>>`, V2X-Sim `<<V2XSIM_ROOT>>`, DAIR-V2X `<<DAIR_ROOT>>`
- Hardware for timing: `<<e.g. Intel i7-12700 / Apple M2 / Jetson Orin>>`

---

## Phase S — Setup from an empty folder (run this first)

Situation: Claude Code is started in an **empty folder**. Code is on GitHub; datasets already
exist elsewhere on this machine. Rules: **never copy, move, modify or delete anything in the
dataset folders** — read-only access via symlinks only. Never download large datasets without asking.

S.1 Project skeleton
- `git init`; create the folder layout from Section 0; create a Python venv `.venv`
  (or conda env `carle`) and `requirements.txt` with pinned versions; install them.
- Check `ffmpeg -version` (needs libx265 and ffv1). If missing, tell me the install command for my OS;
  don't run system-level installers without asking.

S.2 Get the original code
- `git clone https://github.com/Pritamresearch/carle-v2x.git legacy/` (read-only reference;
  do not edit files in `legacy/`). If the clone fails, stop and ask me.
- Write `legacy/INVENTORY.md`: list every script, what it does, its inputs/outputs, and which paper
  table/figure it seems to produce.

S.3 Locate the datasets I already have
- I will give you candidate folders: `<<e.g. ~/datasets, D:/research/data, /media/pritam/HDD>>`.
  Search **only inside those** (max depth 4) for signatures:
  - nuScenes: a folder containing `samples/`, `sweeps/` and `v1.0-mini/` or `v1.0-trainval/`
  - Cityscapes: `gtFine/`, `leftImg8bit/`, optionally `leftImg8bit_sequence/`, `gtFinePanopticParts/`
  - Field-test videos: `*.mp4/*.avi/*.mov` (list names, duration, fps, resolution via ffprobe)
  - Any cached label maps / `.npz` / `.npy` produced by the old code
  - Optional: OPV2V, V2X-Sim, DAIR-V2X, Waymo, Argoverse 2
- Report a table: dataset, path, version/split, number of scenes/frames, size on disk, what is
  missing (e.g. sweeps not downloaded, no `leftImg8bit_sequence`). **Stop and show me this table.**

S.4 Link, don't copy
- Create `data/` with symlinks (Windows: directory junctions `mklink /J`) pointing to the found
  folders, e.g. `data/nuscenes -> /path/to/nuscenes`. Record them in `configs/paths.yaml`;
  every script must read paths only from this file. Add `data/` and `results/**/cache/` to `.gitignore`.

S.5 What to do about missing data
- If something needed is missing, list it with: official download page, size, whether a login/
  licence is required (nuScenes, Cityscapes, Waymo require registration — I must download those
  myself), and which experiments are blocked without it. Continue with experiments that are possible.

S.6 Smoke test
- Load 5 frames from each available dataset, run YOLO11n-seg, map to the 4 classes, save overlays
  to `results/setup/`, and time it. Run `pytest -q` on a trivial test. Commit: "Project setup".

Then proceed to Phase 0 using `legacy/` as the code being audited.

---

## Phase 0 — Audit and reproduce the current paper (no new methods yet)

Goal: find out which numbers in the current paper are actually right.

0.1 Read the existing code. Write `results/phase0/AUDIT.md` covering:
   - how the raw size (B_raw) is computed (8 bits/pixel or log2(C) bits/pixel?)
   - how bits/frame for CARLE is computed (what fields, how many bits each, are they actually
     serialized to bytes or only estimated by formula?)
   - what the latency number includes (segmentation inference? I/O? Python loops?)
   - how frame pairs are formed for nuScenes, field test and Cityscapes
   - how "Overall" CR is aggregated
   - where the DVC/UCVC numbers in Table 3 came from (if not produced by code, flag them for removal)
0.2 **Cityscapes pairing check.** Print 20 example pairs (file names, city, sequence id, frame id).
   Cityscapes `gtFine` annotates one frame per 30-frame snippet, so consecutive gtFine files
   are *not* temporally adjacent. Determine whether the 0.007% δpix comes from (a) comparing an
   image to itself, (b) panoptic vs part labels of the same image, (c) something else.
   Save visual overlays of 5 pairs. Report to me before continuing.
0.3 **Frame-rate check.** Report the time gap between frames used for nuScenes (keyframes are 2 Hz;
   sweeps are ~12 Hz) and field-test videos (native fps and any subsampling).
0.4 Reproduce Tables 1, 3, 5, 6 exactly as the paper defines them and put old-vs-reproduced
   side by side in `results/phase0/reproduction.csv`, with columns
   `table,row,metric,paper_value,reproduced_value,rel_diff`.
0.5 Explain these known contradictions with evidence:
   - Frame-wise RLE (570.2× / 458.8×) beats CARLE (265.1×) in bits/frame and latency.
   - "Change-mask + RLE" and "CARLE (Full)" are identical in Table 1.
   - CARLE memory 0.0052 MB (Table 1) vs 2.75 MB (Table 3).
   - Frame counts 2,424 vs 2,364; 1,073 vs 1,069 vs 547.

Deliverable: `AUDIT.md`, `reproduction.csv`, overlays. **Stop and report.**

---

## Phase 1 — Correct, fair benchmark (core of the Q1 revision)

### 1.1 Canonical data pipeline
- `experiments/exp_01_build_labelmaps.py`: run segmentation once per frame, map to the
  4-class set {1 road, 2 vehicle, 3 pedestrian, 4 other} **and** to an extended safety set
  {road, vehicle, pedestrian, cyclist/two-wheeler, animal, traffic-sign/light, other}.
  Cache label maps as `uint8` `.npz` with metadata (dataset, scene, camera, timestamp, model).
- nuScenes: produce sequences at **2 Hz (keyframes) and ~12 Hz (sweeps)**. Field test: native fps
  and 10 Hz. Cityscapes: if `leftImg8bit_sequence` exists, segment the 30-frame snippets with the
  model to get real temporal pairs; otherwise **drop the Cityscapes temporal experiment** and use
  gtFine only for single-frame (intra) coding tests.
- Log segmentation inference time separately.

### 1.2 Bit accounting — real bitstreams only
Every codec must produce an actual `bytes` object; bits = 8·len(bytes) + any header.
Raw reference sizes to report (all three, clearly labeled):
- `raw8` = H·W·8 bits, `rawlog` = H·W·ceil(log2 C) bits, `png` = lossless PNG of the label map.
The headline CR in the paper must use **`rawlog` and PNG** as references, not `raw8`.

### 1.3 Codecs to implement/compare (all lossless on the label map)
Intra (single frame):
- I1 Frame-wise RLE (fixed-width fields)
- I2 PNG (OpenCV, compression 9), I3 zstd (level 3 and 19) on packed bytes, I4 LZ4, I5 Snappy
- I6 FFV1 / x265-lossless per frame (gray8) via ffmpeg
Inter (temporal):
- T1 CARLE-v1 exactly as in the paper (absolute start index, fixed widths)
- T2 Change-map + zstd, T3 Change-map + LZ4
- T4 Delta (F_{t+1} − F_t mod C) + zstd   ← a *fair* delta baseline, not raw delta
- T5 x265 lossless / FFV1 on the label-map video (GOP 30)
- **T6 CARLE-v2 (new):** gap-coded starts (distance from end of previous run) + run lengths +
  class, all entropy-coded with adaptive rANS (`constriction`), with contexts:
  class conditioned on previous label at that pixel; lengths/gaps with adaptive Golomb or
  binarized adaptive models.
- **T7 CARLE-v2-IP (new):** per frame (and optionally per 64×64 tile) pick intra or inter coding by
  actual byte size; 1 mode bit per frame/tile; forced intra refresh every N frames (N in {10, 30, ∞}).
Implement hot loops with numba or vectorised numpy. No pure-Python per-pixel loops.

### 1.4 Metrics per codec × dataset × frame rate
bits/frame (mean, median, p5, p95), CR vs `rawlog` and vs PNG (sum-ratio), encode ms, decode ms
(median and p95, plus % frames < 100 ms with inference included and excluded), peak memory
(`tracemalloc` + RSS), lossless-check failures (must be 0), 95% bootstrap CIs over scenes.

### 1.5 Ablations (replaces Table 1)
- with/without 4-class reduction (also 7-class safety set and full model classes)
- absolute vs gap-coded starts
- fixed-width vs entropy-coded fields
- inter-only vs intra/inter switching vs + periodic refresh
- frame rate 2 / 5 / 10 / 12 Hz
- segmentation backbone (YOLO11n/l-seg, YOLOv8n/s-seg, LRASPP-MBv3, DeepLabV3-MBv3), all on the
  same hardware path; also report label "flicker rate" (fraction of pixels that flip and flip
  back within 3 frames) per backbone to explain CR differences.

### 1.6 Information-theoretic reference
`experiments/exp_06_entropy.py`: estimate the empirical conditional entropy
H(F_{t+1}(i,j) | F_t(i,j), causal neighbours of F_{t+1}) with count-based context models
(train on half the scenes, evaluate cross-entropy on the other half). Report it as bits/frame
and plot each codec's rate relative to this estimate (gap in %). Use it to replace the
"CR ∝ 1/δ (no fit)" figure. Fit and report the δpix vs bits/frame relationship with R² and CIs.

### 1.7 Figures (matplotlib, vector PDF, consistent colours, readable at column width)
- rate vs δpix (log–log) per dataset with entropy reference
- bits/frame CDF per codec
- latency CDF (encode only; and inference + encode)
- per-camera and per-scene bars with CIs
- time series of bits/frame on the densest field-test video

Deliverable: `results/phase1/*` + `results/phase1/SUMMARY.md` stating plainly which codec wins
under which condition (including if CARLE-v1 does not win). **Stop and report.**

---

## Phase 2 — Robustness to packet loss (required for comms journals)

2.1 Packetisation: split each frame bitstream into packets of ≤ `<<MTU, e.g. 1,200>>` bytes with a
   header (frame id, map version, packet idx/count, intra/inter flag, CRC32).
2.2 Channel models: Bernoulli loss p ∈ {0, 0.01, 0.05, 0.1, 0.2} and Gilbert–Elliott bursty loss
   (two parameter sets). Seeds 0–4.
2.3 Receiver policies: (a) apply-what-arrives, (b) drop frame on any missing packet and wait for
   refresh, (c) NACK-based intra refresh request with RTT `<<e.g. 20 ms>>`.
2.4 Metrics over time: receiver-map mIoU and pixel accuracy vs the sender's true map, per-class IoU
   for pedestrian/vehicle, time-to-recover after a loss, effective bitrate including refreshes.
2.5 Sweep refresh period N and show the bitrate–robustness trade-off curve.
Deliverable: `results/phase2/*`. **Stop and report.**

---

## Phase 3 — Bitrate and channel feasibility

3.1 Convert bits/frame to Mb/s at 10, 20, 30 Hz for 1 camera and 6 cameras.
3.2 Simple analytic NR-V2X sidelink capacity model (state all assumptions: bandwidth 10/20 MHz,
   MCS range, numerology, resource reuse): how many vehicles can share their streams per channel
   at a 100 ms deadline, for each codec. Put the assumptions in a table.
3.3 Optional, if I confirm: packet-level simulation with WiLabV2Xsim (MATLAB) or ns-3 5G-LENA +
   SUMO traces, 10–200 vehicles. Ask me before installing these.
Deliverable: `results/phase3/*`. **Stop and report.**

---

## Phase 4 — Downstream usefulness (BEV) — ask before starting

4.1 Check which BEV-ready cooperative datasets are available locally (OPV2V, V2X-Sim, DAIR-V2X).
4.2 Generate BEV semantic grids (e.g. 0.4 m cells, 100×100 m) per agent; apply ego-motion
   compensation (warp previous grid by relative pose) before differencing; encode with T1, T6, T7.
4.3 At the receiver, transform into the ego frame and fuse; measure BEV IoU (vehicle,
   pedestrian, drivable area) vs bandwidth, with and without loss (Phase 2 channel).
4.4 Compare against at least one reproduced feature-sharing baseline (Where2comm via OpenCOOD or
   its official repo) on the same split; report AP/IoU vs bytes per agent.

---

## Phase 5–6 — Nature-Communications-level extension (only after Phases 0–4 are solid)

5. **Scaling study:** run the Phase 1 pipeline on large datasets (full nuScenes, Waymo Open,
   Argoverse 2, field data from Indian roads) at 10 Hz; estimate the semantic conditional entropy
   rate per vehicle as a function of speed, traffic density, road type, city, and country;
   fit a model with CIs; test generalisation by holding out cities and countries.
6. **Closed-loop safety (CARLA + SUMO):** occluded-pedestrian scenarios; compare
   no-cooperation / CARLE-v2-IP / feature-sharing baseline under bandwidth caps and loss;
   report collision rate, minimum time-to-collision, with ≥ 30 seeds per condition.
   (Real C-V2X radio testbed is done by the authors; provide a logging tool `tools/testbed_logger.py`
   that records send/receive timestamps, packet loss and map accuracy.)

---

## Final deliverables (end of each phase)
- `results/<phase>/SUMMARY.md`: what was run, key table (generated from CSV), surprises,
  and anything that weakens the paper's claims, stated plainly.
- `paper_tables/*.tex` generated automatically from CSVs (no hand-typed numbers).
- `REPRODUCE.md`: exact commands to regenerate every table and figure from scratch.
- All tests passing; `git commit` per completed experiment with a descriptive message.
