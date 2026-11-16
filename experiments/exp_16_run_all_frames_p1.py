"""
Phase 1: Comprehensive Master Benchmark on ALL Cached Frames
Codecs:
1. Yang et al. 2026 (arXiv:2603.03073)
2. SemCM (a) [Frame-Indep, 2-fold prior w=16384] (Frozen Reference)
3. JPEG-XL (Effort 9)
4. WebP Lossless
5. PNG (OpenCV level 9)

Datasets:
- nuScenes Keyframes (2 Hz, all 2,424 frames)
- nuScenes Sweeps (12 Hz, all 6,000 frames)
- Field Native Unflagged (all 3,673 frames: bolpur, malaysia, sikkim, dhaka_day, dhaka_night)
- Field 10 Hz Unflagged (all 1,399 frames: bolpur, malaysia, sikkim, dhaka_day, dhaka_night)
- Flagged Clip: Bhubaneswar Night (Native: 1,811 frames, 10 Hz: 604 frames) evaluated separately.

Clustered 95% bootstrap CIs by scene/clip.
Hardware strictly loaded from results/phase0/env.json:
- CPU: Intel Core i7-13620H
- GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import json
import io
import cv2
import subprocess
import numpy as np
import pandas as pd
from PIL import Image
import imagecodecs
from collections import defaultdict
import concurrent.futures

# Enforce single-thread CPU protocol
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

YANG_DIR = Path("baselines/yang2026").resolve()
YANG_ENC = str(YANG_DIR / "encoder.exe")

CACHE_DIR = Path("results/phase1/cache_pidnet")
SWEEPS_DIR = CACHE_DIR / "sweeps"

FOLD1_SCENES = {"scene-0061", "scene-0103", "scene-0553", "scene-0655", "scene-0757"}
FOLD2_SCENES = {"scene-0796", "scene-0916", "scene-1077", "scene-1094", "scene-1100"}
FOLD1_CLIPS = {"india_bolpur", "malaysia", "dhaka_day"}
FOLD2_CLIPS = {"sikkim", "dhaka_night"}

def get_stream_fold(stream_name: str) -> int:
    for sc in FOLD1_SCENES:
        if sc in stream_name: return 1
    for sc in FOLD2_SCENES:
        if sc in stream_name: return 2
    for cl in FOLD1_CLIPS:
        if cl in stream_name: return 1
    for cl in FOLD2_CLIPS:
        if cl in stream_name: return 2
    return 1

# -----------------------------------------------------------------------------
# SemCM Range Encoder & Pretraining
# -----------------------------------------------------------------------------
from carle.rigorous_codecs import cm_prior_adaptive_encode
from experiments.exp_12_pidnet_sequence_benchmark import cm_k8_prior_adaptive_encode
from experiments.exp_14_unified_freeze_benchmark import pretrain_fold_models

# -----------------------------------------------------------------------------
# All-Frame Stream Generator
# -----------------------------------------------------------------------------
def iterate_all_cached_streams(k_tag: str):
    import gc
    # 1. nuScenes Keyframes (2 Hz) - all 2,424
    kf_p = CACHE_DIR / f"nuscenes_keyframes_2hz_{k_tag}.npz"
    meta_p = CACHE_DIR / "nuscenes_keyframes_2hz_meta.json"
    with open(meta_p) as f: meta = json.load(f)
    kf_data = np.load(kf_p, mmap_mode="r")["masks"]
    by_cam = defaultdict(list)
    for i, m in enumerate(meta):
        by_cam[(m["scene"], m["camera"])].append((i, m["timestamp"]))
    for (sc, cam), items in by_cam.items():
        items.sort(key=lambda x: x[1])
        idxs = [x[0] for x in items]
        yield "nuScenes_Keyframes_2Hz", 2.0, f"{sc}_{cam}", [kf_data[i] for i in idxs]
    del kf_data
    gc.collect()
    
    # 2. nuScenes Sweeps (12 Hz) - all 6,000 (10 scenes x 6 cams x 100 frames)
    for sf in sorted(list(SWEEPS_DIR.glob(f"sweeps_*_{k_tag}.npz"))):
        sc_name = sf.stem.replace(f"_{k_tag}", "").replace("sweeps_", "")
        s_meta_p = SWEEPS_DIR / f"sweeps_{sc_name}_meta.json"
        with open(s_meta_p) as f: s_meta = json.load(f)
        m_arr = np.load(sf)["masks"]
        cam_map = defaultdict(list)
        for i, item in enumerate(s_meta):
            cam_map[item["camera"]].append(i)
        for cam, idxs in cam_map.items():
            yield "nuScenes_Sweeps_12Hz", 12.0, f"{sc_name}_{cam}", [m_arr[i] for i in idxs]
        del m_arr
        gc.collect()
        
    # 3. Field Native & Field 10 Hz - ALL frames
    clips = ["india_bolpur", "malaysia", "sikkim", "dhaka_day", "dhaka_night", "bhubaneswar_night"]
    for c in clips:
        p = CACHE_DIR / f"field_{c}_native_{k_tag}.npz"
        if p.exists():
            arr = np.load(p)["masks"]
            yield "Field_Native", 25.0, c, [arr[i] for i in range(len(arr))]
            del arr
            gc.collect()
            
    for c in clips:
        p = CACHE_DIR / f"field_{c}_10hz_{k_tag}.npz"
        if p.exists():
            arr = np.load(p)["masks"]
            yield "Field_10Hz", 10.0, c, [arr[i] for i in range(len(arr))]
            del arr
            gc.collect()

def encode_yang(frame: np.ndarray, worker_id: int) -> int:
    H, W = frame.shape
    pid = os.getpid()
    t_in = YANG_DIR / f"all_{pid}_{worker_id}_in.yuv"
    t_out = YANG_DIR / f"all_{pid}_{worker_id}_out.bin"
    frame.tofile(t_in)
    cmd = [YANG_ENC, "-i", str(t_in), "-o", str(t_out), "-r", str(H), "-c", str(W), "-f", "1", "-s", "0", "-t", "400"]
    subprocess.run(cmd, capture_output=True, cwd=str(YANG_DIR), stdin=subprocess.DEVNULL)
    bits = t_out.stat().st_size * 8 if t_out.exists() else 0
    for p in [t_in, t_out]:
        if p.exists():
            try: p.unlink()
            except: pass
    return bits

def run_all_frames_evaluation(k_val: int):
    k_tag = f"k{k_val}"
    print(f"\n{'='*80}")
    print(f"RUNNING ALL-FRAME BENCHMARK FOR K={k_val}")
    print(f"{'='*80}")
    
    models = pretrain_fold_models(k_val)
    
    raw_records = []
    flagged_records = []
    
    total_processed = 0
    t0_start = time.perf_counter()
    
    cur_ds = None
    for ds_name, fps, stream_id, frames in iterate_all_cached_streams(k_tag):
        if ds_name != cur_ds:
            cur_ds = ds_name
            print(f"\n--> Starting Dataset: {ds_name} (FPS={fps})")
            
        N = len(frames)
        if N == 0: continue
        
        is_flagged = ("bhubaneswar_night" in stream_id)
        s_fold = get_stream_fold(stream_id)
        prior_intra = models[s_fold]["intra"]
        
        for f_idx in range(N):
            curr = frames[f_idx]
            prev = frames[f_idx - 1] if f_idx > 0 else curr
            is_identical = np.array_equal(curr, prev) and (f_idx > 0)
            H, W = curr.shape
            
            # 1. WebP Lossless
            w_buf = io.BytesIO()
            Image.fromarray(curr).save(w_buf, format="WEBP", lossless=True, quality=100, method=4)
            b_webp = len(w_buf.getvalue()) * 8
            
            # 2. JPEG-XL Effort 9
            jxl9_bytes = imagecodecs.jpegxl_encode(curr, level=100, effort=9, lossless=True)
            b_jxl9 = len(jxl9_bytes) * 8
            
            # 3. PNG level 9
            _, p_buf = cv2.imencode('.png', curr, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            b_png9 = len(p_buf) * 8
            
            # 4. Yang et al. 2026
            b_yang = encode_yang(curr, worker_id=0)
            
            # 5. SemCM (a) [Frame-Indep]
            c_a = prior_intra.copy()
            t_a = c_a.sum(axis=1)
            if k_val == 4:
                p_a = cm_prior_adaptive_encode(curr, prev, False, c_a, t_a, max_total=32768)
            else:
                p_a = cm_k8_prior_adaptive_encode(curr, prev, False, c_a, t_a, max_total=32768)
            b_semcm_a = 8 if is_identical else (len(p_a) * 8 + 1)
            
            rec = {
                "K": k_val,
                "dataset": ds_name,
                "stream": stream_id,
                "frame_idx": f_idx,
                "fps": fps,
                "yang_bits": b_yang,
                "semcm_a_bits": b_semcm_a,
                "jxl9_bits": b_jxl9,
                "webp_bits": b_webp,
                "png9_bits": b_png9
            }
            
            if is_flagged:
                flagged_records.append(rec)
            else:
                raw_records.append(rec)
                
            total_processed += 1
            if total_processed % 1000 == 0:
                elapsed = time.perf_counter() - t0_start
                rate = total_processed / elapsed
                print(f"  Processed {total_processed} frames ({rate:.1f} fps) | Yang={b_yang} b, SemCM={b_semcm_a} b, JXL={b_jxl9} b")
                
    df_raw = pd.DataFrame(raw_records)
    df_flagged = pd.DataFrame(flagged_records)
    
    out_raw = Path(f"results/phase1/p1_all_frames_raw_k{k_val}.csv")
    out_flagged = Path(f"results/phase1/p1_flagged_raw_k{k_val}.csv")
    df_raw.to_csv(out_raw, index=False)
    df_flagged.to_csv(out_flagged, index=False)
    print(f"\nSaved raw results to {out_raw} ({len(df_raw)} rows) and {out_flagged} ({len(df_flagged)} rows)")
    
    # Summary Tables
    generate_summary_tables(df_raw, df_flagged, k_val)

def generate_summary_tables(df_raw, df_flagged, k_val):
    codecs = [
        ("Yang et al. 2026", "yang_bits"),
        ("SemCM (a) [Frame-Indep] (Frozen Ref)", "semcm_a_bits"),
        ("JPEG-XL (Effort 9)", "jxl9_bits"),
        ("WebP Lossless", "webp_bits"),
        ("PNG (OpenCV level 9)", "png9_bits")
    ]
    
    datasets = ["nuScenes_Keyframes_2Hz", "nuScenes_Sweeps_12Hz", "Field_Native", "Field_10Hz"]
    rows = []
    rng = np.random.default_rng(42)
    
    for ds in datasets:
        sub = df_raw[df_raw["dataset"] == ds]
        ref_mean = sub["semcm_a_bits"].mean()
        streams = sub["stream"].unique()
        
        for c_label, col in codecs:
            mean_bpf = float(sub[col].mean())
            
            # Clustered bootstrap CIs over streams
            if len(streams) <= 1:
                vals = sub[col].values
                boot_means = [np.mean(rng.choice(vals, size=len(vals), replace=True)) for _ in range(1000)]
            else:
                s_dict = {s: sub[sub["stream"] == s][col].values for s in streams}
                boot_means = [np.mean(np.concatenate([s_dict[s] for s in rng.choice(streams, size=len(streams), replace=True)])) for _ in range(1000)]
            ci_low = float(np.percentile(boot_means, 2.5))
            ci_high = float(np.percentile(boot_means, 97.5))
            
            kbps_1cam = (mean_bpf * 10.0) / 1000.0
            kbps_6cam = kbps_1cam * 6.0
            gain_vs_semcm = ((ref_mean - mean_bpf) / ref_mean) * 100.0 if ref_mean > 0 else 0.0
            
            rows.append({
                "K": k_val,
                "Dataset": ds,
                "N_Frames": len(sub),
                "Codec": c_label,
                "Bits/Frame": round(mean_bpf, 1),
                "95% CI Lower": round(ci_low, 1),
                "95% CI Upper": round(ci_high, 1),
                "1-Cam 10Hz (kb/s)": round(kbps_1cam, 2),
                "6-Cam 10Hz (kb/s)": round(kbps_6cam, 2),
                "Gain vs SemCM(a) (%)": round(gain_vs_semcm, 2)
            })
            
    df_summary = pd.DataFrame(rows)
    out_summary = Path(f"results/phase1/p1_master_summary_k{k_val}.csv")
    df_summary.to_csv(out_summary, index=False)
    print(f"\n--- Master Summary Table (K={k_val}) ---")
    print(df_summary.to_string(index=False))
    
    # Flagged Table
    f_rows = []
    for ds in ["Field_Native", "Field_10Hz"]:
        sub = df_flagged[df_flagged["dataset"] == ds]
        ref_mean = sub["semcm_a_bits"].mean()
        for c_label, col in codecs:
            mean_bpf = float(sub[col].mean())
            ci_low = float(np.percentile([np.mean(rng.choice(sub[col].values, size=len(sub), replace=True)) for _ in range(1000)], 2.5))
            ci_high = float(np.percentile([np.mean(rng.choice(sub[col].values, size=len(sub), replace=True)) for _ in range(1000)], 97.5))
            kbps_1cam = (mean_bpf * 10.0) / 1000.0
            kbps_6cam = kbps_1cam * 6.0
            gain_vs_semcm = ((ref_mean - mean_bpf) / ref_mean) * 100.0 if ref_mean > 0 else 0.0
            
            f_rows.append({
                "K": k_val,
                "Dataset": f"Bhubaneswar_Night_{ds.replace('Field_', '')}",
                "N_Frames": len(sub),
                "Codec": c_label,
                "Bits/Frame": round(mean_bpf, 1),
                "95% CI Lower": round(ci_low, 1),
                "95% CI Upper": round(ci_high, 1),
                "1-Cam 10Hz (kb/s)": round(kbps_1cam, 2),
                "6-Cam 10Hz (kb/s)": round(kbps_6cam, 2),
                "Gain vs SemCM(a) (%)": round(gain_vs_semcm, 2)
            })
    df_f_summary = pd.DataFrame(f_rows)
    out_f_summary = Path(f"results/phase1/p1_flagged_summary_k{k_val}.csv")
    df_f_summary.to_csv(out_f_summary, index=False)
    print(f"\n--- Flagged Clip Summary Table (K={k_val}) ---")
    print(df_f_summary.to_string(index=False))

if __name__ == "__main__":
    for k in [4, 8]:
        run_all_frames_evaluation(k)
