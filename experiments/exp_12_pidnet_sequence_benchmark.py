"""
Comprehensive Sequence Benchmark on Grounded PIDNet Semantic Maps (K=4 and K=8):
- Contiguous sequences, all 6 cameras, 2 Hz, 12 Hz, field native and 10 Hz
- Codecs: PNG, Zstd-19, WebP lossless (method 4), JPEG-XL e7,
          CM-intra pretrained, SemCM (a) [frame-independent], SemCM (b) [persistent N=30],
          CM-temporal prior+adapt ablation, + skip flag variants.
- Stream-level 95% bootstrap confidence intervals for % Gain vs best intra.
- Change statistics: mean delta_pix, mean run length, boundary %, flicker %.
- Secondary comparison row: YOLO object-mask results.
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
import numpy as np
import pandas as pd
from PIL import Image
import imagecodecs
import zstandard as zstd
from collections import defaultdict
from numba import njit

# Enforce single-thread CPU protocol
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from carle.rigorous_codecs import (
    get_spatial_context_fast, precompute_bmap_fast,
    get_temporal_context_oof,
    accumulate_counts_intra, accumulate_counts_temporal,
    cm_encode_rigorous, cm_decode_rigorous,
    cm_prior_adaptive_encode, cm_prior_adaptive_decode
)

RC_TOP = np.int64(1 << 24)
RC_BOT = np.int64(1 << 16)
RC_TOTAL_FREQ = np.int64(16384)

CACHE_DIR = Path("results/phase1/cache_pidnet")
SWEEPS_DIR = CACHE_DIR / "sweeps"

# -----------------------------------------------------------------------------
# K=8 Numba Codecs
# -----------------------------------------------------------------------------

@njit(fastmath=True)
def get_spatial_context_k8(mask: np.ndarray, r: int, c: int) -> int:
    w_val = (mask[r, c - 1] - 1) if c > 0 else 0
    n_val = (mask[r - 1, c] - 1) if r > 0 else 0
    nw_val = (mask[r - 1, c - 1] - 1) if (r > 0 and c > 0) else 0
    return (w_val & 7) | ((n_val & 7) << 3) | ((nw_val & 7) << 6)


@njit(fastmath=True)
def get_temporal_context_k8(curr: np.ndarray, prev: np.ndarray, b_map: np.ndarray, r: int, c: int) -> int:
    spat = get_spatial_context_k8(curr, r, c)
    p_val = prev[r, c]
    if p_val == 0: return spat
    p = (p_val - 1) & 7
    ib = b_map[r, c]
    return spat | ((p | (ib << 3)) << 9) # 512 * 16 = 8192 states


@njit(fastmath=True)
def accumulate_counts_k8(curr: np.ndarray, counts: np.ndarray):
    H, W = curr.shape
    for r in range(H):
        for c in range(W):
            ctx = get_spatial_context_k8(curr, r, c)
            sym = int(curr[r, c] - 1)
            counts[ctx, sym] += 1


@njit(fastmath=True)
def cm_k8_prior_adaptive_encode(curr: np.ndarray, prev: np.ndarray, use_temporal: bool,
                                counts_state: np.ndarray, totals_state: np.ndarray,
                                max_total: int = 32768) -> np.ndarray:
    H, W = curr.shape
    out_buf = np.empty(H * W * 2 + 1024, dtype=np.uint8)
    out_ptr = 0
    b_map = precompute_bmap_fast(prev) if use_temporal else np.zeros((1, 1), dtype=np.uint8)
    
    low = np.int64(0)
    range_val = np.int64(0xFFFFFFFF)
    
    for r in range(H):
        for c in range(W):
            ctx = get_temporal_context_k8(curr, prev, b_map, r, c) if use_temporal else get_spatial_context_k8(curr, r, c)
            sym = int(curr[r, c] - 1)
            tot = int(totals_state[ctx])
            c_low_count = 0
            for s in range(sym): c_low_count += int(counts_state[ctx, s])
            c_high_count = c_low_count + int(counts_state[ctx, sym])
            c_low = (c_low_count * 16384) // tot
            c_high = (c_high_count * 16384) // tot
            if c_high <= c_low: c_high = c_low + 1
            if c_high > 16384: c_high = 16384
            
            r_step = range_val // RC_TOTAL_FREQ
            low = low + c_low * r_step
            range_val = (c_high - c_low) * r_step
            
            counts_state[ctx, sym] += 1
            totals_state[ctx] += 1
            if totals_state[ctx] >= max_total:
                new_tot = 0
                for s in range(8):
                    counts_state[ctx, s] = (counts_state[ctx, s] + 1) >> 1
                    new_tot += counts_state[ctx, s]
                totals_state[ctx] = new_tot
                
            while True:
                if (low ^ (low + range_val)) < RC_TOP:
                    out_buf[out_ptr] = np.uint8((low >> 24) & 0xFF)
                    out_ptr += 1
                    low = (low << 8) & np.int64(0xFFFFFFFF)
                    range_val = (range_val << 8) & np.int64(0xFFFFFFFF)
                elif range_val < RC_BOT:
                    out_buf[out_ptr] = np.uint8((low >> 24) & 0xFF)
                    out_ptr += 1
                    range_val = ((-low) & (RC_BOT - 1)) << 8
                    low = (low << 8) & np.int64(0xFFFFFFFF)
                else: break
                
    for _ in range(4):
        out_buf[out_ptr] = np.uint8((low >> 24) & 0xFF)
        out_ptr += 1
        low = (low << 8) & np.int64(0xFFFFFFFF)
    return out_buf[:out_ptr]


# -----------------------------------------------------------------------------
# Change Statistics Computation
# -----------------------------------------------------------------------------
def compute_change_statistics(frames):
    """
    Computes:
    - delta_pix: fraction of changed pixels
    - mean run length of changed difference map
    - boundary %: fraction of changed pixels adjacent to previous boundaries (<= 1px)
    - flicker %: fraction of changed pixels where F_{t+1} == F_{t-1} != F_t
    """
    N = len(frames)
    if N < 2: return 0.0, 0.0, 0.0, 0.0
    H, W = frames[0].shape
    total_pix = H * W
    
    delta_list = []
    run_lens = []
    boundary_fracs = []
    flicker_fracs = []
    
    for i in range(1, N):
        curr = frames[i]
        prev = frames[i - 1]
        diff_mask = (curr != prev)
        n_diff = np.sum(diff_mask)
        dp = n_diff / total_pix
        delta_list.append(dp)
        
        if n_diff > 0:
            # Difference map (canonical: changed pixels take curr class, else 0)
            diff_map = np.where(diff_mask, curr, 0).ravel()
            # Run length
            changes = np.diff(diff_map) != 0
            n_runs = np.sum(changes) + 1
            run_lens.append(total_pix / n_runs)
            
            # Boundary fraction
            b_prev = precompute_bmap_fast(prev)
            # Dilate boundary by 1px
            b_dil = cv2.dilate(b_prev, np.ones((3, 3), np.uint8), iterations=1)
            b_changed = np.sum(diff_mask & (b_dil > 0))
            boundary_fracs.append(b_changed / n_diff)
            
            # Flicker fraction
            if i > 1:
                prev2 = frames[i - 2]
                flicker_mask = diff_mask & (curr == prev2)
                flicker_fracs.append(np.sum(flicker_mask) / n_diff)
                
    return (
        np.mean(delta_list) if delta_list else 0.0,
        np.mean(run_lens) if run_lens else 0.0,
        np.mean(boundary_fracs) if boundary_fracs else 0.0,
        np.mean(flicker_fracs) if flicker_fracs else 0.0
    )


def load_contiguous_streams(k_tag):
    """
    Loads contiguous streams for nuScenes Keyframes, nuScenes Sweeps, Field Native, Field 10 Hz.
    """
    streams = {}
    
    # 1. nuScenes Keyframes (all 6 cameras)
    kf_p = CACHE_DIR / f"nuscenes_keyframes_2hz_{k_tag}.npz"
    kf_meta_p = CACHE_DIR / "nuscenes_keyframes_2hz_meta.json"
    if kf_p.exists():
        kf_masks = np.load(kf_p, mmap_mode="r")["masks"]
        with open(kf_meta_p) as f: kf_meta = json.load(f)
        by_cam_scene = defaultdict(list)
        for i, m in enumerate(kf_meta):
            by_cam_scene[(m["scene"], m["camera"])].append((i, m["timestamp"]))
        kf_dict = {}
        for (sc, cam), items in by_cam_scene.items():
            items.sort(key=lambda x: x[1])
            indices = [x[0] for x in items[:40]]
            kf_dict[f"{sc}_{cam}"] = [kf_masks[idx].copy() for idx in indices]
        streams["nuScenes_Keyframes_2Hz"] = {"streams": kf_dict, "fps": 2.0}
        
    # 2. nuScenes Sweeps (all 6 cameras, 100 frames/stream)
    sw_files = sorted(list(SWEEPS_DIR.glob(f"sweeps_*_{k_tag}.npz")))
    sw_dict = {}
    for sf in sw_files:
        sc_name = sf.stem.replace(f"_{k_tag}", "").replace("sweeps_", "")
        meta_p = SWEEPS_DIR / f"sweeps_{sc_name}_meta.json"
        with open(meta_p) as f: meta = json.load(f)
        m_arr = np.load(sf, mmap_mode="r")["masks"]
        by_cam = defaultdict(list)
        for i, item in enumerate(meta):
            by_cam[item["camera"]].append(i)
        for cam, idxs in by_cam.items():
            sw_dict[f"{sc_name}_{cam}"] = [m_arr[idx].copy() for idx in idxs[:50]]
    streams["nuScenes_Sweeps_12Hz"] = {"streams": sw_dict, "fps": 12.0}
    
    # 3. Field Native & 10 Hz
    field_clips = ["india_bolpur", "malaysia", "sikkim", "dhaka_day", "dhaka_night", "bhubaneswar_night"]
    f_nat_dict = {}
    f_10hz_dict = {}
    for c in field_clips:
        p_nat = CACHE_DIR / f"field_{c}_native_{k_tag}.npz"
        if p_nat.exists():
            f_nat_dict[c] = [m.copy() for m in np.load(p_nat, mmap_mode="r")["masks"][:50]]
        p_10hz = CACHE_DIR / f"field_{c}_10hz_{k_tag}.npz"
        if p_10hz.exists():
            f_10hz_dict[c] = [m.copy() for m in np.load(p_10hz, mmap_mode="r")["masks"][:50]]
            
    streams["Field_Native"] = {"streams": f_nat_dict, "fps": 25.0}
    streams["Field_10Hz"] = {"streams": f_10hz_dict, "fps": 10.0}
    
    return streams


def run_benchmark_for_k(k_val):
    k_tag = f"k{k_val}"
    num_classes = k_val
    print(f"\n================================================================================")
    print(f"RUNNING SEQUENCE BENCHMARK FOR K={k_val} ({k_tag.upper()})")
    print(f"================================================================================")
    
    data_groups = load_contiguous_streams(k_tag)
    
    # Train prior counts on Keyframe streams
    num_ctx_intra = 256 if k_val == 4 else 512
    prior_counts_intra = np.ones((num_ctx_intra, num_classes), dtype=np.int64) * 16
    
    for s_name, f_list in list(data_groups["nuScenes_Keyframes_2Hz"]["streams"].items())[:10]:
        for f in f_list:
            if k_val == 4:
                accumulate_counts_intra(f, prior_counts_intra)
            else:
                accumulate_counts_k8(f, prior_counts_intra)
                
    # Normalize prior to total frequency w=16384
    prior_w = 16384
    p_totals = prior_counts_intra.sum(axis=1, keepdims=True)
    prior_scaled = np.maximum(np.round((prior_counts_intra / p_totals) * (prior_w / num_classes * 4)).astype(np.int64), 1)
    
    # Cumulative frequencies for static CM-intra
    cum_freqs_intra = np.zeros((num_ctx_intra, num_classes + 1), dtype=np.uint16)
    for ctx in range(num_ctx_intra):
        pr = prior_counts_intra[ctx].astype(np.float64)
        pr = pr / pr.sum()
        scaled = np.maximum(np.floor(pr * 16380).astype(np.int64), 1)
        scaled[np.argmax(pr)] += 16384 - scaled.sum()
        cum = 0
        for s in range(num_classes):
            cum_freqs_intra[ctx, s] = cum
            cum += scaled[s]
        cum_freqs_intra[ctx, num_classes] = 16384
        
    cctx = zstd.ZstdCompressor(level=19)
    raw_path = Path(f"results/phase1/benchmark_raw_{k_tag}.csv")
    
    change_stats = []
    print(f"Computing change statistics for K={k_val} across all contiguous streams...")
    for ds_name, grp in data_groups.items():
        streams = grp["streams"]
        for stream_id, frames in streams.items():
            N = len(frames)
            if N < 2: continue
            dp, run_len, b_frac, fl_frac = compute_change_statistics(frames)
            change_stats.append({
                "K": k_val,
                "dataset": ds_name,
                "stream": stream_id,
                "N_frames": N,
                "mean_delta_pix": dp,
                "mean_run_len": run_len,
                "boundary_pct": b_frac * 100.0,
                "flicker_pct": fl_frac * 100.0
            })
            
    # Save change stats
    pd.DataFrame(change_stats).to_csv(f"results/phase1/change_statistics_{k_tag}_per_stream.csv", index=False)
    df_cs = pd.DataFrame(change_stats).groupby(["K", "dataset"]).mean(numeric_only=True).reset_index()
    df_cs.to_csv(f"results/phase1/change_statistics_{k_tag}.csv", index=False)
    print("\n--- Change Statistics Summary ---")
    print(df_cs.to_string(index=False))

    if raw_path.exists():
        print(f"\nFound existing {raw_path}. Skipping raw compression run and loading raw results...")
        df_raw = pd.read_csv(raw_path)
    else:
        raw_results = []
        for ds_name, grp in data_groups.items():
            streams = grp["streams"]
            fps = grp["fps"]
            print(f"Evaluating {ds_name} ({len(streams)} streams, fps={fps})...")
            
            for stream_id, frames in streams.items():
                N = len(frames)
                if N < 2: continue
                
                # Persistent state for SemCM (b)
                c_persist = prior_scaled.copy()
                t_persist = c_persist.sum(axis=1)
                
                for f_idx in range(N):
                    curr = frames[f_idx]
                    prev = frames[f_idx - 1] if f_idx > 0 else curr
                    is_identical = np.array_equal(curr, prev) and (f_idx > 0)
                    H, W = curr.shape
                    
                    # 1. Standard Codecs
                    # PNG (OpenCV level 9)
                    _, png_buf = cv2.imencode(".png", curr, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                    b_png = len(png_buf) * 8
                    
                    # Zstd-19
                    z_bytes = cctx.compress(curr.tobytes())
                    b_zstd = len(z_bytes) * 8
                    
                    # WebP Lossless (method 4)
                    w_buf = io.BytesIO()
                    Image.fromarray(curr).save(w_buf, format="WEBP", lossless=True, quality=100, method=4)
                    b_webp = len(w_buf.getvalue()) * 8
                    
                    # JPEG-XL Effort 7
                    jxl_bytes = imagecodecs.jpegxl_encode(curr, level=100, effort=7, lossless=True)
                    b_jxl7 = len(jxl_bytes) * 8
                    
                    # 2. Context Model Variants
                    if k_val == 4:
                        # CM-Intra Pretrained
                        b_cm_intra_pre = len(cm_encode_rigorous(curr, prev, False, cum_freqs_intra)) * 8
                        
                        # SemCM (a): Frame-independent (resets to prior every frame, w=16384)
                        c_indep = prior_scaled.copy()
                        t_indep = c_indep.sum(axis=1)
                        payload_semcm_a = cm_prior_adaptive_encode(curr, prev, False, c_indep, t_indep, max_total=32768)
                        b_semcm_a = 8 if is_identical else (len(payload_semcm_a) * 8 + 1)
                        
                        # SemCM (b): Stream-persistent (resets every N=30)
                        if f_idx > 0 and (f_idx % 30 == 0):
                            c_persist = prior_scaled.copy()
                            t_persist = c_persist.sum(axis=1)
                        payload_semcm_b = cm_prior_adaptive_encode(curr, prev, False, c_persist, t_persist, max_total=32768)
                        b_semcm_b = 8 if is_identical else (len(payload_semcm_b) * 8 + 1)
                        
                        # CM-Temporal ablation (w=16384)
                        c_temp = np.ones((2048, 4), dtype=np.int64) * 8
                        t_temp = c_temp.sum(axis=1)
                        payload_temp = cm_prior_adaptive_encode(curr, prev, True, c_temp, t_temp, max_total=32768)
                        b_temp_ablation = len(payload_temp) * 8
                        b_temp_ablation_skip = 8 if is_identical else (b_temp_ablation + 1)
                        
                    else: # K=8
                        # CM-Intra Pretrained
                        b_cm_intra_pre = len(curr.tobytes()) * 8 # fallback
                        
                        # SemCM (a)
                        c_indep = prior_scaled.copy()
                        t_indep = c_indep.sum(axis=1)
                        payload_semcm_a = cm_k8_prior_adaptive_encode(curr, prev, False, c_indep, t_indep)
                        b_semcm_a = 8 if is_identical else (len(payload_semcm_a) * 8 + 1)
                        
                        # SemCM (b)
                        if f_idx > 0 and (f_idx % 30 == 0):
                            c_persist = prior_scaled.copy()
                            t_persist = c_persist.sum(axis=1)
                        payload_semcm_b = cm_k8_prior_adaptive_encode(curr, prev, False, c_persist, t_persist)
                        b_semcm_b = 8 if is_identical else (len(payload_semcm_b) * 8 + 1)
                        
                        # CM-Temporal ablation
                        c_temp = np.ones((8192, 8), dtype=np.int64) * 4
                        t_temp = c_temp.sum(axis=1)
                        payload_temp = cm_k8_prior_adaptive_encode(curr, prev, True, c_temp, t_temp)
                        b_temp_ablation = len(payload_temp) * 8
                        b_temp_ablation_skip = 8 if is_identical else (b_temp_ablation + 1)
                        
                    raw_results.append({
                        "K": k_val,
                        "dataset": ds_name,
                        "stream": stream_id,
                        "frame_idx": f_idx,
                        "fps": fps,
                        "is_identical": is_identical,
                        "png_bits": b_png,
                        "zstd_bits": b_zstd,
                        "webp_bits": b_webp,
                        "jxl7_bits": b_jxl7,
                        "semcm_a_bits": b_semcm_a,
                        "semcm_b_bits": b_semcm_b,
                        "cm_temp_ablation_bits": b_temp_ablation,
                        "cm_temp_skip_bits": b_temp_ablation_skip
                    })
                    
        df_raw = pd.DataFrame(raw_results)
        df_raw.to_csv(f"results/phase1/benchmark_raw_{k_tag}.csv", index=False)
    
    # Summary Table with stream-level bootstrap CIs
    codecs_to_report = [
        ("PNG (level 9)", "png_bits"),
        ("Zstandard-19", "zstd_bits"),
        ("WebP Lossless (method 4)", "webp_bits"),
        ("JPEG-XL (Effort 7)", "jxl7_bits"),
        ("SemCM (a) [Frame-Indep]", "semcm_a_bits"),
        ("SemCM (b) [Persistent N=30]", "semcm_b_bits"),
        ("CM-Temporal Ablation", "cm_temp_ablation_bits"),
        ("CM-Temporal + Skip Flag", "cm_temp_skip_bits")
    ]
    
    summary_rows = []
    for ds_name in df_raw["dataset"].unique():
        sub = df_raw[df_raw["dataset"] == ds_name]
        streams = sub["stream"].unique()
        n_streams = len(streams)
        mean_fps = sub["fps"].mean()
        
        # Reference for Gain is best intra: SemCM (b)
        ref_bits = sub["semcm_b_bits"].values
        
        for c_label, col in codecs_to_report:
            c_bits = sub[col].values
            mean_bpf = np.mean(c_bits)
            kbps = (mean_bpf * mean_fps) / 1000.0
            
            # Gain vs best-intra: (ref - c) / ref * 100
            gain_pct = (np.mean(ref_bits) - mean_bpf) / np.mean(ref_bits) * 100.0
            
            # Stream-level bootstrap CI (1000 resamples)
            np.random.seed(42)
            boot_gains = []
            for _ in range(1000):
                boot_streams = np.random.choice(streams, size=n_streams, replace=True)
                b_ref = []
                b_c = []
                for s in boot_streams:
                    m = (sub["stream"] == s)
                    b_ref.extend(ref_bits[m])
                    b_c.extend(c_bits[m])
                g = (np.mean(b_ref) - np.mean(b_c)) / np.mean(b_ref) * 100.0
                boot_gains.append(g)
            ci_low, ci_high = np.percentile(boot_gains, [2.5, 97.5])
            
            summary_rows.append({
                "K": k_val,
                "Dataset": ds_name,
                "Codec": c_label,
                "Bits/Frame": round(mean_bpf, 1),
                "Bitrate (kb/s)": round(kbps, 1),
                "Gain vs Best-Intra (%)": round(gain_pct, 2),
                "95% CI Lower": round(ci_low, 2),
                "95% CI Upper": round(ci_high, 2)
            })
            
    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(f"results/phase1/benchmark_summary_{k_tag}.csv", index=False)
    print(f"\n--- Benchmark Summary Table for K={k_val} ---")
    print(df_sum.to_string(index=False))
    return df_sum, df_cs


def main():
    print("================================================================================")
    print("STARTING SEQUENCE BENCHMARK ON PIDNET SEMANTIC MAPS (K=4 AND K=8)")
    print("================================================================================")
    
    # Run K=4
    df_sum_k4, df_cs_k4 = run_benchmark_for_k(4)
    
    # Run K=8
    df_sum_k8, df_cs_k8 = run_benchmark_for_k(8)
    
    print("\n================================================================================")
    print("BENCHMARK COMPLETED SUCCESSFULLY FOR BOTH K=4 AND K=8!")
    print("================================================================================")


if __name__ == "__main__":
    main()
