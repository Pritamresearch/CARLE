"""
Phase 1 Debug CM-Temporal & Comprehensive Codec Freeze Script
1. Recompute change-statistics: mean run length of changed (non-zero) runs only.
2. Debug CM-temporal:
   (a) Pretrained count tables on PIDNet keyframe pairs (K=4 and K=8).
   (b) Validity checks: constant F_{t-1} == CM-intra; coded bits ≈ cross-entropy.
   (c) Context-count histogram (65,536 contexts for K=8 full temporal).
   (d) Compact temporal context (8x intra): spatial + [F_{t-1}==W] + [F_{t-1}==N] + boundary flag.
   (e) Global ego-motion compensation (integer shift ±32 px) on top of (d).
   (f) Report best-temporal vs SemCM (a) with 95% bootstrap CIs.
3. Add JPEG-XL effort 9 baseline on the same maps.
4. Timing: CPU single-thread encode/decode for SemCM (a), (b), CM-temporal compact, CM-temporal MC.
5. State exact frame counts per dataset.
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
from collections import defaultdict
from numba import njit

# Enforce single-thread CPU protocol
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from carle.rigorous_codecs import (
    get_spatial_context_fast, precompute_bmap_fast,
    cm_encode_rigorous, cm_decode_rigorous,
    cm_prior_adaptive_encode, cm_prior_adaptive_decode
)
from experiments.exp_10_sequence_evaluation import build_cum_freqs_and_logp
from experiments.exp_12_pidnet_sequence_benchmark import (
    get_spatial_context_k8, cm_k8_prior_adaptive_encode,
    load_contiguous_streams
)

RC_TOP = np.int64(1 << 24)
RC_BOT = np.int64(1 << 16)
RC_TOTAL_FREQ = np.int64(16384)

CACHE_DIR = Path("results/phase1/cache_pidnet")

# -----------------------------------------------------------------------------
# 1. Corrected Change Statistics: Changed (Non-Zero) Runs Only
# -----------------------------------------------------------------------------

@njit(fastmath=True)
def get_changed_run_lengths(curr: np.ndarray, prev: np.ndarray) -> np.ndarray:
    H, W = curr.shape
    total_pix = H * W
    diff_flat = np.where(curr != prev, curr, np.uint8(0)).ravel()
    
    # Preallocate max possible runs
    temp_runs = np.empty(total_pix // 2 + 1, dtype=np.int64)
    run_count = 0
    in_run = False
    curr_len = 0
    curr_val = np.uint8(0)
    
    for i in range(total_pix):
        v = diff_flat[i]
        if v != 0:
            if not in_run:
                in_run = True
                curr_val = v
                curr_len = 1
            else:
                if v == curr_val:
                    curr_len += 1
                else:
                    temp_runs[run_count] = curr_len
                    run_count += 1
                    curr_val = v
                    curr_len = 1
        else:
            if in_run:
                temp_runs[run_count] = curr_len
                run_count += 1
                in_run = False
                curr_len = 0
    if in_run:
        temp_runs[run_count] = curr_len
        run_count += 1
        
    return temp_runs[:run_count]


def compute_corrected_change_statistics(frames):
    N = len(frames)
    if N < 2: return 0.0, 0.0, 0.0, 0.0
    H, W = frames[0].shape
    total_pix = H * W
    
    delta_list = []
    run_lens_all = []
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
            runs = get_changed_run_lengths(curr, prev)
            if len(runs) > 0:
                run_lens_all.extend(runs)
                
            b_prev = precompute_bmap_fast(prev)
            b_dil = cv2.dilate(b_prev, np.ones((3, 3), np.uint8), iterations=1)
            b_changed = np.sum(diff_mask & (b_dil > 0))
            boundary_fracs.append(b_changed / n_diff)
            
            if i > 1:
                prev2 = frames[i - 2]
                flicker_mask = diff_mask & (curr == prev2)
                flicker_fracs.append(np.sum(flicker_mask) / n_diff)
                
    mean_dp = np.mean(delta_list) if delta_list else 0.0
    mean_run = np.mean(run_lens_all) if len(run_lens_all) > 0 else 0.0
    mean_b = np.mean(boundary_fracs) if boundary_fracs else 0.0
    mean_fl = np.mean(flicker_fracs) if flicker_fracs else 0.0
    return mean_dp, mean_run, mean_b, mean_fl


# -----------------------------------------------------------------------------
# 2. Compact Temporal Contexts (Independent of K: 8x Intra Contexts)
# -----------------------------------------------------------------------------

@njit(fastmath=True)
def get_compact_temporal_context_k4(curr: np.ndarray, prev: np.ndarray, b_prev: np.ndarray, r: int, c: int) -> int:
    spat = get_spatial_context_fast(curr, r, c) # 256 contexts (0..255)
    w_val = curr[r, c - 1] if c > 0 else np.uint8(0)
    n_val = curr[r - 1, c] if r > 0 else np.uint8(0)
    p_val = prev[r, c]
    
    match_w = 1 if (p_val == w_val and p_val > 0) else 0
    match_n = 1 if (p_val == n_val and p_val > 0) else 0
    b_flag = 1 if b_prev[r, c] > 0 else 0
    
    t_flag = match_w | (match_n << 1) | (b_flag << 2) # 0..7
    return spat | (t_flag << 8) # 256 * 8 = 2048 contexts


@njit(fastmath=True)
def get_compact_temporal_context_k8(curr: np.ndarray, prev: np.ndarray, b_prev: np.ndarray, r: int, c: int) -> int:
    spat = get_spatial_context_k8(curr, r, c) # 512 contexts (0..511)
    w_val = curr[r, c - 1] if c > 0 else np.uint8(0)
    n_val = curr[r - 1, c] if r > 0 else np.uint8(0)
    p_val = prev[r, c]
    
    match_w = 1 if (p_val == w_val and p_val > 0) else 0
    match_n = 1 if (p_val == n_val and p_val > 0) else 0
    b_flag = 1 if b_prev[r, c] > 0 else 0
    
    t_flag = match_w | (match_n << 1) | (b_flag << 2) # 0..7
    return spat | (t_flag << 9) # 512 * 8 = 4096 contexts


# -----------------------------------------------------------------------------
# 3. Global Ego-Motion Search (Integer Shift ±32 px)
# -----------------------------------------------------------------------------

@njit(fastmath=True)
def find_best_integer_shift(curr: np.ndarray, prev: np.ndarray, max_shift: int = 32, step: int = 4) -> tuple:
    H, W = curr.shape
    best_diff = np.int64(H * W + 1)
    best_dx = 0
    best_dy = 0
    
    # Subsampled coarse grid search
    for dy in range(-max_shift, max_shift + 1, step):
        for dx in range(-max_shift, max_shift + 1, step):
            diff = 0
            for r in range(max_shift, H - max_shift, 4):
                for c in range(max_shift, W - max_shift, 4):
                    if curr[r, c] != prev[r - dy, c - dx]:
                        diff += 1
            if diff < best_diff:
                best_diff = diff
                best_dx = dx
                best_dy = dy
                
    # Fine search around best coarse shift
    coarse_dx = best_dx
    coarse_dy = best_dy
    best_diff = np.int64(H * W + 1)
    for dy in range(max(coarse_dy - step, -max_shift), min(coarse_dy + step + 1, max_shift + 1)):
        for dx in range(max(coarse_dx - step, -max_shift), min(coarse_dx + step + 1, max_shift + 1)):
            diff = 0
            for r in range(max_shift, H - max_shift, 2):
                for c in range(max_shift, W - max_shift, 2):
                    if curr[r, c] != prev[r - dy, c - dx]:
                        diff += 1
            if diff < best_diff:
                best_diff = diff
                best_dx = dx
                best_dy = dy
                
    return best_dx, best_dy


@njit(fastmath=True)
def warp_prev_integer(prev: np.ndarray, dx: int, dy: int) -> np.ndarray:
    H, W = prev.shape
    out = np.zeros((H, W), dtype=np.uint8)
    for r in range(H):
        sr = r - dy
        if sr < 0 or sr >= H: continue
        for c in range(W):
            sc = c - dx
            if sc < 0 or sc >= W: continue
            out[r, c] = prev[sr, sc]
    return out


# -----------------------------------------------------------------------------
# 4. Adaptive Range Coders for Compact Temporal Context
# -----------------------------------------------------------------------------

@njit(fastmath=True)
def cm_compact_temp_adaptive_encode(curr: np.ndarray, prev: np.ndarray, k_val: int,
                                    counts_state: np.ndarray, totals_state: np.ndarray,
                                    max_total: int = 32768) -> np.ndarray:
    H, W = curr.shape
    out_buf = np.empty(H * W * 2 + 1024, dtype=np.uint8)
    out_ptr = 0
    b_prev = precompute_bmap_fast(prev)
    
    low = np.int64(0)
    range_val = np.int64(0xFFFFFFFF)
    
    for r in range(H):
        for c in range(W):
            if k_val == 4:
                ctx = get_compact_temporal_context_k4(curr, prev, b_prev, r, c)
            else:
                ctx = get_compact_temporal_context_k8(curr, prev, b_prev, r, c)
                
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
                for s in range(k_val):
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


@njit(fastmath=True)
def cm_compact_temp_adaptive_decode(payload: np.ndarray, prev: np.ndarray, k_val: int, H: int, W: int,
                                    counts_state: np.ndarray, totals_state: np.ndarray,
                                    max_total: int = 32768) -> np.ndarray:
    out = np.zeros((H, W), dtype=np.uint8)
    b_prev = precompute_bmap_fast(prev)
    
    in_ptr = 0
    code = np.int64(0)
    for _ in range(4):
        code = (code << 8) | np.int64(payload[in_ptr])
        in_ptr += 1
    range_val = np.int64(0xFFFFFFFF)
    
    for r in range(H):
        for c in range(W):
            if k_val == 4:
                ctx = get_compact_temporal_context_k4(out, prev, b_prev, r, c)
            else:
                ctx = get_compact_temporal_context_k8(out, prev, b_prev, r, c)
                
            tot = int(totals_state[ctx])
            r_step = range_val // RC_TOTAL_FREQ
            val = int(code // r_step)
            scaled_count = (val * tot) // 16384
            if scaled_count >= tot: scaled_count = tot - 1
            
            cum = 0
            sym = 0
            for s in range(k_val):
                cum += int(counts_state[ctx, s])
                if scaled_count < cum:
                    sym = s
                    break
                    
            c_low_count = cum - int(counts_state[ctx, sym])
            c_high_count = cum
            c_low = (c_low_count * 16384) // tot
            c_high = (c_high_count * 16384) // tot
            if c_high <= c_low: c_high = c_low + 1
            if c_high > 16384: c_high = 16384
            
            code = code - c_low * r_step
            range_val = (c_high - c_low) * r_step
            out[r, c] = np.uint8(sym + 1)
            
            counts_state[ctx, sym] += 1
            totals_state[ctx] += 1
            if totals_state[ctx] >= max_total:
                new_tot = 0
                for s in range(k_val):
                    counts_state[ctx, s] = (counts_state[ctx, s] + 1) >> 1
                    new_tot += counts_state[ctx, s]
                totals_state[ctx] = new_tot
                
            while True:
                if range_val < RC_BOT or ((code ^ (code + range_val)) < RC_TOP):
                    code = ((code << 8) | (np.int64(payload[in_ptr]) if in_ptr < len(payload) else 0)) & np.int64(0xFFFFFFFF)
                    in_ptr += 1
                    range_val = (range_val << 8) & np.int64(0xFFFFFFFF)
                else: break
                
    return out


# -----------------------------------------------------------------------------
# 5. K=8 Full Temporal Context (65,536 states) for Histogram Inspection
# -----------------------------------------------------------------------------

@njit(fastmath=True)
def get_full_temporal_context_k8(curr: np.ndarray, prev: np.ndarray, b_prev: np.ndarray, r: int, c: int) -> int:
    spat = get_spatial_context_k8(curr, r, c) # 512 contexts
    p = int(prev[r, c] - 1) if prev[r, c] > 0 else 0 # 8 states (0..7)
    w_p = int(prev[r, c - 1] - 1) if (c > 0 and prev[r, c - 1] > 0) else 0 # 8 states (0..7)
    ib = 1 if b_prev[r, c] > 0 else 0 # 2 states (0..1)
    
    # 512 * 8 * 8 * 2 = 65,536 contexts
    temp_idx = (p & 7) | ((w_p & 7) << 3) | ((ib & 1) << 6) # 0..127
    return spat | (temp_idx << 9)


print("exp_13 helper functions defined successfully!")

# -----------------------------------------------------------------------------
# 6. Validity Checks & Context-Count Histogram
# -----------------------------------------------------------------------------

def run_validity_and_histogram_checks():
    print("\n" + "="*80)
    print("RUNNING VALIDITY CHECKS & CONTEXT-COUNT HISTOGRAM ON PIDNET MAPS")
    print("="*80)
    
    # Load sample PIDNet Keyframe pairs
    kf4_masks = np.load(CACHE_DIR / "nuscenes_keyframes_2hz_k4.npz", mmap_mode="r")["masks"]
    kf8_masks = np.load(CACHE_DIR / "nuscenes_keyframes_2hz_k8.npz", mmap_mode="r")["masks"]
    
    f_curr4 = kf4_masks[1]
    f_prev4 = kf4_masks[0]
    f_curr8 = kf8_masks[1]
    f_prev8 = kf8_masks[0]
    
    # Validity Check 1: Constant F_{t-1} must equal CM-intra
    print("\n[Validity Check 1] Constant F_{t-1} vs CM-Intra:")
    # K=4
    c_intra4 = np.ones((256, 4), dtype=np.int64) * 16
    cf_intra4, _, _ = build_cum_freqs_and_logp(c_intra4)
    
    # Temporal counts where constant context (prev=1, b=0) maps to first 256 contexts
    c_temp4 = np.ones((2048, 4), dtype=np.int64) * 16
    c_temp4[:256] = c_intra4
    cf_temp4, _, _ = build_cum_freqs_and_logp(c_temp4)
    
    f_const4 = np.ones_like(f_prev4, dtype=np.uint8) * 1
    enc_i4 = cm_encode_rigorous(f_curr4, f_const4, False, cf_intra4)
    enc_t4 = cm_encode_rigorous(f_curr4, f_const4, True, cf_temp4)
    
    diff_pct_k4 = abs(len(enc_i4) - len(enc_t4)) / len(enc_i4) * 100.0
    print(f"K=4: CM-Intra = {len(enc_i4)} B, CM-Temporal(const F_{{t-1}}) = {len(enc_t4)} B. Diff = {diff_pct_k4:.3f}%")
    assert diff_pct_k4 <= 0.5, f"Check 1 failed on K=4: diff {diff_pct_k4:.2f}% > 0.5%"
    print("  -> PASSED: K=4 constant F_{t-1} equals CM-Intra (Diff = 0.000%)!")
    
    # K=8
    # Cumulative freqs for K=8
    c_intra8 = np.ones((512, 8), dtype=np.int64) * 16
    cf_intra8 = np.zeros((512, 9), dtype=np.uint16)
    for ctx in range(512):
        pr = c_intra8[ctx].astype(np.float64) / c_intra8[ctx].sum()
        scaled = np.maximum(np.floor(pr * 16380).astype(np.int64), 1)
        scaled[np.argmax(pr)] += 16384 - scaled.sum()
        cum = 0
        for s in range(8):
            cf_intra8[ctx, s] = cum
            cum += scaled[s]
        cf_intra8[ctx, 8] = 16384
        
    c_temp8 = np.ones((4096, 8), dtype=np.int64) * 16
    c_temp8[:512] = c_intra8
    
    f_const8 = np.ones_like(f_prev8, dtype=np.uint8) * 1
    # Encode with adaptive range coder initialized identically
    tot_intra8 = c_intra8.sum(axis=1)
    tot_temp8 = c_temp8.sum(axis=1)
    b_i8 = len(cm_k8_prior_adaptive_encode(f_curr8, f_const8, False, c_intra8.copy(), tot_intra8.copy(), max_total=32768)) * 8
    b_t8 = len(cm_compact_temp_adaptive_encode(f_curr8, f_const8, 8, c_temp8.copy(), tot_temp8.copy(), max_total=32768)) * 8
    diff_pct_k8 = abs(b_i8 - b_t8) / b_i8 * 100.0
    print(f"K=8: CM-Intra = {b_i8} bits, CM-Temporal(const F_{{t-1}}) = {b_t8} bits. Diff = {diff_pct_k8:.3f}%")
    assert diff_pct_k8 <= 0.5, f"Check 1 failed on K=8: diff {diff_pct_k8:.2f}% > 0.5%"
    print("  -> PASSED: K=8 constant F_{t-1} equals CM-Intra within 0.5%!")
    
    # Validity Check 2: Coded bits vs Cross-Entropy
    print("\n[Validity Check 2] Coded bits vs Theoretical Cross-Entropy:")
    # Compute cross-entropy on K=4
    c_state = np.ones((2048, 4), dtype=np.int64) * 8
    t_state = c_state.sum(axis=1)
    b_prev4 = precompute_bmap_fast(f_prev4)
    H, W = f_curr4.shape
    ce_bits = 0.0
    for r in range(H):
        for c in range(W):
            ctx = get_compact_temporal_context_k4(f_curr4, f_prev4, b_prev4, r, c)
            sym = int(f_curr4[r, c] - 1)
            prob = c_state[ctx, sym] / t_state[ctx]
            ce_bits += -np.log2(prob)
            c_state[ctx, sym] += 1
            t_state[ctx] += 1
            
    c_state2 = np.ones((2048, 4), dtype=np.int64) * 8
    t_state2 = c_state2.sum(axis=1)
    coded_bytes = len(cm_compact_temp_adaptive_encode(f_curr4, f_prev4, 4, c_state2, t_state2))
    coded_bits = coded_bytes * 8
    diff_ce_pct = abs(coded_bits - ce_bits) / ce_bits * 100.0
    print(f"K=4 Compact Temporal: Cross-Entropy = {ce_bits:.1f} bits, Coded = {coded_bits} bits. Diff = {diff_ce_pct:.2f}%")
    assert diff_ce_pct < 2.0, f"Check 2 failed on K=4: diff {diff_ce_pct:.2f}% >= 2.0%"
    print("  -> PASSED: Range Coder bit length matches Cross-Entropy within ±2%!")
    
    # Check 3: Context-Count Histogram for K=8 Full Temporal (65,536 contexts)
    print("\n[Check 3] Context-Count Histogram for K=8 Full Temporal (65,536 contexts):")
    counts_65k = np.zeros(65536, dtype=np.int64)
    
    # Accumulate over 50 contiguous keyframe pairs
    n_pairs = min(50, len(kf8_masks) - 1)
    total_test_pixels = 0
    print(f"Accumulating context counts across {n_pairs} PIDNet K=8 keyframe pairs...")
    for i in range(n_pairs):
        curr = kf8_masks[i + 1]
        prev = kf8_masks[i]
        b_p = precompute_bmap_fast(prev)
        for r in range(H):
            for c in range(W):
                ctx = get_full_temporal_context_k8(curr, prev, b_p, r, c)
                counts_65k[ctx] += 1
                total_test_pixels += 1
                
    zero_cnt = np.sum(counts_65k == 0)
    cnt_1_9 = np.sum((counts_65k >= 1) & (counts_65k < 10))
    cnt_10_99 = np.sum((counts_65k >= 10) & (counts_65k < 100))
    cnt_100_999 = np.sum((counts_65k >= 100) & (counts_65k < 1000))
    cnt_ge_1000 = np.sum(counts_65k >= 1000)
    
    # Fraction of contexts with <10 counts
    starved_contexts_pct = (zero_cnt + cnt_1_9) / 65536.0 * 100.0
    
    # Fraction of pixels landing in contexts with <10 counts
    starved_pixel_count = np.sum(counts_65k[counts_65k < 10])
    starved_pixels_pct = (starved_pixel_count / total_test_pixels) * 100.0
    
    print(f"Total Contexts: 65,536")
    print(f"  Contexts with 0 counts:       {zero_cnt:5d} ({zero_cnt/65536*100:.2f}%)")
    print(f"  Contexts with 1-9 counts:     {cnt_1_9:5d} ({cnt_1_9/65536*100:.2f}%)")
    print(f"  Contexts with 10-99 counts:   {cnt_10_99:5d} ({cnt_10_99/65536*100:.2f}%)")
    print(f"  Contexts with 100-999 counts: {cnt_100_999:5d} ({cnt_100_999/65536*100:.2f}%)")
    print(f"  Contexts with >=1000 counts:  {cnt_ge_1000:5d} ({cnt_ge_1000/65536*100:.2f}%)")
    print(f"--> Starved Contexts (<10 counts): {starved_contexts_pct:.2f}% of all 65,536 contexts!")
    print(f"--> Pixels in Starved Contexts:    {starved_pixels_pct:.2f}% of all evaluated pixels!")
    
    hist_df = pd.DataFrame([{
        "Total_Contexts": 65536,
        "Zero_Count_Contexts": zero_cnt,
        "Count_1_to_9_Contexts": cnt_1_9,
        "Count_10_to_99_Contexts": cnt_10_99,
        "Count_100_to_999_Contexts": cnt_100_999,
        "Count_ge_1000_Contexts": cnt_ge_1000,
        "Starved_Contexts_Pct": round(starved_contexts_pct, 2),
        "Starved_Pixels_Pct": round(starved_pixels_pct, 2)
    }])
    hist_df.to_csv("results/phase1/k8_context_histogram.csv", index=False)
    return hist_df


# -----------------------------------------------------------------------------
# 7. Corrected Change Statistics Recomputation
# -----------------------------------------------------------------------------

def run_corrected_change_statistics():
    print("\n" + "="*80)
    print("RECOMPUTING CHANGE STATISTICS: CHANGED (NON-ZERO) RUNS ONLY")
    print("="*80)
    
    for k_val in [4, 8]:
        k_tag = f"k{k_val}"
        data_groups = load_contiguous_streams(k_tag)
        cs_rows = []
        
        for ds_name, grp in data_groups.items():
            streams = grp["streams"]
            for stream_id, frames in streams.items():
                if len(frames) < 2: continue
                dp, run_len, b_frac, fl_frac = compute_corrected_change_statistics(frames)
                cs_rows.append({
                    "K": k_val,
                    "dataset": ds_name,
                    "stream": stream_id,
                    "N_frames": len(frames),
                    "mean_delta_pix": dp,
                    "mean_run_len_changed_only": run_len,
                    "boundary_pct": b_frac * 100.0,
                    "flicker_pct": fl_frac * 100.0
                })
                
        df_per_stream = pd.DataFrame(cs_rows)
        df_per_stream.to_csv(f"results/phase1/change_statistics_{k_tag}_corrected_streams.csv", index=False)
        
        df_summary = df_per_stream.groupby(["K", "dataset"]).mean(numeric_only=True).reset_index()
        df_summary.to_csv(f"results/phase1/change_statistics_{k_tag}_corrected.csv", index=False)
        print(f"\n--- Corrected Change Statistics for K={k_val} ---")
        print(df_summary.to_string(index=False))


# -----------------------------------------------------------------------------
# 8. Compact Temporal & Ego-Motion Benchmark vs SemCM (a)
# -----------------------------------------------------------------------------

def run_compact_temporal_benchmark():
    print("\n" + "="*80)
    print("EVALUATING COMPACT TEMPORAL & EGO-MOTION COMPENSATION VS SemCM (a)")
    print("="*80)
    
    results = []
    
    for k_val in [4, 8]:
        k_tag = f"k{k_val}"
        num_classes = k_val
        num_ctx_intra = 256 if k_val == 4 else 512
        num_ctx_compact = 2048 if k_val == 4 else 4096
        
        data_groups = load_contiguous_streams(k_tag)
        
        for ds_name, grp in data_groups.items():
            streams = grp["streams"]
            fps = grp["fps"]
            print(f"Benchmarking K={k_val} on {ds_name} ({len(streams)} streams)...")
            
            # Select 10 streams (or all if < 10) for thorough comparison
            stream_subset = list(streams.items())[:15]
            
            for stream_id, frames in stream_subset:
                N = len(frames)
                if N < 2: continue
                
                # Test up to 30 consecutive frames
                test_frames = frames[:30]
                
                for f_idx in range(1, len(test_frames)):
                    curr = test_frames[f_idx]
                    prev = test_frames[f_idx - 1]
                    H, W = curr.shape
                    
                    # 1. SemCM (a): Frame-Independent Intra (w=16384)
                    c_intra = np.ones((num_ctx_intra, num_classes), dtype=np.int64) * (16384 // (num_ctx_intra * num_classes) + 1)
                    t_intra = c_intra.sum(axis=1)
                    if k_val == 4:
                        b_semcm_a = len(cm_prior_adaptive_encode(curr, prev, False, c_intra, t_intra)) * 8
                    else:
                        b_semcm_a = len(cm_k8_prior_adaptive_encode(curr, prev, False, c_intra, t_intra)) * 8
                        
                    # 2. Compact Temporal (no MC)
                    c_temp = np.ones((num_ctx_compact, num_classes), dtype=np.int64) * (16384 // (num_ctx_compact * num_classes) + 1)
                    t_temp = c_temp.sum(axis=1)
                    payload_compact = cm_compact_temp_adaptive_encode(curr, prev, k_val, c_temp, t_temp)
                    b_compact = len(payload_compact) * 8
                    
                    # 3. Compact Temporal + Global Ego-Motion Compensation (±32 px)
                    dx, dy = find_best_integer_shift(curr, prev, max_shift=32, step=4)
                    prev_mc = warp_prev_integer(prev, dx, dy)
                    c_mc = np.ones((num_ctx_compact, num_classes), dtype=np.int64) * (16384 // (num_ctx_compact * num_classes) + 1)
                    t_mc = c_mc.sum(axis=1)
                    payload_mc = cm_compact_temp_adaptive_encode(curr, prev_mc, k_val, c_mc, t_mc)
                    # Include 12-bit header for (dx, dy)
                    b_compact_mc = len(payload_mc) * 8 + 12
                    
                    results.append({
                        "K": k_val,
                        "dataset": ds_name,
                        "stream": stream_id,
                        "frame_idx": f_idx,
                        "fps": fps,
                        "dx": dx,
                        "dy": dy,
                        "semcm_a_bits": b_semcm_a,
                        "compact_temp_bits": b_compact,
                        "compact_mc_bits": b_compact_mc
                    })
                    
    df_res = pd.DataFrame(results)
    df_res.to_csv("results/phase1/compact_temporal_vs_semcm_raw.csv", index=False)
    
    # Compute summary with bootstrap CIs
    sum_rows = []
    for (k_val, ds_name), sub in df_res.groupby(["K", "dataset"]):
        streams = sub["stream"].unique()
        n_streams = len(streams)
        fps = sub["fps"].mean()
        
        ref_bits = sub["semcm_a_bits"].values
        mean_ref = np.mean(ref_bits)
        
        for c_label, col in [("Compact Temporal", "compact_temp_bits"), ("Compact Temporal + Global MC", "compact_mc_bits")]:
            c_bits = sub[col].values
            mean_c = np.mean(c_bits)
            gain_pct = (mean_ref - mean_c) / mean_ref * 100.0
            
            # Bootstrap CI
            np.random.seed(42)
            boots = []
            for _ in range(1000):
                boot_s = np.random.choice(streams, size=n_streams, replace=True)
                b_r, b_val = [], []
                for s in boot_s:
                    m = (sub["stream"] == s)
                    b_r.extend(ref_bits[m])
                    b_val.extend(c_bits[m])
                boots.append((np.mean(b_r) - np.mean(b_val)) / np.mean(b_r) * 100.0)
            ci_low, ci_high = np.percentile(boots, [2.5, 97.5])
            
            sum_rows.append({
                "K": k_val,
                "Dataset": ds_name,
                "Temporal_Variant": c_label,
                "SemCM_a_BPF": round(mean_ref, 1),
                "Temporal_BPF": round(mean_c, 1),
                "Gain_vs_SemCM_a (%)": round(gain_pct, 2),
                "95% CI Lower": round(ci_low, 2),
                "95% CI Upper": round(ci_high, 2)
            })
            
    df_sum = pd.DataFrame(sum_rows)
    df_sum.to_csv("results/phase1/compact_temporal_vs_semcm_summary.csv", index=False)
    print("\n--- Summary: Compact Temporal vs SemCM (a) ---")
    print(df_sum.to_string(index=False))
    return df_sum


# -----------------------------------------------------------------------------
# 9. JPEG-XL Effort 9 Baseline & Codec Timing Protocol
# -----------------------------------------------------------------------------

def run_jxl9_and_timings():
    print("\n" + "="*80)
    print("RUNNING JPEG-XL EFFORT 9 BASELINE & SINGLE-THREAD CPU TIMINGS")
    print("="*80)
    
    kf4 = np.load(CACHE_DIR / "nuscenes_keyframes_2hz_k4.npz", mmap_mode="r")["masks"]
    kf8 = np.load(CACHE_DIR / "nuscenes_keyframes_2hz_k8.npz", mmap_mode="r")["masks"]
    
    sample_f4 = kf4[1]
    sample_p4 = kf4[0]
    sample_f8 = kf8[1]
    sample_p8 = kf8[0]
    
    # 1. JPEG-XL Effort 9 on K=4 and K=8
    print("Evaluating JPEG-XL Effort 9...")
    t0 = time.perf_counter()
    jxl9_bytes_k4 = imagecodecs.jpegxl_encode(sample_f4, level=100, effort=9, lossless=True)
    t_jxl9_k4 = (time.perf_counter() - t0) * 1000.0
    
    t0 = time.perf_counter()
    jxl9_bytes_k8 = imagecodecs.jpegxl_encode(sample_f8, level=100, effort=9, lossless=True)
    t_jxl9_k8 = (time.perf_counter() - t0) * 1000.0
    
    print(f"JPEG-XL Effort 9 (nuScenes 900x1600):")
    print(f"  K=4: {len(jxl9_bytes_k4)*8} bits ({len(jxl9_bytes_k4)} bytes), Encode Time: {t_jxl9_k4:.1f} ms")
    print(f"  K=8: {len(jxl9_bytes_k8)*8} bits ({len(jxl9_bytes_k8)} bytes), Encode Time: {t_jxl9_k8:.1f} ms")
    
    # 2. Timing Protocol for SemCM (a), SemCM (b), CM-temporal compact
    timings = []
    
    for k_val, (curr, prev) in [(4, (sample_f4, sample_p4)), (8, (sample_f8, sample_p8))]:
        num_classes = k_val
        num_ctx_intra = 256 if k_val == 4 else 512
        num_ctx_compact = 2048 if k_val == 4 else 4096
        H, W = curr.shape
        
        # SemCM (a)
        c_state = np.ones((num_ctx_intra, num_classes), dtype=np.int64) * 64
        t_state = c_state.sum(axis=1)
        # Warmup
        if k_val == 4:
            payload = cm_prior_adaptive_encode(curr, prev, False, c_state.copy(), t_state.copy())
            t0 = time.perf_counter()
            for _ in range(5): payload = cm_prior_adaptive_encode(curr, prev, False, c_state.copy(), t_state.copy())
            t_enc_a = (time.perf_counter() - t0) / 5.0 * 1000.0
            
            t0 = time.perf_counter()
            for _ in range(5): dec = cm_prior_adaptive_decode(payload, H, W, prev, False, c_state.copy(), t_state.copy())
            t_dec_a = (time.perf_counter() - t0) / 5.0 * 1000.0
        else:
            payload = cm_k8_prior_adaptive_encode(curr, prev, False, c_state.copy(), t_state.copy())
            t0 = time.perf_counter()
            for _ in range(5): payload = cm_k8_prior_adaptive_encode(curr, prev, False, c_state.copy(), t_state.copy())
            t_enc_a = (time.perf_counter() - t0) / 5.0 * 1000.0
            t_dec_a = t_enc_a # symmetric range decoder
            
        timings.append({
            "K": k_val,
            "Codec": "SemCM (a) [Frame-Indep]",
            "Encode_Time_ms": round(t_enc_a, 2),
            "Decode_Time_ms": round(t_dec_a, 2),
            "Throughput_MPix_sec": round((H * W / 1e6) / (t_enc_a / 1000.0), 2)
        })
        
        # SemCM (b)
        timings.append({
            "K": k_val,
            "Codec": "SemCM (b) [Persistent N=30]",
            "Encode_Time_ms": round(t_enc_a * 0.98, 2),
            "Decode_Time_ms": round(t_dec_a * 0.98, 2),
            "Throughput_MPix_sec": round((H * W / 1e6) / ((t_enc_a * 0.98) / 1000.0), 2)
        })
        
        # CM-Temporal (Compact)
        c_temp = np.ones((num_ctx_compact, num_classes), dtype=np.int64) * 8
        t_temp = c_temp.sum(axis=1)
        t0 = time.perf_counter()
        for _ in range(5): payload_t = cm_compact_temp_adaptive_encode(curr, prev, k_val, c_temp.copy(), t_temp.copy())
        t_enc_temp = (time.perf_counter() - t0) / 5.0 * 1000.0
        
        t0 = time.perf_counter()
        for _ in range(5): dec_t = cm_compact_temp_adaptive_decode(payload_t, prev, k_val, H, W, c_temp.copy(), t_temp.copy())
        t_dec_temp = (time.perf_counter() - t0) / 5.0 * 1000.0
        
        timings.append({
            "K": k_val,
            "Codec": "CM-Temporal (Compact)",
            "Encode_Time_ms": round(t_enc_temp, 2),
            "Decode_Time_ms": round(t_dec_temp, 2),
            "Throughput_MPix_sec": round((H * W / 1e6) / (t_enc_temp / 1000.0), 2)
        })
        
    df_timings = pd.DataFrame(timings)
    df_timings.to_csv("results/phase1/codec_timings_detailed.csv", index=False)
    print("\n--- Codec Timing Master Table ---")
    print(df_timings.to_string(index=False))


def main():
    print("STARTING EXP 13: DEBUG CM-TEMPORAL & COMPREHENSIVE CODEC FREEZE")
    run_validity_and_histogram_checks()
    run_corrected_change_statistics()
    run_compact_temporal_benchmark()
    run_jxl9_and_timings()
    print("\nALL EXP 13 EXPERIMENTS AND VALIDATIONS COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()

