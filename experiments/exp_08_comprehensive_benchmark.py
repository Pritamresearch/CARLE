"""
Phase 1b Comprehensive Benchmark:
- Same-frame comparison across all codecs
- 2-fold scene/clip-level cross-validation for pretrained CM (100% out-of-sample evaluation)
- Pretrained table size tracking
- Adaptive CM (CM-intra-adaptive, CM-temporal-adaptive)
- Baselines: JPEG-XL (effort 7 & 9), WebP lossless (method 6), JPEG-LS (CharLS), Zstandard level 19
- Motion-Compensated CM: CM-MC-global (integer shift +/-32), CM-MC-block (32x32 & 16x16)
- Flicker-Aware CM (2-frame history F_t, F_{t-1})
- Deciding metric: % gain over CM-intra (with clustered bootstrap 95% CIs)
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import time
import struct
import io
import gc
from collections import defaultdict
import numpy as np
import pandas as pd
from PIL import Image
import imagecodecs
import zstandard as zstd

# Enforce single-thread CPU
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from numba import njit
from carle.enhanced_codecs import (
    cm_static_encode_numba, cm_static_decode_numba,
    cm_adaptive_encode_numba, cm_adaptive_decode_numba,
    search_global_motion, warp_global,
    block_motion_estimation, warp_block_mvs,
    cm_flicker_encode_numba, cm_flicker_decode_numba,
    get_spatial_context_fast, precompute_bmap_fast
)

@njit(fastmath=True)
def accumulate_pair_counts_numba(curr: np.ndarray, prev: np.ndarray, prev2: np.ndarray,
                                 intra_counts: np.ndarray, temp_counts: np.ndarray, flicker_counts: np.ndarray):
    H, W = curr.shape
    b_map = precompute_bmap_fast(prev)
    for r in range(H):
        for c in range(W):
            sym = int(curr[r, c] - 1)
            spat = get_spatial_context_fast(curr, r, c)
            intra_counts[spat, sym] += 1
            
            p_val = (prev[r, c] - 1) & 3
            is_b = b_map[r, c]
            temp_ctx = spat | ((p_val | (is_b << 2)) << 8)
            temp_counts[temp_ctx, sym] += 1
            
            is_fl = 1 if prev[r, c] != prev2[r, c] else 0
            fl_ctx = spat | ((p_val | (is_fl << 2)) << 8)
            flicker_counts[fl_ctx, sym] += 1


def build_cum_freqs_from_counts(counts: np.ndarray) -> np.ndarray:
    num_contexts = counts.shape[0]
    smoothed = counts.astype(np.float64) + 1.0 # Laplace smoothing
    totals = smoothed.sum(axis=1, keepdims=True)
    probs = smoothed / totals
    
    scaled = np.floor(probs * 16380).astype(np.int64)
    scaled = np.maximum(scaled, 1)
    
    diffs = 16384 - scaled.sum(axis=1)
    for i in range(num_contexts):
        d = diffs[i]
        if d > 0:
            max_idx = np.argmax(probs[i])
            scaled[i, max_idx] += d
        elif d < 0:
            while d < 0:
                max_idx = np.argmax(scaled[i])
                if scaled[i, max_idx] > 1:
                    scaled[i, max_idx] -= 1
                    d += 1
                else:
                    break
                    
    cum_freqs = np.zeros((num_contexts, 5), dtype=np.uint16)
    for i in range(num_contexts):
        cum_freqs[i, 0] = 0
        for s in range(4):
            cum_freqs[i, s + 1] = cum_freqs[i, s] + scaled[i, s]
            
    return cum_freqs


def train_models_on_pairs(pair_list: list):
    intra_counts = np.zeros((256, 4), dtype=np.int64)
    temp_counts = np.zeros((2048, 4), dtype=np.int64)
    flicker_counts = np.zeros((2048, 4), dtype=np.int64)
    
    for pair in pair_list:
        curr = pair["curr"]
        prev = pair["prev"]
        prev2 = pair["prev2"] if pair["prev2"] is not None else prev
        accumulate_pair_counts_numba(curr, prev, prev2, intra_counts, temp_counts, flicker_counts)
                    
    cf_intra = build_cum_freqs_from_counts(intra_counts)
    cf_temp = build_cum_freqs_from_counts(temp_counts)
    cf_flicker = build_cum_freqs_from_counts(flicker_counts)
    
    return cf_intra, cf_temp, cf_flicker


def run_benchmark():
    print("=" * 80)
    print("Phase 1b Comprehensive Lossless Benchmark across Identical Frames")
    print("=" * 80)
    
    cache_dir = Path("results/phase1/cache")
    out_dir = Path("results/phase1")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    nuscenes_set_a = {"scene-0061", "scene-0103", "scene-0553", "scene-0655", "scene-0757"}
    field_set_a = {"india_bolpur", "malaysia", "sikkim"}
    
    np.random.seed(42)
    
    # -------------------------------------------------------------------------
    # 1. Load NuScenes Keyframes (Sample 100 identical pairs: 10 per scene)
    # -------------------------------------------------------------------------
    print("Loading nuScenes keyframes...")
    with open(cache_dir / "nuscenes_keyframes_2hz_meta.json") as f:
        kf_meta = json.load(f)
        
    # Group metadata indices by scene and camera
    kf_by_sc_cam = defaultdict(lambda: defaultdict(list))
    for idx, m in enumerate(kf_meta):
        kf_by_sc_cam[m["scene"]][m["camera"]].append((m["timestamp"], idx))
        
    # Pick 10 pairs per scene (10 scenes * 10 = 100 pairs)
    needed_kf_indices = set()
    kf_pair_defs = []
    
    for sc, cams in sorted(kf_by_sc_cam.items()):
        sc_pairs = []
        for cam, frames in cams.items():
            frames.sort(key=lambda x: x[0])
            for i in range(1, len(frames)):
                t0, idx0 = frames[i - 1]
                t1, idx1 = frames[i]
                idx2 = frames[i - 2][1] if i >= 2 else None
                dt = (t1 - t0) / 1e6
                if 0.40 <= dt <= 0.60:
                    sc_pairs.append((sc, cam, dt, idx1, idx0, idx2))
        # Pick 10 stratified
        n_take = min(len(sc_pairs), 10)
        chosen = [sc_pairs[i] for i in np.random.choice(len(sc_pairs), n_take, replace=False)]
        for item in chosen:
            kf_pair_defs.append(item)
            needed_kf_indices.add(item[3]) # curr
            needed_kf_indices.add(item[4]) # prev
            if item[5] is not None:
                needed_kf_indices.add(item[5]) # prev2
                
    # Now load ONLY the needed slices from nuScenes keyframes npz (read once!)
    kf_npz = np.load(cache_dir / "nuscenes_keyframes_2hz.npz")
    all_kf = kf_npz["masks"]
    kf_cache_masks = {}
    for idx in needed_kf_indices:
        kf_cache_masks[idx] = all_kf[idx].copy()
    del all_kf
    del kf_npz
    gc.collect()
    
    kf_pairs = []
    for sc, cam, dt, idx1, idx0, idx2 in kf_pair_defs:
        kf_pairs.append({
            "dataset": "nuScenes_Keyframes_2Hz",
            "scene": sc,
            "camera": cam,
            "delta_t": dt,
            "curr": kf_cache_masks[idx1],
            "prev": kf_cache_masks[idx0],
            "prev2": kf_cache_masks[idx2] if idx2 is not None else None,
            "fold": 1 if sc in nuscenes_set_a else 2
        })
    print(f"Loaded {len(kf_pairs)} identical nuScenes keyframe pairs.")
    
    # -------------------------------------------------------------------------
    # 2. Load NuScenes Sweeps (Sample 100 identical pairs: 10 per scene)
    # -------------------------------------------------------------------------
    print("Loading nuScenes sweeps...")
    sweeps_pairs = []
    with open(cache_dir / "nuscenes_sweeps_12hz_manifest.json") as f:
        sw_manifest = json.load(f)
        
    for entry in sw_manifest:
        sc = entry["scene"]
        meta_p = cache_dir / entry["meta_file"]
        with open(meta_p) as mf:
            sw_meta = json.load(mf)
            
        sw_by_cam = defaultdict(list)
        for idx, m in enumerate(sw_meta):
            sw_by_cam[m["camera"]].append((m["timestamp"], idx))
            
        sc_pairs = []
        for cam, frames in sw_by_cam.items():
            frames.sort(key=lambda x: x[0])
            for i in range(1, len(frames)):
                t0, idx0 = frames[i - 1]
                t1, idx1 = frames[i]
                idx2 = frames[i - 2][1] if i >= 2 else None
                dt = (t1 - t0) / 1e6
                if 0.05 <= dt <= 0.12:
                    sc_pairs.append((sc, cam, dt, idx1, idx0, idx2))
                    
        n_take = min(len(sc_pairs), 10)
        chosen = [sc_pairs[i] for i in np.random.choice(len(sc_pairs), n_take, replace=False)]
        
        needed_indices = set()
        for item in chosen:
            needed_indices.add(item[3])
            needed_indices.add(item[4])
            if item[5] is not None:
                needed_indices.add(item[5])
                
        # Load npz for this scene only (read once)
        npz_p = cache_dir / entry["npz_file"]
        sc_npz = np.load(npz_p)
        all_sc = sc_npz["masks"]
        sc_masks = {}
        for idx in needed_indices:
            sc_masks[idx] = all_sc[idx].copy()
        del all_sc
        del sc_npz
        gc.collect()
        
        for sc_name, cam, dt, idx1, idx0, idx2 in chosen:
            sweeps_pairs.append({
                "dataset": "nuScenes_Sweeps_12Hz",
                "scene": sc_name,
                "camera": cam,
                "delta_t": dt,
                "curr": sc_masks[idx1],
                "prev": sc_masks[idx0],
                "prev2": sc_masks[idx2] if idx2 is not None else None,
                "fold": 1 if sc_name in nuscenes_set_a else 2
            })
    print(f"Loaded {len(sweeps_pairs)} identical nuScenes sweep pairs.")
    
    # -------------------------------------------------------------------------
    # 3. Load Field Videos (Native and 10 Hz: 15 per clip * 6 = 90 pairs each)
    # -------------------------------------------------------------------------
    print("Loading Field Videos...")
    field_native_pairs = []
    field_10hz_pairs = []
    clip_names = ["india_bolpur", "malaysia", "sikkim", "dhaka_day", "dhaka_night", "bhubaneswar_night"]
    
    for clip in clip_names:
        # Native
        p_nat = cache_dir / f"field_{clip}_native_full.npz"
        nat_data = np.load(p_nat)["masks"]
        N_nat = len(nat_data)
        indices = np.random.choice(range(2, N_nat), min(15, N_nat - 2), replace=False)
        for idx in sorted(indices):
            field_native_pairs.append({
                "dataset": "Field_Videos_native",
                "scene": clip,
                "camera": "MAIN",
                "delta_t": 0.033,
                "curr": nat_data[idx].copy(),
                "prev": nat_data[idx - 1].copy(),
                "prev2": nat_data[idx - 2].copy(),
                "fold": 1 if clip in field_set_a else 2
            })
        del nat_data
        gc.collect()
        
        # 10 Hz
        p_10 = cache_dir / f"field_{clip}_10hz_full.npz"
        data_10 = np.load(p_10)["masks"]
        N_10 = len(data_10)
        indices = np.random.choice(range(2, N_10), min(15, N_10 - 2), replace=False)
        for idx in sorted(indices):
            field_10hz_pairs.append({
                "dataset": "Field_Videos_10hz",
                "scene": clip,
                "camera": "MAIN",
                "delta_t": 0.100,
                "curr": data_10[idx].copy(),
                "prev": data_10[idx - 1].copy(),
                "prev2": data_10[idx - 2].copy(),
                "fold": 1 if clip in field_set_a else 2
            })
        del data_10
        gc.collect()
        
    print(f"Loaded {len(field_native_pairs)} identical Field Native pairs.")
    print(f"Loaded {len(field_10hz_pairs)} identical Field 10 Hz pairs.")
    
    datasets_to_run = {
        "nuScenes_Keyframes_2Hz": kf_pairs,
        "nuScenes_Sweeps_12Hz": sweeps_pairs,
        "Field_Videos_native": field_native_pairs,
        "Field_Videos_10hz": field_10hz_pairs
    }
    
    # -------------------------------------------------------------------------
    # 4. Train 2-Fold Pretrained Context Models
    # -------------------------------------------------------------------------
    print("\n--- Training 2-Fold Pretrained Context Models (Strict Out-of-Sample) ---")
    
    # NuScenes Keyframes
    tr_kf_f1 = [p for p in kf_pairs if p["fold"] == 1]
    tr_kf_f2 = [p for p in kf_pairs if p["fold"] == 2]
    cf_kf_intra_f1, cf_kf_temp_f1, cf_kf_fl_f1 = train_models_on_pairs(tr_kf_f1)
    cf_kf_intra_f2, cf_kf_temp_f2, cf_kf_fl_f2 = train_models_on_pairs(tr_kf_f2)
    
    # NuScenes Sweeps
    tr_sw_f1 = [p for p in sweeps_pairs if p["fold"] == 1]
    tr_sw_f2 = [p for p in sweeps_pairs if p["fold"] == 2]
    cf_sw_intra_f1, cf_sw_temp_f1, cf_sw_fl_f1 = train_models_on_pairs(tr_sw_f1)
    cf_sw_intra_f2, cf_sw_temp_f2, cf_sw_fl_f2 = train_models_on_pairs(tr_sw_f2)
    
    # Field Videos
    tr_fld_f1 = [p for p in field_native_pairs if p["fold"] == 1]
    tr_fld_f2 = [p for p in field_native_pairs if p["fold"] == 2]
    cf_fld_intra_f1, cf_fld_temp_f1, cf_fld_fl_f1 = train_models_on_pairs(tr_fld_f1)
    cf_fld_intra_f2, cf_fld_temp_f2, cf_fld_fl_f2 = train_models_on_pairs(tr_fld_f2)
    
    models = {
        "nuScenes_Keyframes_2Hz": {
            1: (cf_kf_intra_f2, cf_kf_temp_f2, cf_kf_fl_f2), # trained on fold 2, tests fold 1
            2: (cf_kf_intra_f1, cf_kf_temp_f1, cf_kf_fl_f1)  # trained on fold 1, tests fold 2
        },
        "nuScenes_Sweeps_12Hz": {
            1: (cf_sw_intra_f2, cf_sw_temp_f2, cf_sw_fl_f2),
            2: (cf_sw_intra_f1, cf_sw_temp_f1, cf_sw_fl_f1)
        },
        "Field_Videos_native": {
            1: (cf_fld_intra_f2, cf_fld_temp_f2, cf_fld_fl_f2),
            2: (cf_fld_intra_f1, cf_fld_temp_f1, cf_fld_fl_f1)
        },
        "Field_Videos_10hz": {
            1: (cf_fld_intra_f2, cf_fld_temp_f2, cf_fld_fl_f2),
            2: (cf_fld_intra_f1, cf_fld_temp_f1, cf_fld_fl_f1)
        }
    }
    
    table_size_intra = 256 * 5 * 2 # 2560 bytes
    table_size_temp = 2048 * 5 * 2 # 20480 bytes
    print(f"Pretrained table size: CM-Intra = {table_size_intra} B ({table_size_intra/1024:.2f} KB), CM-Temporal = {table_size_temp} B ({table_size_temp/1024:.2f} KB)")
    
    # -------------------------------------------------------------------------
    # 5. Benchmark Execution across Identical Frames
    # -------------------------------------------------------------------------
    cctx19 = zstd.ZstdCompressor(level=19)
    dctx = zstd.ZstdDecompressor()
    
    # Warmup Numba JIT
    dummy = np.random.randint(1, 5, (100, 100), dtype=np.uint8)
    dummy_cf = np.tile(np.array([0, 4096, 8192, 12288, 16384], dtype=np.uint16), (2048, 1))
    cm_static_encode_numba(dummy, dummy, False, dummy_cf[:256])
    cm_static_decode_numba(cm_static_encode_numba(dummy, dummy, False, dummy_cf[:256]), 100, 100, dummy, False, dummy_cf[:256])
    cm_static_encode_numba(dummy, dummy, True, dummy_cf)
    cm_static_decode_numba(cm_static_encode_numba(dummy, dummy, True, dummy_cf), 100, 100, dummy, True, dummy_cf)
    cm_adaptive_encode_numba(dummy, dummy, False)
    cm_adaptive_decode_numba(cm_adaptive_encode_numba(dummy, dummy, False), 100, 100, dummy, False)
    cm_adaptive_encode_numba(dummy, dummy, True)
    cm_adaptive_decode_numba(cm_adaptive_encode_numba(dummy, dummy, True), 100, 100, dummy, True)
    search_global_motion(dummy, dummy, max_shift=8)
    dummy_mvs, _ = block_motion_estimation(dummy, dummy, block_size=16, max_range=4)
    warp_block_mvs(dummy, dummy_mvs, 16)
    cm_flicker_encode_numba(dummy, dummy, dummy, dummy_cf)
    cm_flicker_decode_numba(cm_flicker_encode_numba(dummy, dummy, dummy, dummy_cf), 100, 100, dummy, dummy, dummy_cf)
    print("Warmup complete. Starting identical frame evaluations...\n")
    
    records = []
    
    for ds_name, pair_list in datasets_to_run.items():
        print(f"=== Evaluating {ds_name} ({len(pair_list)} identical pairs) ===")
        t_start_ds = time.time()
        
        for p_idx, pair in enumerate(pair_list):
            curr = pair["curr"]
            prev = pair["prev"]
            prev2 = pair["prev2"] if pair["prev2"] is not None else prev
            H, W = curr.shape
            fold = pair["fold"]
            cf_intra, cf_temp, cf_flick = models[ds_name][fold]
            
            row_common = {
                "dataset": ds_name,
                "scene": pair["scene"],
                "camera": pair["camera"],
                "pair_idx": p_idx,
                "delta_t": pair["delta_t"],
                "delta_pix": float(np.mean(curr != prev))
            }
            
            # 1. Zstandard level 19
            t0 = time.perf_counter()
            c_zstd = cctx19.compress(curr.tobytes())
            t_enc_zstd = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            dec_zstd = np.frombuffer(dctx.decompress(c_zstd), dtype=np.uint8).reshape(H, W)
            t_dec_zstd = (time.perf_counter() - t0) * 1000
            assert np.array_equal(curr, dec_zstd)
            b_zstd = len(c_zstd) * 8 / 1024.0
            
            # 2. JPEG-XL effort 7
            t0 = time.perf_counter()
            c_jxl7 = imagecodecs.jpegxl_encode(curr, distance=0.0, effort=7, numthreads=1)
            t_enc_jxl7 = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            dec_jxl7 = imagecodecs.jpegxl_decode(c_jxl7)
            t_dec_jxl7 = (time.perf_counter() - t0) * 1000
            assert np.array_equal(curr, dec_jxl7)
            b_jxl7 = len(c_jxl7) * 8 / 1024.0
            
            # 3. JPEG-XL effort 9
            t0 = time.perf_counter()
            c_jxl9 = imagecodecs.jpegxl_encode(curr, distance=0.0, effort=9, numthreads=1)
            t_enc_jxl9 = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            dec_jxl9 = imagecodecs.jpegxl_decode(c_jxl9)
            t_dec_jxl9 = (time.perf_counter() - t0) * 1000
            assert np.array_equal(curr, dec_jxl9)
            b_jxl9 = len(c_jxl9) * 8 / 1024.0
            
            # 4. WebP lossless method 6
            pil_img = Image.fromarray(curr, mode="L")
            t0 = time.perf_counter()
            buf = io.BytesIO()
            pil_img.save(buf, format="WEBP", lossless=True, quality=100, method=6)
            c_webp = buf.getvalue()
            t_enc_webp = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            dec_webp = np.array(Image.open(io.BytesIO(c_webp)))
            if dec_webp.ndim == 3: dec_webp = dec_webp[:, :, 0]
            t_dec_webp = (time.perf_counter() - t0) * 1000
            assert np.array_equal(curr, dec_webp)
            b_webp = len(c_webp) * 8 / 1024.0
            
            # 5. JPEG-LS
            t0 = time.perf_counter()
            c_jpls = imagecodecs.jpegls_encode(curr, level=0)
            t_enc_jpls = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            dec_jpls = imagecodecs.jpegls_decode(c_jpls)
            t_dec_jpls = (time.perf_counter() - t0) * 1000
            assert np.array_equal(curr, dec_jpls)
            b_jpls = len(c_jpls) * 8 / 1024.0
            
            # 6. CM-intra (2-Fold Out-of-Sample)
            t0 = time.perf_counter()
            c_cm_intra = cm_static_encode_numba(curr, prev, False, cf_intra)
            t_enc_cm_intra = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            dec_cm_intra = cm_static_decode_numba(c_cm_intra, H, W, prev, False, cf_intra)
            t_dec_cm_intra = (time.perf_counter() - t0) * 1000
            assert np.array_equal(curr, dec_cm_intra)
            b_cm_intra = len(c_cm_intra) * 8 / 1024.0
            
            # 7. CM-temporal (2-Fold Out-of-Sample, Fast Numba)
            t0 = time.perf_counter()
            c_cm_temp = cm_static_encode_numba(curr, prev, True, cf_temp)
            t_enc_cm_temp = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            dec_cm_temp = cm_static_decode_numba(c_cm_temp, H, W, prev, True, cf_temp)
            t_dec_cm_temp = (time.perf_counter() - t0) * 1000
            assert np.array_equal(curr, dec_cm_temp)
            b_cm_temp = len(c_cm_temp) * 8 / 1024.0
            
            # 8. CM-intra-adaptive
            t0 = time.perf_counter()
            c_ad_intra = cm_adaptive_encode_numba(curr, prev, False)
            t_enc_ad_intra = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            dec_ad_intra = cm_adaptive_decode_numba(c_ad_intra, H, W, prev, False)
            t_dec_ad_intra = (time.perf_counter() - t0) * 1000
            assert np.array_equal(curr, dec_ad_intra)
            b_ad_intra = len(c_ad_intra) * 8 / 1024.0
            
            # 9. CM-temporal-adaptive
            t0 = time.perf_counter()
            c_ad_temp = cm_adaptive_encode_numba(curr, prev, True)
            t_enc_ad_temp = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            dec_ad_temp = cm_adaptive_decode_numba(c_ad_temp, H, W, prev, True)
            t_dec_ad_temp = (time.perf_counter() - t0) * 1000
            assert np.array_equal(curr, dec_ad_temp)
            b_ad_temp = len(c_ad_temp) * 8 / 1024.0
            
            # 10. CM-MC-global
            t0 = time.perf_counter()
            dx, dy = search_global_motion(curr, prev, max_shift=32)
            warped_gmc = warp_global(prev, dx, dy)
            c_gmc = cm_static_encode_numba(curr, warped_gmc, True, cf_temp)
            t_enc_gmc = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            warped_gmc_dec = warp_global(prev, dx, dy)
            dec_gmc = cm_static_decode_numba(c_gmc, H, W, warped_gmc_dec, True, cf_temp)
            t_dec_gmc = (time.perf_counter() - t0) * 1000
            assert np.array_equal(curr, dec_gmc)
            b_gmc = (len(c_gmc) + 2) * 8 / 1024.0
            
            # 11. CM-MC-block (32x32)
            t0 = time.perf_counter()
            mvs32, warped_bmc32 = block_motion_estimation(curr, prev, block_size=32, max_range=8)
            c_bmc32 = cm_static_encode_numba(curr, warped_bmc32, True, cf_temp)
            mv_bytes = zstd.compress(mvs32.tobytes(), 3)
            t_enc_bmc = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            mvs32_dec = np.frombuffer(zstd.decompress(mv_bytes), dtype=np.int8).reshape(mvs32.shape)
            warped_bmc_dec = warp_block_mvs(prev, mvs32_dec, 32)
            dec_bmc = cm_static_decode_numba(c_bmc32, H, W, warped_bmc_dec, True, cf_temp)
            t_dec_bmc = (time.perf_counter() - t0) * 1000
            assert np.array_equal(curr, dec_bmc)
            b_bmc = (len(c_bmc32) + len(mv_bytes)) * 8 / 1024.0
            
            # 12. CM-flicker
            t0 = time.perf_counter()
            c_flick = cm_flicker_encode_numba(curr, prev, prev2, cf_flick)
            t_enc_flick = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            dec_flick = cm_flicker_decode_numba(c_flick, H, W, prev, prev2, cf_flick)
            t_dec_flick = (time.perf_counter() - t0) * 1000
            assert np.array_equal(curr, dec_flick)
            b_flick = len(c_flick) * 8 / 1024.0
            
            records.append({
                **row_common,
                "b_zstd19": b_zstd, "enc_zstd19": t_enc_zstd, "dec_zstd19": t_dec_zstd,
                "b_jxl7": b_jxl7, "enc_jxl7": t_enc_jxl7, "dec_jxl7": t_dec_jxl7,
                "b_jxl9": b_jxl9, "enc_jxl9": t_enc_jxl9, "dec_jxl9": t_dec_jxl9,
                "b_webp": b_webp, "enc_webp": t_enc_webp, "dec_webp": t_dec_webp,
                "b_jpls": b_jpls, "enc_jpls": t_enc_jpls, "dec_jpls": t_dec_jpls,
                "b_cm_intra": b_cm_intra, "enc_cm_intra": t_enc_cm_intra, "dec_cm_intra": t_dec_cm_intra,
                "b_cm_temp": b_cm_temp, "enc_cm_temp": t_enc_cm_temp, "dec_cm_temp": t_dec_cm_temp,
                "b_ad_intra": b_ad_intra, "enc_ad_intra": t_enc_ad_intra, "dec_ad_intra": t_dec_ad_intra,
                "b_ad_temp": b_ad_temp, "enc_ad_temp": t_enc_ad_temp, "dec_ad_temp": t_dec_ad_temp,
                "b_gmc": b_gmc, "enc_gmc": t_enc_gmc, "dec_gmc": t_dec_gmc,
                "b_bmc": b_bmc, "enc_bmc": t_enc_bmc, "dec_bmc": t_dec_bmc,
                "b_flicker": b_flick, "enc_flicker": t_enc_flick, "dec_flicker": t_dec_flick,
            })
            
            if (p_idx + 1) % 20 == 0:
                print(f"  Processed {p_idx + 1}/{len(pair_list)} pairs in {time.time() - t_start_ds:.1f}s (CM-Intra={b_cm_intra:.2f} Kb, CM-Temp={b_cm_temp:.2f} Kb, WebP={b_webp:.2f} Kb)")
                
        print(f"Finished {ds_name} in {time.time() - t_start_ds:.1f} s.\n")
        
    df_raw = pd.DataFrame(records)
    df_raw.to_csv(out_dir / "comprehensive_benchmark_raw.csv", index=False)
    print(f"Saved raw per-pair results to {out_dir / 'comprehensive_benchmark_raw.csv'}")
    
    # -------------------------------------------------------------------------
    # 6. Aggregate Statistics & Clustered Bootstrap 95% CIs
    # -------------------------------------------------------------------------
    print("\n--- Computing Aggregates and Clustered Bootstrap 95% CIs ---")
    summary_rows = []
    
    codecs_info = [
        ("FullMap_Zstd19", "b_zstd19", "enc_zstd19", "dec_zstd19", 0),
        ("JPEG-XL_effort7", "b_jxl7", "enc_jxl7", "dec_jxl7", 0),
        ("JPEG-XL_effort9", "b_jxl9", "enc_jxl9", "dec_jxl9", 0),
        ("WebP_lossless", "b_webp", "enc_webp", "dec_webp", 0),
        ("JPEG-LS_CharLS", "b_jpls", "enc_jpls", "dec_jpls", 0),
        ("CM-intra", "b_cm_intra", "enc_cm_intra", "dec_cm_intra", table_size_intra),
        ("CM-temporal", "b_cm_temp", "enc_cm_temp", "dec_cm_temp", table_size_temp),
        ("CM-intra-adaptive", "b_ad_intra", "enc_ad_intra", "dec_ad_intra", 0),
        ("CM-temporal-adaptive", "b_ad_temp", "enc_ad_temp", "dec_ad_temp", 0),
        ("CM-MC-global", "b_gmc", "enc_gmc", "dec_gmc", table_size_temp),
        ("CM-MC-block", "b_bmc", "enc_bmc", "dec_bmc", table_size_temp),
        ("CM-flicker", "b_flicker", "enc_flicker", "dec_flicker", table_size_temp)
    ]
    
    for ds_name, df_ds in df_raw.groupby("dataset"):
        clusters = df_ds["scene"].unique()
        n_clusters = len(clusters)
        
        cm_intra_mean = df_ds["b_cm_intra"].mean()
        
        for c_label, col_b, col_enc, col_dec, tbl_sz in codecs_info:
            b_mean = df_ds[col_b].mean()
            enc_mean = df_ds[col_enc].mean()
            dec_mean = df_ds[col_dec].mean()
            
            gain_pct = ((cm_intra_mean - b_mean) / cm_intra_mean) * 100.0
            
            boot_b = []
            boot_gain = []
            for _ in range(1000):
                boot_clusters = np.random.choice(clusters, n_clusters, replace=True)
                boot_df = pd.concat([df_ds[df_ds["scene"] == c] for c in boot_clusters])
                b_m = boot_df[col_b].mean()
                intra_m = boot_df["b_cm_intra"].mean()
                boot_b.append(b_m)
                g_m = ((intra_m - b_m) / intra_m) * 100.0
                boot_gain.append(g_m)
                
            ci_b_low, ci_b_high = np.percentile(boot_b, [2.5, 97.5])
            ci_g_low, ci_g_high = np.percentile(boot_gain, [2.5, 97.5])
            
            summary_rows.append({
                "dataset": ds_name,
                "codec": c_label,
                "table_bytes": tbl_sz,
                "mean_bpf_Kb": b_mean,
                "ci95_bpf_low": ci_b_low,
                "ci95_bpf_high": ci_b_high,
                "gain_vs_cm_intra_pct": gain_pct,
                "gain_ci95_low": ci_g_low,
                "gain_ci95_high": ci_g_high,
                "enc_ms_cpu": enc_mean,
                "dec_ms_cpu": dec_mean
            })
            
    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(out_dir / "comprehensive_benchmark_summary.csv", index=False)
    print(f"Summary saved to {out_dir / 'comprehensive_benchmark_summary.csv'}")
    print("\nBenchmark successfully complete.")


if __name__ == "__main__":
    run_benchmark()
