"""
Phase 1b - Decisive Experiment:
Empirical Conditional Entropy and Real Lossless Context-Model Coders.

1. Estimate H(F_{t+1} | causal spatial context) [CM-Intra]
2. Estimate H(F_{t+1} | causal spatial context + F_t(i,j) + F_t 3x3 neighbourhood) [CM-Temporal]
   Trained on 50% split (5 nuScenes scenes, 3 field clips)
   Evaluated on 50% test split (remaining 5 nuScenes scenes, 3 field clips)
3. Bit-exact Lossless Context-Model Coding (CM-Intra and CM-Temporal) in Numba
4. Comparison against FullMap_Zstd19 (best intra baseline)
5. Plain statement of % bit saving of CM-Temporal over best intra codec.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(line_buffering=True)

import time
import json
import numpy as np
import pandas as pd
from numba import njit, int64, float64

from carle.fast_codecs import (
    get_spatial_context_numba,
    get_temporal_context_numba,
    ContextModelCoder,
    cm_encode_numba,
    cm_decode_numba
)
from carle.codecs import FullMapZstd

CACHE_DIR = Path("results/phase1/cache")
RESULTS_DIR = Path("results/phase1")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 50/50 Splits
NUSC_TRAIN_SCENES = {"scene-0061", "scene-0103", "scene-0553", "scene-0655", "scene-0757"}
NUSC_TEST_SCENES = {"scene-0796", "scene-0916", "scene-1077", "scene-1094", "scene-1100"}

FIELD_TRAIN_CLIPS = {"india_bolpur", "malaysia", "dhaka_day"}
FIELD_TEST_CLIPS = {"sikkim", "dhaka_night", "bhubaneswar_night"}

@njit(fastmath=True)
def accumulate_counts_numba(mask_curr: np.ndarray, mask_prev: np.ndarray, counts_intra: np.ndarray, counts_temp: np.ndarray):
    H, W = mask_curr.shape
    for r in range(H):
        for c in range(W):
            sym = int(mask_curr[r, c] - 1)
            ctx_intra = get_spatial_context_numba(mask_curr, r, c)
            ctx_temp = get_temporal_context_numba(mask_curr, mask_prev, r, c)
            counts_intra[ctx_intra, sym] += 1
            counts_temp[ctx_temp, sym] += 1

@njit(fastmath=True)
def compute_cross_entropy_numba(mask_curr: np.ndarray, mask_prev: np.ndarray, log_probs_intra: np.ndarray, log_probs_temp: np.ndarray):
    H, W = mask_curr.shape
    bits_intra = 0.0
    bits_temp = 0.0
    for r in range(H):
        for c in range(W):
            sym = int(mask_curr[r, c] - 1)
            ctx_intra = get_spatial_context_numba(mask_curr, r, c)
            ctx_temp = get_temporal_context_numba(mask_curr, mask_prev, r, c)
            bits_intra -= log_probs_intra[ctx_intra, sym]
            bits_temp -= log_probs_temp[ctx_temp, sym]
    return bits_intra, bits_temp

def build_cum_freqs(counts: np.ndarray, total_scale: int = 16384):
    num_ctx, num_sym = counts.shape
    smoothed = counts.astype(np.float64) + 1.0
    probs = smoothed / np.sum(smoothed, axis=1, keepdims=True)
    rem_scale = total_scale - num_sym
    
    cum_freqs = np.zeros((num_ctx, num_sym + 1), dtype=np.int64)
    for c in range(num_ctx):
        freqs = 1 + np.round(probs[c] * rem_scale).astype(np.int64)
        diff = total_scale - np.sum(freqs)
        max_idx = np.argmax(freqs)
        freqs[max_idx] += diff
        assert np.all(freqs >= 1)
        assert np.sum(freqs) == total_scale
        
        cum = 0
        for s in range(num_sym):
            cum_freqs[c, s] = cum
            cum += freqs[s]
        cum_freqs[c, num_sym] = total_scale
            
    log_probs = np.log2(probs)
    return cum_freqs, log_probs

def run_experiment_on_pairs(dataset_label: str, train_pairs_with_masks: list, test_pairs_with_masks: list):
    print(f"\n{'='*70}\nDecisive Experiment: {dataset_label}\nTrain pairs: {len(train_pairs_with_masks)}, Test pairs: {len(test_pairs_with_masks)}\n{'='*70}", flush=True)
    
    counts_intra = np.zeros((256, 4), dtype=np.int64)
    counts_temp = np.zeros((2048, 4), dtype=np.int64)
    
    # Train
    t0 = time.perf_counter()
    for m0, m1, meta in train_pairs_with_masks:
        accumulate_counts_numba(m1, m0, counts_intra, counts_temp)
    print(f"  Training completed in {time.perf_counter() - t0:.2f} s.", flush=True)
    
    cum_intra, logp_intra = build_cum_freqs(counts_intra)
    cum_temp, logp_temp = build_cum_freqs(counts_temp)
    
    cm_intra = ContextModelCoder(mode="intra", trained_cum_freqs=cum_intra)
    cm_temp = ContextModelCoder(mode="temporal", trained_cum_freqs=cum_temp)
    zstd19 = FullMapZstd(level=19)
    
    records = []
    # Test on up to 500 representative test pairs to keep runtime fast and rigorous
    eval_test = test_pairs_with_masks[:500] if len(test_pairs_with_masks) > 500 else test_pairs_with_masks
    print(f"  Evaluating {len(eval_test)} test pairs...", flush=True)
    
    for m0, m1, meta in eval_test:
        H, W = m1.shape
        # Cross-entropy
        b_intra, b_temp = compute_cross_entropy_numba(m1, m0, logp_intra, logp_temp)
        h_intra_bpf = b_intra / 1024.0
        h_temp_bpf = b_temp / 1024.0
        
        # Zstd19 baseline
        t_enc = time.perf_counter()
        z_bytes = zstd19.encode(m1)
        z_enc_ms = (time.perf_counter() - t_enc) * 1000.0
        t_dec = time.perf_counter()
        z_rec = zstd19.decode(z_bytes)
        z_dec_ms = (time.perf_counter() - t_dec) * 1000.0
        assert np.array_equal(z_rec, m1), "Zstd19 fail"
        z_bpf = len(z_bytes) * 8 / 1024.0
        
        # CM-Intra coder
        cm_intra.prev_mask = m0.copy()
        t_enc = time.perf_counter()
        cmi_bytes = cm_intra.encode(m1)
        cmi_enc_ms = (time.perf_counter() - t_enc) * 1000.0
        cm_intra.prev_mask = m0.copy()
        t_dec = time.perf_counter()
        cmi_rec = cm_intra.decode(cmi_bytes)
        cmi_dec_ms = (time.perf_counter() - t_dec) * 1000.0
        assert np.array_equal(cmi_rec, m1), "CM-Intra fail"
        cmi_bpf = len(cmi_bytes) * 8 / 1024.0
        
        # CM-Temporal coder
        cm_temp.prev_mask = m0.copy()
        t_enc = time.perf_counter()
        cmt_bytes = cm_temp.encode(m1)
        cmt_enc_ms = (time.perf_counter() - t_enc) * 1000.0
        cm_temp.prev_mask = m0.copy()
        t_dec = time.perf_counter()
        cmt_rec = cm_temp.decode(cmt_bytes)
        cmt_dec_ms = (time.perf_counter() - t_dec) * 1000.0
        assert np.array_equal(cmt_rec, m1), "CM-Temporal fail"
        cmt_bpf = len(cmt_bytes) * 8 / 1024.0
        
        records.append({
            "dataset": dataset_label,
            "scene": meta.get("scene", meta.get("clip_id")),
            "camera": meta.get("camera", "cam0"),
            "delta_t_s": meta.get("delta_t_s", 0.0),
            "delta_pix": meta.get("delta_pix", 0.0),
            "H_intra_bpf": round(h_intra_bpf, 2),
            "H_temp_bpf": round(h_temp_bpf, 2),
            "zstd19_bpf": round(z_bpf, 2),
            "cm_intra_bpf": round(cmi_bpf, 2),
            "cm_temp_bpf": round(cmt_bpf, 2),
            "zstd19_enc_ms": round(z_enc_ms, 3),
            "zstd19_dec_ms": round(z_dec_ms, 3),
            "cm_intra_enc_ms": round(cmi_enc_ms, 3),
            "cm_intra_dec_ms": round(cmi_dec_ms, 3),
            "cm_temp_enc_ms": round(cmt_enc_ms, 3),
            "cm_temp_dec_ms": round(cmt_dec_ms, 3),
        })
        
    df_res = pd.DataFrame(records)
    print(f"  Summary {dataset_label}:")
    print(f"    Zstd19     : {df_res['zstd19_bpf'].mean():.2f} Kb (enc {df_res['zstd19_enc_ms'].mean():.2f} ms)")
    print(f"    CM-Intra   : {df_res['cm_intra_bpf'].mean():.2f} Kb (H={df_res['H_intra_bpf'].mean():.2f} Kb, enc {df_res['cm_intra_enc_ms'].mean():.2f} ms)")
    print(f"    CM-Temporal: {df_res['cm_temp_bpf'].mean():.2f} Kb (H={df_res['H_temp_bpf'].mean():.2f} Kb, enc {df_res['cm_temp_enc_ms'].mean():.2f} ms)")
    saving = (df_res['zstd19_bpf'].mean() - df_res['cm_temp_bpf'].mean()) / df_res['zstd19_bpf'].mean() * 100.0
    print(f"    CM-Temporal bit saving over Zstd19: {saving:.1f}%!")
    return df_res

def main():
    pairs_csv = RESULTS_DIR / "pairs_log.csv"
    if not pairs_csv.exists():
        print(f"Missing {pairs_csv}")
        return
        
    df_pairs = pd.read_csv(pairs_csv)
    all_res = []
    
    # 1. nuScenes Keyframes (2 Hz)
    kf_path = CACHE_DIR / "nuscenes_keyframes_2hz.npz"
    if kf_path.exists():
        kf_masks = np.load(kf_path)["masks"]
        df_kf = df_pairs[df_pairs["dataset"] == "nuScenes_Keyframes_2Hz"]
        train_p = []
        test_p = []
        for _, row in df_kf.iterrows():
            item = (kf_masks[int(row["frame_t"])], kf_masks[int(row["frame_t1"])], row.to_dict())
            if row["scene"] in NUSC_TRAIN_SCENES:
                train_p.append(item)
            elif row["scene"] in NUSC_TEST_SCENES:
                test_p.append(item)
        res_kf = run_experiment_on_pairs("nuScenes_Keyframes_2Hz", train_p, test_p)
        all_res.append(res_kf)
        
    # 2. nuScenes Sweeps (~12 Hz)
    sweeps_dir = CACHE_DIR / "sweeps"
    if sweeps_dir.exists():
        df_sw = df_pairs[df_pairs["dataset"] == "nuScenes_Sweeps_12Hz"]
        train_p = []
        test_p = []
        # Load scenes
        for s_npz in sorted(sweeps_dir.glob("sweeps_*.npz")):
            s_name = s_npz.stem.replace("sweeps_", "")
            sub_pairs = df_sw[df_sw["scene"] == s_name]
            if len(sub_pairs) == 0:
                continue
            is_train = s_name in NUSC_TRAIN_SCENES
            is_test = s_name in NUSC_TEST_SCENES
            if not is_train and not is_test:
                continue
                
            s_masks = np.load(s_npz)["masks"]
            sub_pairs_samp = sub_pairs.sample(n=min(len(sub_pairs), 100), random_state=42)
            for _, row in sub_pairs_samp.iterrows():
                ft = int(row["frame_t"])
                ft1 = int(row["frame_t1"])
                if ft1 < len(s_masks):
                    item = (s_masks[ft].copy(), s_masks[ft1].copy(), row.to_dict())
                    if is_train:
                        train_p.append(item)
                    elif is_test:
                        test_p.append(item)
            del s_masks
            import gc; gc.collect()
                        
        res_sw = run_experiment_on_pairs("nuScenes_Sweeps_12Hz", train_p, test_p)
        all_res.append(res_sw)
        
    # 3. Field Videos (Native and 10 Hz)
    for cond in ["native", "10hz"]:
        train_p = []
        test_p = []
        for clip_id in FIELD_TRAIN_CLIPS:
            f_npz = CACHE_DIR / f"field_{clip_id}_{cond}_full.npz"
            if f_npz.exists():
                f_masks = np.load(f_npz)["masks"]
                sub_pairs = df_pairs[df_pairs["dataset"] == f"Field_{clip_id}_{cond}"]
                sub_pairs_samp = sub_pairs.sample(n=min(len(sub_pairs), 100), random_state=42)
                for _, row in sub_pairs_samp.iterrows():
                    ft = int(row["frame_t"])
                    ft1 = int(row["frame_t1"])
                    if ft1 < len(f_masks):
                        train_p.append((f_masks[ft].copy(), f_masks[ft1].copy(), row.to_dict()))
                del f_masks; gc.collect()
                        
        for clip_id in FIELD_TEST_CLIPS:
            f_npz = CACHE_DIR / f"field_{clip_id}_{cond}_full.npz"
            if f_npz.exists():
                f_masks = np.load(f_npz)["masks"]
                sub_pairs = df_pairs[df_pairs["dataset"] == f"Field_{clip_id}_{cond}"]
                sub_pairs_samp = sub_pairs.sample(n=min(len(sub_pairs), 100), random_state=42)
                for _, row in sub_pairs_samp.iterrows():
                    ft = int(row["frame_t"])
                    ft1 = int(row["frame_t1"])
                    if ft1 < len(f_masks):
                        test_p.append((f_masks[ft].copy(), f_masks[ft1].copy(), row.to_dict()))
                del f_masks; gc.collect()
                        
        res_f = run_experiment_on_pairs(f"Field_Videos_{cond}", train_p, test_p)
        all_res.append(res_f)
        
    final_df = pd.concat(all_res, ignore_index=True)
    out_file = RESULTS_DIR / "decisive_experiment_results.csv"
    final_df.to_csv(out_file, index=False)
    print(f"\nSaved all decisive experiment results to {out_file}!")

if __name__ == "__main__":
    main()
