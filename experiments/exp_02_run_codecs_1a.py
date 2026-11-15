"""
Phase 1a - Experiment 02: Full Codec Benchmark on Cached Label Maps.

Evaluates all codecs across:
- nuScenes Keyframes (2 Hz)
- nuScenes Sweeps (~12 Hz)
- Field Test Videos (Native fps)
- Field Test Videos (10 Hz subsampled)

All codec execution is strictly CPU single-threaded.
Losslessness is verified on every single frame.
Outputs: results/phase1/results_1a.csv and formatted summary tables.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(line_buffering=True)

import time
import json
import numpy as np
import pandas as pd
import torch

# Force CPU single thread for codec timing
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from carle.codecs import (
    FullMapPNG,
    FullMapZstd,
    DiffMapPNG,
    DiffMapZstd,
    CARLECanonical,
    CARLEv2_rANS,
    CARLEv2_IP,
    XORLegacy,
)

CACHE_DIR = Path("results/phase1/cache")
RESULTS_DIR = Path("results/phase1")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def get_codecs():
    return {
        "FullMap_PNG9": FullMapPNG(level=9),
        "FullMap_Zstd3": FullMapZstd(level=3),
        "FullMap_Zstd19": FullMapZstd(level=19),
        "DiffMap_PNG9": DiffMapPNG(level=9),
        "DiffMap_Zstd3": DiffMapZstd(level=3),
        "DiffMap_Zstd19": DiffMapZstd(level=19),
        "CARLE_Canonical": CARLECanonical(),
        "CARLE_v2_rANS": CARLEv2_rANS(),
        "CARLE_v2_IP_10": CARLEv2_IP(refresh_N=10),
        "CARLE_v2_IP_30": CARLEv2_IP(refresh_N=30),
        "CARLE_v2_IP_inf": CARLEv2_IP(refresh_N=float("inf")),
        "XOR_Legacy": XORLegacy(),
    }

def bootstrap_ci(values, n_boot=1000, ci=95):
    """Compute bootstrap confidence interval for the mean."""
    if len(values) == 0:
        return 0.0, 0.0
    arr = np.array(values)
    boot_means = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_means.append(np.mean(sample))
    low = np.percentile(boot_means, (100 - ci) / 2.0)
    high = np.percentile(boot_means, 100 - (100 - ci) / 2.0)
    return float(low), float(high)

def run_benchmark_on_sequence(dataset_name: str, masks: np.ndarray, meta: list, sample_limit: int = 100):
    N, H, W = masks.shape
    if sample_limit is not None and N > sample_limit:
        masks = masks[:sample_limit]
        meta = meta[:sample_limit]
        N = sample_limit
        
    print(f"\nEvaluating {dataset_name}: {N} frames of {H}x{W}...", flush=True)
    
    raw_2bit_bytes_per_frame = H * W * 2 / 8.0 # log2(C=4)
    raw_2bit_total_bytes = raw_2bit_bytes_per_frame * N
    
    codecs = get_codecs()
    
    results_records = []
    
    # Pre-compute PNG bytes to get accurate baseline sum for CR_vs_PNG
    png_codec = codecs["FullMap_PNG9"]
    png_bytes_list = [len(png_codec.encode(masks[i])) for i in range(N)]
    total_png_bytes = sum(png_bytes_list)
    
    for codec_name, codec in codecs.items():
        if hasattr(codec, "reset"):
            codec.reset()
            
        bytes_list = []
        enc_times = []
        dec_times = []
        lossless_passed = True
        
        prev_mask = None
        for i in range(N):
            m = masks[i]
            
            # Encode timing
            t0 = time.perf_counter_ns()
            payload = codec.encode(m, prev_mask)
            t1 = time.perf_counter_ns()
            enc_ms = (t1 - t0) / 1e6
            enc_times.append(enc_ms)
            
            b_size = len(payload)
            bytes_list.append(b_size)
            
            # Decode timing
            t2 = time.perf_counter_ns()
            dec_mask = codec.decode(payload, prev_mask)
            t3 = time.perf_counter_ns()
            dec_ms = (t3 - t2) / 1e6
            dec_times.append(dec_ms)
            
            # Lossless verification
            if codec_name != "XOR_Legacy":
                if not np.array_equal(dec_mask, m):
                    lossless_passed = False
                prev_mask = dec_mask
            else:
                prev_mask = m
                
        total_bytes = sum(bytes_list)
        bpf_list = [b * 8.0 / 1000.0 for b in bytes_list] # in Kb
        
        mean_bpf = float(np.mean(bpf_list))
        median_bpf = float(np.median(bpf_list))
        p5_bpf = float(np.percentile(bpf_list, 5))
        p95_bpf = float(np.percentile(bpf_list, 95))
        
        ci_low_bpf, ci_high_bpf = bootstrap_ci(bpf_list)
        
        sum_cr_raw2 = float(raw_2bit_total_bytes / total_bytes) if total_bytes > 0 else 0.0
        sum_cr_png = float(total_png_bytes / total_bytes) if total_bytes > 0 else 0.0
        
        mean_enc_ms = float(np.mean(enc_times))
        median_enc_ms = float(np.median(enc_times))
        p95_enc_ms = float(np.percentile(enc_times, 95))
        
        mean_dec_ms = float(np.mean(dec_times))
        median_dec_ms = float(np.median(dec_times))
        p95_dec_ms = float(np.percentile(dec_times, 95))
        
        record = {
            "dataset": dataset_name,
            "codec": codec_name,
            "frames": N,
            "lossless": lossless_passed if codec_name != "XOR_Legacy" else "KNOWN_CORRUPT",
            "bpf_mean_Kb": round(mean_bpf, 2),
            "bpf_median_Kb": round(median_bpf, 2),
            "bpf_p5_Kb": round(p5_bpf, 2),
            "bpf_p95_Kb": round(p95_bpf, 2),
            "bpf_ci95_low": round(ci_low_bpf, 2),
            "bpf_ci95_high": round(ci_high_bpf, 2),
            "CR_vs_raw2": round(sum_cr_raw2, 2),
            "CR_vs_PNG": round(sum_cr_png, 2),
            "enc_ms_mean": round(mean_enc_ms, 3),
            "enc_ms_median": round(median_enc_ms, 3),
            "enc_ms_p95": round(p95_enc_ms, 3),
            "dec_ms_mean": round(mean_dec_ms, 3),
            "dec_ms_median": round(median_dec_ms, 3),
            "dec_ms_p95": round(p95_dec_ms, 3),
        }
        results_records.append(record)
        print(f"  {codec_name:<16}: bpf={mean_bpf:6.2f} Kb | CR_raw2={sum_cr_raw2:5.1f}x | CR_png={sum_cr_png:4.2f}x | enc={mean_enc_ms:5.2f}ms | dec={mean_dec_ms:5.2f}ms | Lossless: {lossless_passed}", flush=True)
        
    return results_records

def main():
    all_results = []
    
    # 1. nuScenes 2 Hz Keyframes
    kf_path = CACHE_DIR / "nuscenes_keyframes_2hz.npz"
    if kf_path.exists():
        data = np.load(kf_path)
        with open(CACHE_DIR / "nuscenes_keyframes_2hz_meta.json") as f:
            meta = json.load(f)
        res = run_benchmark_on_sequence("nuScenes_Keyframes_2Hz", data["masks"], meta, sample_limit=120)
        all_results.extend(res)
        
    # 2. nuScenes ~12 Hz Sweeps
    sw_path = CACHE_DIR / "nuscenes_sweeps_12hz.npz"
    if sw_path.exists():
        data = np.load(sw_path)
        with open(CACHE_DIR / "nuscenes_sweeps_12hz_meta.json") as f:
            meta = json.load(f)
        res = run_benchmark_on_sequence("nuScenes_Sweeps_12Hz", data["masks"], meta, sample_limit=120)
        all_results.extend(res)
        
    # 3. Field Test Videos
    for condition in ["native", "10hz"]:
        for clip_id in ["india_bolpur", "malaysia", "sikkim", "dhaka_day", "dhaka_night", "bhubaneswar_night"]:
            f_path = CACHE_DIR / f"field_{clip_id}_{condition}.npz"
            if f_path.exists():
                data = np.load(f_path)
                with open(CACHE_DIR / f"field_{clip_id}_{condition}_meta.json") as f:
                    meta = json.load(f)
                d_name = f"Field_{clip_id}_{condition}"
                res = run_benchmark_on_sequence(d_name, data["masks"], meta, sample_limit=100)
                all_results.extend(res)

    df_out = pd.DataFrame(all_results)
    csv_path = RESULTS_DIR / "results_1a.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"\n==================================================================")
    print(f"Saved complete Phase 1a results ({len(df_out)} rows) to {csv_path}")
    print(f"==================================================================\n", flush=True)

if __name__ == "__main__":
    main()
