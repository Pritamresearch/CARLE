"""
Dead-Code Counter and High-Resolution Timing Benchmark
Runs on a stratified sample of 300 frames per K (K=4, K=8).
Hardware strictly loaded from results/phase0/env.json:
- CPU: Intel Core i7-13620H
- GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU
"""

import sys
import os
import re
import json
import time
import subprocess
import numpy as np
from pathlib import Path
from collections import defaultdict

# Enforce single-thread CPU protocol
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

YANG_DIR = Path("baselines/yang2026").resolve()
ENC_INST = YANG_DIR / "encoder_instrumented.exe"
DEC_INST = YANG_DIR / "decoder_instrumented.exe"

CACHE_DIR = Path("results/phase1/cache_pidnet")
SWEEPS_DIR = CACHE_DIR / "sweeps"

def load_env_meta():
    with open("results/phase0/env.json") as f:
        meta = json.load(f)
    print(f"Verified Hardware: CPU = {meta['cpu']['model']} | GPU = {meta['gpu']['name']}")
    return meta

def collect_all_frame_sources(k_val):
    k_tag = f"k{k_val}"
    sources = []
    
    # 1. nuScenes keyframes
    kf_p = CACHE_DIR / f"nuscenes_keyframes_2hz_{k_tag}.npz"
    if kf_p.exists():
        with np.load(kf_p) as d:
            count = d["masks"].shape[0]
        sources.append(("nuscenes_keyframes_2hz", kf_p, count))
        
    # 2. nuScenes sweeps
    if SWEEPS_DIR.exists():
        for sp in sorted(SWEEPS_DIR.glob(f"sweeps_*_{k_tag}.npz")):
            with np.load(sp) as d:
                count = d["masks"].shape[0]
            sources.append((sp.stem, sp, count))
            
    # 3. Field clips (native & 10hz)
    for fp in sorted(CACHE_DIR.glob(f"field_*_{k_tag}.npz")):
        with np.load(fp) as d:
            count = d["masks"].shape[0]
        sources.append((fp.stem, fp, count))
        
    return sources

def build_stratified_sample(k_val, total_target=300, seed=42):
    rng = np.random.default_rng(seed)
    sources = collect_all_frame_sources(k_val)
    total_frames = sum(s[2] for s in sources)
    
    samples = []
    # Stratified allocation proportional to source size, with minimum 2 per source
    allocated = {}
    for name, path, count in sources:
        prop = count / total_frames
        alloc = max(2, int(round(prop * total_target)))
        allocated[name] = min(count, alloc)
        
    # Adjust total to exactly total_target
    current_total = sum(allocated.values())
    if current_total > total_target:
        diff = current_total - total_target
        # reduce from largest
        for name in sorted(allocated.keys(), key=lambda k: allocated[k], reverse=True):
            sub = min(diff, allocated[name] - 2)
            allocated[name] -= sub
            diff -= sub
            if diff <= 0: break
    elif current_total < total_target:
        diff = total_target - current_total
        for name in sorted(allocated.keys(), key=lambda k: allocated[k], reverse=True):
            allocated[name] += diff
            break
            
    for name, path, count in sources:
        n_sel = allocated[name]
        indices = sorted(rng.choice(count, size=n_sel, replace=False))
        for idx in indices:
            samples.append((name, path, idx))
            
    rng.shuffle(samples)
    return samples[:total_target]

def run_evaluation():
    load_env_meta()
    
    results = {}
    
    for k_val in [4, 8]:
        k_tag = f"k{k_val}"
        print(f"\n{'='*70}", flush=True)
        print(f"STARTING STRATIFIED EVALUATION FOR K={k_val} (Target: 300 frames)", flush=True)
        print(f"{'='*70}", flush=True)
        
        samples = build_stratified_sample(k_val, total_target=300, seed=42 + k_val)
        print(f"Selected {len(samples)} stratified frames across {len(set(s[0] for s in samples))} source files.", flush=True)
        
        counters_sum = [0] * 7 # g_dead_1 to g_dead_7
        enc_chrono_ms = []
        dec_chrono_ms = []
        wall_enc_ms = []
        wall_dec_ms = []
        timeouts = 0
        lossless_matches = 0
        bytes_list = []
        
        # Temp file paths in YANG_DIR
        pid = os.getpid()
        tmp_in = YANG_DIR / f"bench_{pid}_{k_val}_in.yuv"
        tmp_out = YANG_DIR / f"bench_{pid}_{k_val}_out.bin"
        tmp_rec = YANG_DIR / f"bench_{pid}_{k_val}_rec.yuv"
        
        # Sort samples by src_path to minimize disk reads and RAM usage
        samples.sort(key=lambda s: str(s[1]))
        
        current_path = None
        current_masks = None
        
        for i, (src_name, src_path, f_idx) in enumerate(samples):
            if src_path != current_path:
                current_masks = None
                import gc
                gc.collect()
                current_masks = np.load(src_path)["masks"]
                current_path = src_path
                
            frame = np.array(current_masks[f_idx], dtype=np.uint8)
            H, W = frame.shape
            
            frame.tofile(tmp_in)
            
            # Run encoder
            cmd_enc = [
                str(ENC_INST),
                "-i", str(tmp_in),
                "-o", str(tmp_out),
                "-r", str(H),
                "-c", str(W),
                "-f", "1",
                "-s", "0",
                "-t", "400"
            ]
            
            t0 = time.perf_counter()
            try:
                res_enc = subprocess.run(
                    cmd_enc,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd=str(YANG_DIR),
                    stdin=subprocess.DEVNULL
                )
                t_enc_wall = (time.perf_counter() - t0) * 1000.0
            except subprocess.TimeoutExpired:
                timeouts += 1
                print(f"[TIMEOUT] Frame {i+1} ({src_name} idx {f_idx}) encoder timed out (>60s)!")
                continue
                
            # Parse encoder dead counters and chrono time
            combined_enc = (res_enc.stderr + "\n" + res_enc.stdout)
            m_enc_dead = re.search(r"DEAD_COUNTERS_ENC:\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", combined_enc)
            if m_enc_dead:
                for c_idx in range(6):
                    counters_sum[c_idx] += int(m_enc_dead.group(c_idx + 1))
            m_enc_time = re.search(r"CHRONO_CORE_ENCODE_MS:\s*([\d\.]+)", combined_enc)
            if m_enc_time:
                enc_chrono_ms.append(float(m_enc_time.group(1)))
            wall_enc_ms.append(t_enc_wall)
            
            # Run decoder
            cmd_dec = [
                str(DEC_INST),
                "-i", str(tmp_out),
                "-o", str(tmp_rec)
            ]
            
            t0 = time.perf_counter()
            try:
                res_dec = subprocess.run(
                    cmd_dec,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd=str(YANG_DIR),
                    stdin=subprocess.DEVNULL
                )
                t_dec_wall = (time.perf_counter() - t0) * 1000.0
            except subprocess.TimeoutExpired:
                timeouts += 1
                print(f"[TIMEOUT] Frame {i+1} ({src_name} idx {f_idx}) decoder timed out (>60s)!")
                continue
                
            # Parse decoder dead counter and chrono time
            combined_dec = (res_dec.stderr + "\n" + res_dec.stdout)
            m_dec_dead = re.search(r"DEAD_COUNTERS_DEC:\s*(\d+)", combined_dec)
            if m_dec_dead:
                counters_sum[6] += int(m_dec_dead.group(1))
            m_dec_time = re.search(r"CHRONO_CORE_DECODE_MS:\s*([\d\.]+)", combined_dec)
            if m_dec_time:
                dec_chrono_ms.append(float(m_dec_time.group(1)))
            wall_dec_ms.append(t_dec_wall)
            
            # Verify lossless reconstruction
            if tmp_rec.exists():
                rec = np.fromfile(tmp_rec, dtype=np.uint8).reshape((H, W))
                if np.array_equal(frame, rec):
                    lossless_matches += 1
            if tmp_out.exists():
                bytes_list.append(tmp_out.stat().st_size)
                
            if (i + 1) % 25 == 0 or (i + 1) == len(samples):
                print(f"Progress [{k_tag}]: {i+1:3d}/{len(samples)} frames | "
                      f"Chrono Enc: {np.mean(enc_chrono_ms):.1f}ms, Dec: {np.mean(dec_chrono_ms):.1f}ms | "
                      f"Timeouts: {timeouts} | Counters: {counters_sum}", flush=True)
                      
        # Clean up temporary files
        for p in [tmp_in, tmp_out, tmp_rec]:
            if p.exists():
                try: p.unlink()
                except: pass
                
        results[k_tag] = {
            "k": k_val,
            "total_evaluated": len(samples),
            "lossless_matches": lossless_matches,
            "timeouts": timeouts,
            "counters": counters_sum,
            "chrono_enc_mean_ms": float(np.mean(enc_chrono_ms)),
            "chrono_enc_median_ms": float(np.median(enc_chrono_ms)),
            "chrono_dec_mean_ms": float(np.mean(dec_chrono_ms)),
            "chrono_dec_median_ms": float(np.median(dec_chrono_ms)),
            "wall_enc_mean_ms": float(np.mean(wall_enc_ms)),
            "wall_dec_mean_ms": float(np.mean(wall_dec_ms)),
            "mean_bytes": float(np.mean(bytes_list))
        }
        
    out_json = Path("results/phase1/p0_dead_code_and_timing_results.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved benchmark results to {out_json}")
    return results

if __name__ == "__main__":
    run_evaluation()
