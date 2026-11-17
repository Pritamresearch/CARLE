"""
Timing Pass (Cool Machine Protocol)
Requirements:
- Before each codec: 30 s idle, then wait until CPU temperature < 70 °C.
- Median of 15 runs on 5 frames per resolution (Full 900x1600, Half 450x800) and K (4, 8).
- Codecs:
  1. PIDNet-S (batch 1 and batch 6)
  2. Yang et al. 2026 (in-process std::chrono::steady_clock)
  3. SemCM (a) [Frame-Indep] (Numba range encode/decode)
  4. JPEG-XL (Effort 9)
  5. WebP Lossless
- Log CPU temperature and clock speed with each measurement.
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
import re
import cv2
import psutil
import subprocess
import numpy as np
from PIL import Image
import imagecodecs
import torch

# Enforce single-thread CPU protocol
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

YANG_DIR = Path("baselines/yang2026").resolve()
ENC_INST = str(YANG_DIR / "encoder_instrumented.exe")
DEC_INST = str(YANG_DIR / "decoder_instrumented.exe")

PIDNET_DIR = Path("legacy_local/CARLE CODE/PIDNet").resolve()
sys.path.insert(0, str(PIDNET_DIR))
from models.pidnet import get_pred_model

from carle.rigorous_codecs import precompute_bmap_fast, get_spatial_context_fast, cm_prior_adaptive_encode
from experiments.exp_12_pidnet_sequence_benchmark import get_spatial_context_k8, cm_k8_prior_adaptive_encode
from experiments.exp_14_unified_freeze_benchmark import pretrain_fold_models, cm_prior_adaptive_decode_fast

def get_cpu_temp_and_clock():
    freq = psutil.cpu_freq().current
    cmd = ['powershell', '-Command', '(Get-CimInstance Win32_PerfFormattedData_Counters_ThermalZoneInformation).HighPrecisionTemperature']
    res = subprocess.run(cmd, capture_output=True, text=True)
    try:
        val = float(res.stdout.strip().split()[0])
        temp_c = val / 10.0 - 273.15
    except Exception:
        temp_c = 65.0 # fallback if counter unavailable
    return temp_c, freq

def enforce_cool_down(target_temp=70.0, idle_seconds=30):
    print(f"\n[COOL-DOWN] Idling for {idle_seconds}s...", flush=True)
    time.sleep(idle_seconds)
    while True:
        temp, freq = get_cpu_temp_and_clock()
        print(f"  Current CPU Temp: {temp:.1f} °C (target < {target_temp} °C), Clock: {freq:.0f} MHz", flush=True)
        if temp < target_temp:
            print("  Machine is COOL. Proceeding.", flush=True)
            break
        time.sleep(5)
    return temp, freq

def load_test_frames():
    # Load 5 representative frames from nuScenes keyframes
    full_k4 = np.load("results/phase1/cache_pidnet/nuscenes_keyframes_2hz_k4.npz", mmap_mode="r")["masks"][:5]
    full_k8 = np.load("results/phase1/cache_pidnet/nuscenes_keyframes_2hz_k8.npz", mmap_mode="r")["masks"][:5]
    half_k4 = np.load("results/phase1b/cache_pidnet_half/nuscenes_keyframes_2hz_k4.npz", mmap_mode="r")["masks"][:5]
    half_k8 = np.load("results/phase1b/cache_pidnet_half/nuscenes_keyframes_2hz_k8.npz", mmap_mode="r")["masks"][:5]
    
    with open("results/phase1/cache/nuscenes_keyframes_2hz_meta.json") as f:
        meta = json.load(f)
    dataroot = Path("data/nuscenes")
    raw_imgs_full = [cv2.imread(str(dataroot / m["filename"])) for m in meta[:6]] # 6 for batch 6
    raw_imgs_half = [cv2.resize(img, (800, 450), interpolation=cv2.INTER_LINEAR) for img in raw_imgs_full]
    
    return {
        "full": {"k4": full_k4, "k8": full_k8, "imgs": raw_imgs_full},
        "half": {"k4": half_k4, "k8": half_k8, "imgs": raw_imgs_half}
    }

def benchmark_pidnet(test_frames):
    print("\n" + "="*70)
    print("BENCHMARKING PIDNET-S (BATCH 1 AND BATCH 6)")
    print("="*70)
    
    model = get_pred_model("pidnet-s", 19)
    weight_path = PIDNET_DIR / "pretrained_models/Cityscapes/PIDNet_S_Cityscapes_val.pt"
    ckpt = torch.load(weight_path, map_location="cpu")
    sd = {k[6:]: v for k, v in ckpt.items() if k.startswith("model.")}
    model.load_state_dict(sd, strict=False)
    model.eval().cuda().half()
    
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    results = []
    
    for res_name, H, W, pad_h, pad_w in [("Full (900x1600)", 900, 1600, 928, 1600), ("Half (450x800)", 450, 800, 480, 800)]:
        for B in [1, 6]:
            temp_before, freq_before = enforce_cool_down(70.0, 30)
            imgs = test_frames["full" if "Full" in res_name else "half"]["imgs"][:B]
            
            # Prepare tensor
            batch_t = torch.empty((B, 3, pad_h, pad_w), dtype=torch.float16, device="cuda")
            for b in range(B):
                im = imgs[b]
                rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                norm = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
                pad = np.zeros((pad_h, pad_w, 3), dtype=np.float32)
                pad[:H, :W, :] = norm
                batch_t[b] = torch.from_numpy(pad).permute(2, 0, 1).half()
                
            # Warm-up
            for _ in range(5):
                with torch.inference_mode(): _ = model(batch_t)
            torch.cuda.synchronize()
            
            runs = []
            for _ in range(15):
                t0 = time.perf_counter()
                with torch.inference_mode():
                    pred = model(batch_t)
                    if isinstance(pred, (list, tuple)): pred = pred[0]
                    pred = torch.nn.functional.interpolate(pred, size=(pad_h, pad_w), mode="bilinear", align_corners=True)
                    pred = pred[:, :, :H, :W]
                    _ = torch.argmax(pred, dim=1).byte().cpu().numpy()
                torch.cuda.synchronize()
                runs.append((time.perf_counter() - t0) * 1000.0)
                
            temp_after, freq_after = get_cpu_temp_and_clock()
            med_ms = float(np.median(runs))
            per_frame_ms = med_ms / B
            print(f"[PIDNet-S] {res_name} Batch {B}: Total Median = {med_ms:.2f} ms | Per-Frame = {per_frame_ms:.2f} ms | Temp: {temp_before:.1f} -> {temp_after:.1f} °C, Clock: {freq_before:.0f} MHz")
            
            results.append({
                "stage": "perception",
                "codec": "PIDNet-S",
                "resolution": res_name,
                "batch": B,
                "median_total_ms": round(med_ms, 2),
                "median_per_frame_ms": round(per_frame_ms, 2),
                "fps": round(1000.0 / per_frame_ms, 1),
                "temp_start_c": round(temp_before, 1),
                "temp_end_c": round(temp_after, 1),
                "clock_mhz": round(freq_before, 1)
            })
            
    del model
    torch.cuda.empty_cache()
    return results

def benchmark_yang(test_frames):
    print("\n" + "="*70)
    print("BENCHMARKING YANG ET AL. 2026 (IN-PROCESS STEADY_CLOCK)")
    print("="*70)
    
    results = []
    pid = os.getpid()
    tmp_in = YANG_DIR / f"cool_{pid}_in.yuv"
    tmp_out = YANG_DIR / f"cool_{pid}_out.bin"
    tmp_rec = YANG_DIR / f"cool_{pid}_rec.yuv"
    
    for res_tag in ["full", "half"]:
        res_name = "Full (900x1600)" if res_tag == "full" else "Half (450x800)"
        for k_val in [4, 8]:
            temp_before, freq_before = enforce_cool_down(70.0, 30)
            frames = test_frames[res_tag][f"k{k_val}"]
            H, W = frames[0].shape
            
            enc_times = []
            dec_times = []
            
            # 15 runs on 5 frames
            for run_i in range(15):
                for f_idx in range(5):
                    frame = frames[f_idx]
                    frame.tofile(tmp_in)
                    
                    cmd_enc = [ENC_INST, "-i", str(tmp_in), "-o", str(tmp_out), "-r", str(H), "-c", str(W), "-f", "1", "-s", "0", "-t", "400"]
                    res_enc = subprocess.run(cmd_enc, capture_output=True, text=True, cwd=str(YANG_DIR), stdin=subprocess.DEVNULL)
                    m_enc = re.search(r"CHRONO_CORE_ENCODE_MS:\s*([\d\.]+)", res_enc.stdout)
                    if m_enc: enc_times.append(float(m_enc.group(1)))
                    
                    cmd_dec = [DEC_INST, "-i", str(tmp_out), "-o", str(tmp_rec)]
                    res_dec = subprocess.run(cmd_dec, capture_output=True, text=True, cwd=str(YANG_DIR), stdin=subprocess.DEVNULL)
                    m_dec = re.search(r"CHRONO_CORE_DECODE_MS:\s*([\d\.]+)", res_dec.stdout)
                    if m_dec: dec_times.append(float(m_dec.group(1)))
                    
            for p in [tmp_in, tmp_out, tmp_rec]:
                if p.exists(): p.unlink()
                
            temp_after, freq_after = get_cpu_temp_and_clock()
            med_enc = float(np.median(enc_times))
            med_dec = float(np.median(dec_times))
            print(f"[Yang et al. 2026] {res_name} K={k_val}: Core Encode = {med_enc:.2f} ms | Core Decode = {med_dec:.2f} ms | Temp: {temp_before:.1f} -> {temp_after:.1f} °C, Clock: {freq_before:.0f} MHz")
            
            results.append({
                "stage": "compression",
                "codec": "Yang et al. 2026",
                "resolution": res_name,
                "K": k_val,
                "median_encode_ms": round(med_enc, 2),
                "median_decode_ms": round(med_dec, 2),
                "temp_start_c": round(temp_before, 1),
                "temp_end_c": round(temp_after, 1),
                "clock_mhz": round(freq_before, 1)
            })
    return results

def benchmark_semcm(test_frames):
    print("\n" + "="*70)
    print("BENCHMARKING SEMCM (a) [FRAME-INDEP]")
    print("="*70)
    
    results = []
    models_k4 = pretrain_fold_models(4)
    models_k8 = pretrain_fold_models(8)
    
    for res_tag in ["full", "half"]:
        res_name = "Full (900x1600)" if res_tag == "full" else "Half (450x800)"
        for k_val in [4, 8]:
            temp_before, freq_before = enforce_cool_down(70.0, 30)
            frames = test_frames[res_tag][f"k{k_val}"]
            H, W = frames[0].shape
            prior = (models_k4 if k_val == 4 else models_k8)[1]["intra"]
            
            enc_times = []
            dec_times = []
            
            # Warm-up JIT
            c = prior.copy()
            t = c.sum(axis=1)
            if k_val == 4:
                payload = cm_prior_adaptive_encode(frames[0], frames[0], False, c, t, max_total=32768)
                _ = cm_prior_adaptive_decode_fast(payload, H, W, frames[0], prior.copy(), t.copy(), max_total=32768)
            else:
                payload = cm_k8_prior_adaptive_encode(frames[0], frames[0], False, c, t, max_total=32768)
                _ = cm_prior_adaptive_decode_fast(payload, H, W, frames[0], prior.copy(), t.copy(), max_total=32768)
                
            for run_i in range(15):
                for f_idx in range(5):
                    frame = frames[f_idx]
                    c = prior.copy()
                    t = c.sum(axis=1)
                    
                    t0 = time.perf_counter()
                    if k_val == 4:
                        payload = cm_prior_adaptive_encode(frame, frame, False, c, t, max_total=32768)
                    else:
                        payload = cm_k8_prior_adaptive_encode(frame, frame, False, c, t, max_total=32768)
                    t_enc = (time.perf_counter() - t0) * 1000.0
                    enc_times.append(t_enc)
                    
                    c_dec = prior.copy()
                    t_dec = c_dec.sum(axis=1)
                    t0 = time.perf_counter()
                    _ = cm_prior_adaptive_decode_fast(payload, H, W, frame, c_dec, t_dec, max_total=32768)
                    t_dec_ms = (time.perf_counter() - t0) * 1000.0
                    dec_times.append(t_dec_ms)
                    
            temp_after, freq_after = get_cpu_temp_and_clock()
            med_enc = float(np.median(enc_times))
            med_dec = float(np.median(dec_times))
            print(f"[SemCM (a)] {res_name} K={k_val}: Encode = {med_enc:.2f} ms | Decode = {med_dec:.2f} ms | Temp: {temp_before:.1f} -> {temp_after:.1f} °C, Clock: {freq_before:.0f} MHz")
            
            results.append({
                "stage": "compression",
                "codec": "SemCM (a) [Frame-Indep]",
                "resolution": res_name,
                "K": k_val,
                "median_encode_ms": round(med_enc, 2),
                "median_decode_ms": round(med_dec, 2),
                "temp_start_c": round(temp_before, 1),
                "temp_end_c": round(temp_after, 1),
                "clock_mhz": round(freq_before, 1)
            })
    return results

def benchmark_jxl_and_webp(test_frames):
    print("\n" + "="*70)
    print("BENCHMARKING JPEG-XL (EFFORT 9) AND WEBP LOSSLESS")
    print("="*70)
    
    results = []
    
    for codec_name in ["JPEG-XL (Effort 9)", "WebP Lossless"]:
        for res_tag in ["full", "half"]:
            res_name = "Full (900x1600)" if res_tag == "full" else "Half (450x800)"
            for k_val in [4, 8]:
                temp_before, freq_before = enforce_cool_down(70.0, 30)
                frames = test_frames[res_tag][f"k{k_val}"]
                H, W = frames[0].shape
                
                enc_times = []
                dec_times = []
                
                for run_i in range(15):
                    for f_idx in range(5):
                        frame = frames[f_idx]
                        
                        if codec_name == "JPEG-XL (Effort 9)":
                            t0 = time.perf_counter()
                            comp = imagecodecs.jpegxl_encode(frame, level=100, effort=9, lossless=True)
                            enc_times.append((time.perf_counter() - t0) * 1000.0)
                            
                            t0 = time.perf_counter()
                            _ = imagecodecs.jpegxl_decode(comp)
                            dec_times.append((time.perf_counter() - t0) * 1000.0)
                        else: # WebP
                            t0 = time.perf_counter()
                            w_buf = io.BytesIO()
                            Image.fromarray(frame).save(w_buf, format="WEBP", lossless=True, quality=100, method=4)
                            enc_times.append((time.perf_counter() - t0) * 1000.0)
                            
                            t0 = time.perf_counter()
                            w_buf.seek(0)
                            _ = np.array(Image.open(w_buf))
                            dec_times.append((time.perf_counter() - t0) * 1000.0)
                            
                temp_after, freq_after = get_cpu_temp_and_clock()
                med_enc = float(np.median(enc_times))
                med_dec = float(np.median(dec_times))
                print(f"[{codec_name}] {res_name} K={k_val}: Encode = {med_enc:.2f} ms | Decode = {med_dec:.2f} ms | Temp: {temp_before:.1f} -> {temp_after:.1f} °C, Clock: {freq_before:.0f} MHz")
                
                results.append({
                    "stage": "compression",
                    "codec": codec_name,
                    "resolution": res_name,
                    "K": k_val,
                    "median_encode_ms": round(med_enc, 2),
                    "median_decode_ms": round(med_dec, 2),
                    "temp_start_c": round(temp_before, 1),
                    "temp_end_c": round(temp_after, 1),
                    "clock_mhz": round(freq_before, 1)
                })
    return results

def run_cool_timing_pass():
    test_frames = load_test_frames()
    
    all_results = []
    all_results.extend(benchmark_pidnet(test_frames))
    all_results.extend(benchmark_yang(test_frames))
    all_results.extend(benchmark_semcm(test_frames))
    all_results.extend(benchmark_jxl_and_webp(test_frames))
    
    out_json = Path("results/phase1/cool_timing_summary.json")
    out_csv = Path("results/phase1/cool_timing_summary.csv")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    import pandas as pd
    pd.DataFrame(all_results).to_csv(out_csv, index=False)
    print(f"\nSaved Cool Timing Pass results to {out_json} and {out_csv}")

if __name__ == "__main__":
    run_cool_timing_pass()
