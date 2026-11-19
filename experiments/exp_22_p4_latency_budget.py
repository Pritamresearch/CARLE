"""
Phase 4: Comprehensive Latency Budget & End-to-End Multi-Camera Profiling
Target Operating Point:
- Perception: PIDNet-S FP16 on NVIDIA RTX 3050 Laptop GPU (Driver 610.91, CUDA 13.3/12.4)
- Input Resolution: Half resolution (450x800)
- Camera Rig: 6 surround cameras (CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT)
- Batch Size: 6 (all surround cameras concurrently on GPU)
- Label Granularity: K = 4 and K = 8
- Compression: Yang et al. 2026 with S = 4 horizontal slices per frame
- Execution Modes:
  * Sequential Single-Core CPU Encoding
  * Parallel 6-Core CPU Encoding (Intel Core i7-13620H)
- Transmission Channels: Dedicated V2X at 5 Mb/s, 10 Mb/s, and 20 Mb/s
- Verification: Strict comparison against 100 ms real-time V2X budget

Saves results to:
- results/phase4/p4_latency_budget_summary.csv
- results/phase4/p4_latency_budget_detailed.json
"""

import sys
import os
import time
import json
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Environment loaded from results/phase0/env.json

YANG_DIR = Path("baselines/yang2026").resolve()
YANG_ENC = str(YANG_DIR / "encoder.exe")
YANG_DEC = str(YANG_DIR / "decoder.exe")
PIDNET_DIR = Path("legacy_local/CARLE CODE/PIDNet").resolve()
sys.path.insert(0, str(PIDNET_DIR))
from models.pidnet import get_pred_model

OUT_DIR = Path("results/phase4")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def encode_slice_worker(args):
    slice_data, r_shape, worker_id = args
    pid = os.getpid()
    t_in = YANG_DIR / f"p4_bench_{pid}_{worker_id}_in.yuv"
    t_out = YANG_DIR / f"p4_bench_{pid}_{worker_id}_out.bin"
    slice_data.tofile(t_in)
    H, W = r_shape
    cmd = [YANG_ENC, "-i", str(t_in), "-o", str(t_out), "-r", str(H), "-c", str(W), "-f", "1", "-s", "0", "-t", "400"]
    t0 = time.perf_counter()
    subprocess.run(cmd, capture_output=True, cwd=str(YANG_DIR), stdin=subprocess.DEVNULL)
    enc_time = time.perf_counter() - t0
    size = t_out.stat().st_size if t_out.exists() else 0
    for p in [t_in, t_out]:
        if p.exists():
            try: p.unlink()
            except: pass
    return size, enc_time

def decode_slice_worker(args):
    pld_bytes, r_shape, worker_id = args
    pid = os.getpid()
    t_bin = YANG_DIR / f"p4_dec_{pid}_{worker_id}_in.bin"
    t_yuv = YANG_DIR / f"p4_dec_{pid}_{worker_id}_out.yuv"
    t_bin.write_bytes(pld_bytes)
    H, W = r_shape
    cmd = [YANG_DEC, "-i", str(t_bin), "-o", str(t_yuv), "-r", str(H), "-c", str(W), "-f", "1", "-s", "0", "-t", "400"]
    t0 = time.perf_counter()
    subprocess.run(cmd, capture_output=True, cwd=str(YANG_DIR), stdin=subprocess.DEVNULL)
    dec_time = time.perf_counter() - t0
    for p in [t_bin, t_yuv]:
        if p.exists():
            try: p.unlink()
            except: pass
    return dec_time

def get_slice_splits(H: int, S: int = 4):
    base = H // S
    rem = H % S
    heights = [base + (1 if i < rem else 0) for i in range(S)]
    splits = []
    curr = 0
    for h in heights:
        splits.append((curr, curr + h))
        curr += h
    return splits

def run_p4_latency_budget():
    print("\n" + "="*80)
    print("STARTING PHASE 4: V2X MULTI-CAMERA LATENCY BUDGET BENCHMARK")
    print("="*80)
    
    # 1. Load Hardware Information
    with open("results/phase0/env.json") as f:
        env_meta = json.load(f)
    cpu_str = env_meta["cpu"]["model"]
    gpu_str = f"{env_meta['gpu']['name']} (Driver {env_meta['gpu']['driver_version']}, CUDA {env_meta['gpu']['cuda_version']})"
    print(f"Verified Hardware:\n  CPU: {cpu_str}\n  GPU: {gpu_str}")

    # 2. Benchmark Stage 1 & Stage 2: Ingestion & PIDNet-S Batch 6 Inference
    print("\n---> Benchmarking Stage 1 & 2: Batch 6 Preprocessing & Neural Perception (RTX 3050 FP16)...")
    model = get_pred_model("pidnet-s", 19)
    weight_path = PIDNET_DIR / "pretrained_models/Cityscapes/PIDNet_S_Cityscapes_val.pt"
    ckpt = torch.load(weight_path, map_location="cpu")
    sd = {k[6:]: v for k, v in ckpt.items() if k.startswith("model.")}
    model.load_state_dict(sd, strict=False)
    model.eval().cuda().half()
    
    # Prepare batch 6 of 450x800 half resolution padded to 480x800 (multiple of 32 for PIDNet)
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    pad_h, pad_w = 480, 800
    dummy_imgs = [np.random.randint(0, 256, (450, 800, 3), dtype=np.uint8) for _ in range(6)]
    
    # Stage 1: Preprocessing & Ingestion Latency
    prep_times = []
    for _ in range(15):
        t0 = time.perf_counter()
        batch_t = torch.empty((6, 3, pad_h, pad_w), dtype=torch.float16, device="cuda")
        for b in range(6):
            im = dummy_imgs[b]
            rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            norm = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
            pad = np.zeros((pad_h, pad_w, 3), dtype=np.float32)
            pad[:450, :800, :] = norm
            batch_t[b] = torch.from_numpy(pad).permute(2, 0, 1).half()
        torch.cuda.synchronize()
        prep_times.append((time.perf_counter() - t0) * 1000.0)
    stage1_prep_ms = float(np.median(prep_times[5:]))
    
    # Stage 2: Batch 6 GPU Inference Latency
    # Warmup
    for _ in range(5):
        with torch.inference_mode():
            pred = model(batch_t)
            if isinstance(pred, (list, tuple)): pred = pred[0]
            pred = torch.nn.functional.interpolate(pred, size=(pad_h, pad_w), mode="bilinear", align_corners=True)
            _ = torch.argmax(pred[:, :, :450, :800], dim=1).byte().cpu().numpy()
    torch.cuda.synchronize()
        
    infer_times = []
    for _ in range(25):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            pred = model(batch_t)
            if isinstance(pred, (list, tuple)): pred = pred[0]
            pred = torch.nn.functional.interpolate(pred, size=(pad_h, pad_w), mode="bilinear", align_corners=True)
            _ = torch.argmax(pred[:, :, :450, :800], dim=1).byte().cpu().numpy()
        torch.cuda.synchronize()
        infer_times.append((time.perf_counter() - t0) * 1000.0)
    stage2_infer_ms = float(np.median(infer_times[5:]))
    stage2_per_cam_ms = stage2_infer_ms / 6.0
    print(f"  Stage 1 (Ingestion & Preprocessing, Batch 6): {stage1_prep_ms:.2f} ms")
    print(f"  Stage 2 (Neural Perception Batch 6): {stage2_infer_ms:.2f} ms ({stage2_per_cam_ms:.2f} ms/cam, {6000.0/stage2_infer_ms:.1f} FPS)")

    # 3. Load 6 Real Surround Camera Frames for S=4 Slicing
    with open("results/phase1/cache/nuscenes_keyframes_2hz_meta.json") as f:
        meta = json.load(f)
    k4_all = np.load("results/phase1b/cache_pidnet_half/nuscenes_keyframes_2hz_k4.npz", mmap_mode="r")["masks"]
    k8_all = np.load("results/phase1b/cache_pidnet_half/nuscenes_keyframes_2hz_k8.npz", mmap_mode="r")["masks"]
    
    # Extract 6 distinct camera views from first scene
    cams = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
    scene_name = meta[0]["scene"]
    cam_indices = []
    for c in cams:
        for idx, m in enumerate(meta):
            if m["scene"] == scene_name and m["camera"] == c:
                cam_indices.append(idx)
                break
    if len(cam_indices) < 6:
        cam_indices = list(range(6))
        
    frames_k4 = [k4_all[i] for i in cam_indices]
    frames_k8 = [k8_all[i] for i in cam_indices]
    
    H, W = frames_k4[0].shape
    splits4 = get_slice_splits(H, 4)
    
    # 4. Benchmark Stage 3: Compression (Yang et al. 2026, S=4 Slices)
    # Sequential vs Parallel 6 Cores
    p4_results = []
    
    for k_val, frames in [(4, frames_k4), (8, frames_k8)]:
        print(f"\n---> Benchmarking Compression Stage 3 for K={k_val} (6 Surround Cameras x 4 Slices = 24 Slices)...")
        # Prepare slice tasks
        slice_tasks = []
        for c_idx, frame in enumerate(frames):
            for s_idx, (r0, r1) in enumerate(splits4):
                sl = np.ascontiguousarray(frame[r0:r1, :])
                slice_tasks.append((sl, sl.shape, c_idx * 4 + s_idx))
                
        # (a) Sequential Encoding (1 Core)
        print("  Running Sequential Single-Thread Encoding...")
        seq_times = []
        payload_sizes = []
        for rep in range(15):
            t0 = time.perf_counter()
            sizes = []
            for task in slice_tasks:
                sz, _ = encode_slice_worker(task)
                sizes.append(sz)
            dur = (time.perf_counter() - t0) * 1000.0
            seq_times.append(dur)
            if rep == 0: payload_sizes = sizes
        median_seq_enc_ms = float(np.median(seq_times[5:]))
        
        # (b) Parallel 6-Core Encoding
        print("  Running Parallel 6-Core Encoding (Intel i7-13620H 6 P-Cores)...")
        par_times = []
        with ProcessPoolExecutor(max_workers=6) as executor:
            for rep in range(15):
                t0 = time.perf_counter()
                _ = list(executor.map(encode_slice_worker, slice_tasks))
                dur = (time.perf_counter() - t0) * 1000.0
                par_times.append(dur)
        median_par_enc_ms = float(np.median(par_times[5:]))
        
        # (c) Receiver Decoding Latency (Sequential & Parallel)
        # Pre-read payloads for decode
        encoded_payloads = []
        for task in slice_tasks:
            sl, shape, wid = task
            t_in = YANG_DIR / f"temp_{wid}_in.yuv"
            t_out = YANG_DIR / f"temp_{wid}_out.bin"
            sl.tofile(t_in)
            cmd = [YANG_ENC, "-i", str(t_in), "-o", str(t_out), "-r", str(shape[0]), "-c", str(shape[1]), "-f", "1", "-s", "0", "-t", "400"]
            subprocess.run(cmd, capture_output=True, cwd=str(YANG_DIR), stdin=subprocess.DEVNULL)
            pld = t_out.read_bytes() if t_out.exists() else b""
            encoded_payloads.append((pld, shape, wid))
            for p in [t_in, t_out]:
                if p.exists(): p.unlink()
                
        # Sequential Decode
        dec_seq_times = []
        for _ in range(15):
            t0 = time.perf_counter()
            for task in encoded_payloads:
                _ = decode_slice_worker(task)
            dec_seq_times.append((time.perf_counter() - t0) * 1000.0)
        median_seq_dec_ms = float(np.median(dec_seq_times[5:]))
        
        # Parallel Decode (6 cores)
        dec_par_times = []
        with ProcessPoolExecutor(max_workers=6) as executor:
            for _ in range(15):
                t0 = time.perf_counter()
                _ = list(executor.map(decode_slice_worker, encoded_payloads))
                dec_par_times.append((time.perf_counter() - t0) * 1000.0)
        median_par_dec_ms = float(np.median(dec_par_times[5:]))
        
        # Total 6-camera transmission payload with 12-byte headers (24 slices x 12 bytes = 288 bytes)
        raw_payload_bytes = sum(payload_sizes)
        total_header_bytes = 24 * 12 # 288 bytes
        total_tx_bytes = raw_payload_bytes + total_header_bytes
        total_tx_bits = total_tx_bytes * 8
        
        # Stage 4: V2X Air Time at 5 Mb/s, 10 Mb/s, and 20 Mb/s
        air_time_5mbps_ms = (total_tx_bits / (5.0 * 1e6)) * 1000.0
        air_time_10mbps_ms = (total_tx_bits / (10.0 * 1e6)) * 1000.0
        air_time_20mbps_ms = (total_tx_bits / (20.0 * 1e6)) * 1000.0
        
        # End-to-End Latency Evaluations (Sequential Pipeline vs Parallel Pipeline)
        # Using 10 Mb/s standard C-V2X channel:
        seq_e2e_ms = stage1_prep_ms + stage2_infer_ms + median_seq_enc_ms + air_time_10mbps_ms + median_seq_dec_ms
        par_e2e_ms = stage1_prep_ms + stage2_infer_ms + median_par_enc_ms + air_time_10mbps_ms + median_par_dec_ms
        
        # Pipelined Throughput (bottleneck stage is max(Perception, Encoding))
        bottleneck_stage_seq_ms = max(stage2_infer_ms, median_seq_enc_ms)
        bottleneck_stage_par_ms = max(stage2_infer_ms, median_par_enc_ms)
        pipelined_fps_seq = 1000.0 / bottleneck_stage_seq_ms
        pipelined_fps_par = 1000.0 / bottleneck_stage_par_ms
        
        row_summary = {
            "K": k_val,
            "Total_6Cam_Payload_Bytes": raw_payload_bytes,
            "Headers_24Slices_Bytes": total_header_bytes,
            "Total_Transmitted_Bytes": total_tx_bytes,
            "Total_Transmitted_Bits": total_tx_bits,
            "Stage1_Ingest_Prep_ms": round(stage1_prep_ms, 2),
            "Stage2_PIDNet_Batch6_ms": round(stage2_infer_ms, 2),
            "Stage2_Per_Camera_ms": round(stage2_per_cam_ms, 2),
            "Stage3_Enc_Seq_1Core_ms": round(median_seq_enc_ms, 2),
            "Stage3_Enc_Par_6Core_ms": round(median_par_enc_ms, 2),
            "Stage4_AirTime_5Mbps_ms": round(air_time_5mbps_ms, 2),
            "Stage4_AirTime_10Mbps_ms": round(air_time_10mbps_ms, 2),
            "Stage4_AirTime_20Mbps_ms": round(air_time_20mbps_ms, 2),
            "Stage5_Dec_Seq_1Core_ms": round(median_seq_dec_ms, 2),
            "Stage5_Dec_Par_6Core_ms": round(median_par_dec_ms, 2),
            "Sequential_E2E_1Core_10Mbps_ms": round(seq_e2e_ms, 2),
            "Parallel_E2E_6Core_10Mbps_ms": round(par_e2e_ms, 2),
            "Pipelined_Bottleneck_Par_ms": round(bottleneck_stage_par_ms, 2),
            "Pipelined_Throughput_Par_FPS": round(pipelined_fps_par, 1),
            "Meets_100ms_Budget_Sequential_Par": "YES" if par_e2e_ms <= 100.0 else "NO",
            "Meets_100ms_Budget_Pipelined": "YES" if bottleneck_stage_par_ms <= 100.0 else "NO"
        }
        p4_results.append(row_summary)
        
    df_p4 = pd.DataFrame(p4_results)
    out_csv = OUT_DIR / "p4_latency_budget_summary.csv"
    out_json = OUT_DIR / "p4_latency_budget_detailed.json"
    df_p4.to_csv(out_csv, index=False)
    with open(out_json, "w") as f:
        json.dump(p4_results, f, indent=2)
        
    print("\n" + "="*80)
    print("PHASE 4 LATENCY BUDGET BENCHMARK RESULTS")
    print("="*80)
    print(df_p4.to_string(index=False))
    return df_p4

if __name__ == "__main__":
    run_p4_latency_budget()
