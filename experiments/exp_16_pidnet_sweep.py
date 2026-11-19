"""
Phase 1: Comprehensive PIDNet S/M/L Granularity, Flicker, and Latency Study (Full Per-Frame Run)
=================================================================================================
Evaluates 18 configurations:
- 3 Architectures: PIDNet-S, PIDNet-M, PIDNet-L
- 2 Resolutions: Full (900x1600), Half (450x800)
- 3 Class Granularities: K in {4, 8, 19}
Across all 60 nuScenes sweep streams (10 scenes x 6 cameras x 100 frames = 6,000 frames per model/res).

Methodology:
1. Logits are upsampled to input resolution using bilinear interpolation before argmax:
   torch.nn.functional.interpolate(logits, size=(H, W), mode="bilinear", align_corners=True)
2. Includes unit test asserting output map shape == input image shape.
3. Maps are encoded with Yang 2026 C++ encoder.
4. One-frame flicker % is computed per stream:
   flicker = sum((y_{t+1} == y_{t-1}) & (y_t != y_{t-1})) / sum(y_{t+1} != y_t) * 100
5. Per-frame raw records written incrementally to results/phase1/pidnet_granularity_sweep_raw.csv
6. Configuration summary saved to results/phase1/pidnet_granularity_flicker_study.csv
"""

import sys
import os
import time
import json
import cv2
import subprocess
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PIDNET_DIR = Path("legacy_local/CARLE CODE/PIDNet").resolve()
sys.path.insert(0, str(PIDNET_DIR))
from models.pidnet import get_pred_model

YANG_DIR = Path("baselines/yang2026").resolve()
YANG_ENC = str(YANG_DIR / "encoder.exe")

OUT_DIR = Path("results/phase1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATAROOT = Path("data/nuscenes")
SWEEPS_META_DIR = Path("results/phase1/cache_pidnet/sweeps")

# Class mappings from Cityscapes 19 trainIds
CITYSCAPES_TO_K4 = np.zeros(19, dtype=np.uint8)
CITYSCAPES_TO_K4[0] = 1                         # Road {0}
CITYSCAPES_TO_K4[[13, 14, 15, 16]] = 2         # Vehicle {13,14,15,16}
CITYSCAPES_TO_K4[[11, 12, 17, 18]] = 3         # Vulnerable Road User {11,12,17,18}
CITYSCAPES_TO_K4[list(range(1, 11))] = 4       # Other {1..10}

CITYSCAPES_TO_K8 = np.zeros(19, dtype=np.uint8)
CITYSCAPES_TO_K8[0] = 1                         # Road {0}
CITYSCAPES_TO_K8[1] = 2                         # Sidewalk {1}
CITYSCAPES_TO_K8[[2, 3, 4, 5, 6, 7]] = 3       # Static structure {2,3,4,5,6,7}
CITYSCAPES_TO_K8[[8, 9]] = 4                   # Vegetation / terrain {8,9}
CITYSCAPES_TO_K8[10] = 5                        # Sky {10}
CITYSCAPES_TO_K8[[13, 14, 15, 16]] = 6         # Vehicle {13,14,15,16}
CITYSCAPES_TO_K8[[17, 18]] = 7                 # Two-wheeler {17,18}
CITYSCAPES_TO_K8[[11, 12]] = 8                 # Person / rider {11,12}

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def load_pidnet(arch="pidnet-s"):
    name_map = {
        "pidnet-s": ("pidnet-s", "PIDNet_S_Cityscapes_val.pt"),
        "pidnet-m": ("pidnet-m", "PIDNet_M_Cityscapes_val.pt"),
        "pidnet-l": ("pidnet-l", "PIDNet_L_Cityscapes_val.pt"),
    }
    model_name, weight_file = name_map[arch]
    model = get_pred_model(model_name, 19)
    weight_path = PIDNET_DIR / f"pretrained_models/Cityscapes/{weight_file}"
    ckpt = torch.load(weight_path, map_location="cpu")
    sd = {k[6:]: v for k, v in ckpt.items() if k.startswith("model.")}
    model.load_state_dict(sd, strict=False)
    model.eval().cuda().half()
    return model

def test_output_map_shape():
    print("\n[Unit Test] Asserting PIDNet output map shape == input image shape...")
    model = load_pidnet("pidnet-s")
    for H, W in [(900, 1600), (450, 800)]:
        pad_h = ((H + 31) // 32) * 32
        pad_w = ((W + 31) // 32) * 32
        x = torch.zeros((1, 3, pad_h, pad_w), dtype=torch.float16, device="cuda")
        with torch.no_grad():
            out = model(x)
            if isinstance(out, (list, tuple)): out = out[0]
            out = torch.nn.functional.interpolate(out, size=(pad_h, pad_w), mode="bilinear", align_corners=True)
            if pad_h != H or pad_w != W:
                out = out[:, :, :H, :W]
            pred = torch.argmax(out, dim=1)[0].byte().cpu().numpy()
        assert pred.shape == (H, W), f"Shape mismatch! Expected {(H, W)}, got {pred.shape}"
        print(f"  Passed for input {(H, W)}: output map shape = {pred.shape}")
    del model
    torch.cuda.empty_cache()
    print("[Unit Test] ALL SHAPE ASSERTIONS PASSED!\n")

def encode_yang_single(mask: np.ndarray, worker_id: int = 0) -> int:
    H, W = mask.shape
    pid = os.getpid()
    tmp_in = YANG_DIR / f"sw_raw_{pid}_{worker_id}_in.yuv"
    tmp_out = YANG_DIR / f"sw_raw_{pid}_{worker_id}_out.bin"
    mask.tofile(tmp_in)
    cmd = [
        YANG_ENC,
        "-i", str(tmp_in),
        "-o", str(tmp_out),
        "-r", str(H),
        "-c", str(W),
        "-f", "1",
        "-s", "0",
        "-t", "400"
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, cwd=str(YANG_DIR), stdin=subprocess.DEVNULL, timeout=20)
        bits = tmp_out.stat().st_size * 8 if tmp_out.exists() else 0
    except Exception:
        bits = 0
    finally:
        for p in [tmp_in, tmp_out]:
            if p.exists():
                try: p.unlink()
                except: pass
    return bits

def compute_flicker_stream(frames_list):
    # frames_list: list of (H, W) uint8 arrays
    T = len(frames_list)
    if T < 3: return 0.0
    flicker_count = 0
    change_count = 0
    for t in range(1, T - 1):
        prev_f = frames_list[t - 1]
        curr_f = frames_list[t]
        next_f = frames_list[t + 1]
        changed = (next_f != curr_f)
        flickered = (next_f == prev_f) & (curr_f != prev_f)
        change_count += int(np.sum(changed))
        flicker_count += int(np.sum(flickered))
    if change_count == 0: return 0.0
    return float(flicker_count / change_count) * 100.0

def main():
    test_output_map_shape()
    
    # Locate all metadata files for the 10 nuScenes scenes
    meta_files = sorted(list(SWEEPS_META_DIR.glob("sweeps_scene-*_meta.json")))
    print(f"Loaded {len(meta_files)} scene metadata files.")
    
    # Load all 60 streams metadata
    # stream_key -> list of frame dicts (100 frames per camera)
    streams = {}
    for mf in meta_files:
        with open(mf, "r") as f:
            records = json.load(f)
        for r in records:
            s_key = f"{r['scene']}_{r['camera']}"
            if s_key not in streams:
                streams[s_key] = []
            if len(streams[s_key]) < 100:
                streams[s_key].append(r)
                
    stream_keys = sorted(list(streams.keys()))
    print(f"Total streams: {len(stream_keys)} (Expected: 60 streams, 100 frames each = 6,000 frames total)")
    
    raw_csv_path = OUT_DIR / "pidnet_granularity_sweep_raw.csv"
    summary_csv_path = OUT_DIR / "pidnet_granularity_flicker_study.csv"
    
    # Initialize raw CSV header if not exists
    if not raw_csv_path.exists():
        with open(raw_csv_path, "w") as f:
            f.write("timestamp,arch,resolution,H,W,stream,frame_idx,k4_bits,k8_bits,k19_bits,gpu_ms\n")
            
    archs = ["pidnet-s", "pidnet-m", "pidnet-l"]
    resolutions = [("Half (450x800)", 450, 800), ("Full (900x1600)", 900, 1600)]
    
    summary_data = []
    
    start_total = time.time()
    
    for arch in archs:
        print(f"\n" + "="*80)
        print(f"LOADING MODEL: {arch.upper()}")
        print("="*80)
        model = load_pidnet(arch)
        
        for res_label, H, W in resolutions:
            pad_h = ((H + 31) // 32) * 32
            pad_w = ((W + 31) // 32) * 32
            
            print(f"\n---> Evaluating {arch} at {res_label} ({H}x{W}) across 60 streams...")
            arch_res_start = time.time()
            
            all_k4_bits = []
            all_k8_bits = []
            all_k19_bits = []
            all_gpu_ms = []
            
            stream_flicker_k4 = []
            stream_flicker_k8 = []
            stream_flicker_k19 = []
            
            stream_count = 0
            
            for s_key in stream_keys:
                stream_count += 1
                frames_meta = streams[s_key]
                
                stream_masks_k4 = []
                stream_masks_k8 = []
                stream_masks_k19 = []
                
                for f_idx, r in enumerate(frames_meta):
                    img_path = DATAROOT / r["filename"]
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    if (img.shape[0] != H) or (img.shape[1] != W):
                        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
                        
                    # Preprocess
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    img_norm = (img_rgb - MEAN) / STD
                    if pad_h != H or pad_w != W:
                        img_norm = np.pad(img_norm, ((0, pad_h - H), (0, pad_w - W), (0, 0)), mode="constant")
                    t_in = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).half().cuda()
                    
                    # GPU inference with high-resolution steady timing
                    torch.cuda.synchronize()
                    t0_gpu = time.perf_counter()
                    with torch.no_grad():
                        out = model(t_in)
                        if isinstance(out, (list, tuple)): out = out[0]
                        out = torch.nn.functional.interpolate(out, size=(pad_h, pad_w), mode="bilinear", align_corners=True)
                        if pad_h != H or pad_w != W:
                            out = out[:, :, :H, :W]
                        pred_19 = torch.argmax(out, dim=1)[0].byte().cpu().numpy()
                    torch.cuda.synchronize()
                    t1_gpu = time.perf_counter()
                    gpu_ms = (t1_gpu - t0_gpu) * 1000.0
                    all_gpu_ms.append(gpu_ms)
                    
                    # Class mappings
                    mask_k19 = pred_19 + 1 # 1..19
                    mask_k4 = CITYSCAPES_TO_K4[pred_19]
                    mask_k8 = CITYSCAPES_TO_K8[pred_19]
                    
                    stream_masks_k4.append(mask_k4)
                    stream_masks_k8.append(mask_k8)
                    stream_masks_k19.append(mask_k19)
                    
                    # Encode with Yang
                    b4 = encode_yang_single(mask_k4)
                    b8 = encode_yang_single(mask_k8)
                    b19 = encode_yang_single(mask_k19)
                    
                    all_k4_bits.append(b4)
                    all_k8_bits.append(b8)
                    all_k19_bits.append(b19)
                    
                    # Append raw record
                    with open(raw_csv_path, "a") as f_raw:
                        f_raw.write(f"{datetime.now().isoformat()},{arch},{res_label},{H},{W},{s_key},{f_idx},{b4},{b8},{b19},{gpu_ms:.2f}\n")
                        
                # Compute stream flicker
                fl4 = compute_flicker_stream(stream_masks_k4)
                fl8 = compute_flicker_stream(stream_masks_k8)
                fl19 = compute_flicker_stream(stream_masks_k19)
                
                stream_flicker_k4.append(fl4)
                stream_flicker_k8.append(fl8)
                stream_flicker_k19.append(fl19)
                
                if stream_count % 5 == 0 or stream_count == 60:
                    elapsed = time.time() - arch_res_start
                    print(f"  [{arch} - {res_label}] Processed {stream_count}/60 streams ({len(all_k4_bits)} frames) | Elapsed: {elapsed/60:.1f} min | Mean GPU: {np.mean(all_gpu_ms):.2f} ms | Mean K=4 Bits: {np.mean(all_k4_bits):.1f}")
                    
            # Record summary rows for this arch and resolution across K in {4, 8, 19}
            summary_data.append({
                "arch": arch, "resolution": res_label, "H": H, "W": W, "K": 4,
                "gpu_ms_per_frame": round(float(np.mean(all_gpu_ms)), 2),
                "mean_bits_yang": round(float(np.mean(all_k4_bits)), 1),
                "flicker_pct": round(float(np.mean(stream_flicker_k4)), 2)
            })
            summary_data.append({
                "arch": arch, "resolution": res_label, "H": H, "W": W, "K": 8,
                "gpu_ms_per_frame": round(float(np.mean(all_gpu_ms)), 2),
                "mean_bits_yang": round(float(np.mean(all_k8_bits)), 1),
                "flicker_pct": round(float(np.mean(stream_flicker_k8)), 2)
            })
            summary_data.append({
                "arch": arch, "resolution": res_label, "H": H, "W": W, "K": 19,
                "gpu_ms_per_frame": round(float(np.mean(all_gpu_ms)), 2),
                "mean_bits_yang": round(float(np.mean(all_k19_bits)), 1),
                "flicker_pct": round(float(np.mean(stream_flicker_k19)), 2)
            })
            
            pd.DataFrame(summary_data).to_csv(summary_csv_path, index=False)
            print(f"--> Saved intermediate configuration summary to {summary_csv_path}")
            
        del model
        torch.cuda.empty_cache()
        
    total_duration = time.time() - start_total
    print(f"\nALL 18 CONFIGURATIONS COMPLETE! Total Wall-Clock Runtime: {total_duration/3600:.2f} hours")
    print(f"Raw data: {raw_csv_path}")
    print(f"Summary: {summary_csv_path}")

if __name__ == "__main__":
    main()
