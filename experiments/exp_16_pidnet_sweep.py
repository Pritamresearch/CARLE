"""
Phase 1: PIDNet S/M/L Latency, Throughput, Flicker, and Granularity Study
Authors: Antigravity Agent & User
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
import cv2
import numpy as np
import torch
import subprocess
from collections import defaultdict

PIDNET_DIR = Path("legacy_local/CARLE CODE/PIDNet").resolve()
sys.path.insert(0, str(PIDNET_DIR))
from models.pidnet import get_pred_model

YANG_DIR = Path("baselines/yang2026").resolve()
YANG_ENC = str(YANG_DIR / "encoder.exe")

# Cityscapes 19 trainIds:
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

def benchmark_pidnet_gpu_times():
    print("\n" + "="*70)
    print("1. PIDNET S/M/L LATENCY & THROUGHPUT BENCHMARK (RTX 3050 6GB Laptop GPU)")
    print("="*70)
    
    results = []
    resolutions = [
        ("Full (900x1600)", 900, 1600, 928, 1600), # pad to multiple of 32
        ("Half (450x800)", 450, 800, 480, 800)
    ]
    batches = [1, 6]
    architectures = ["pidnet-s", "pidnet-m", "pidnet-l"]
    
    for arch in architectures:
        model = load_pidnet(arch)
        for res_name, H, W, pad_h, pad_w in resolutions:
            for B in batches:
                # Create dummy input tensor
                x = torch.zeros((B, 3, pad_h, pad_w), dtype=torch.float16, device="cuda")
                
                # Warm-up
                for _ in range(5):
                    with torch.no_grad():
                        _ = model(x)
                torch.cuda.synchronize()
                
                # Timing with CUDA events
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                n_reps = 20
                start_event.record()
                for _ in range(n_reps):
                    with torch.no_grad():
                        _ = model(x)
                end_event.record()
                torch.cuda.synchronize()
                
                total_ms = start_event.elapsed_time(end_event)
                batch_ms = total_ms / n_reps
                frame_ms = batch_ms / B
                fps = 1000.0 / frame_ms
                
                print(f"[{arch.upper()}] {res_name} | Batch={B} : Batch Latency={batch_ms:.2f} ms | Per-Frame={frame_ms:.2f} ms | FPS={fps:.1f}")
                results.append({
                    "arch": arch,
                    "resolution": res_name,
                    "H": H, "W": W,
                    "batch": B,
                    "batch_latency_ms": round(batch_ms, 2),
                    "per_frame_ms": round(frame_ms, 2),
                    "throughput_fps": round(fps, 1),
                    "supports_10hz": (frame_ms <= 100.0 if B == 1 else batch_ms <= 100.0) # all 6 cams within 100ms
                })
        del model
        torch.cuda.empty_cache()
        
    out_p = Path("results/phase1/pidnet_latency_benchmark.json")
    with open(out_p, "w") as f:
        json.dump(results, f, indent=2)
    return results

def compute_flicker(frames):
    # frames: (T, H, W)
    T = len(frames)
    if T < 3: return 0.0
    flicker_count = 0
    change_count = 0
    for t in range(1, T - 1):
        prev_f = frames[t - 1]
        curr_f = frames[t]
        next_f = frames[t + 1]
        
        changed = (next_f != curr_f)
        flickered = (next_f == prev_f) & (curr_f != prev_f)
        
        change_count += np.sum(changed)
        flicker_count += np.sum(flickered)
        
    if change_count == 0: return 0.0
    return float(flicker_count / change_count) * 100.0

def run_yang_single(mask):
    H, W = mask.shape
    pid = os.getpid()
    tmp_in = YANG_DIR / f"sweep_{pid}_in.yuv"
    tmp_out = YANG_DIR / f"sweep_{pid}_out.bin"
    
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
        subprocess.run(cmd, capture_output=True, cwd=str(YANG_DIR), stdin=subprocess.DEVNULL, timeout=30)
        bits = tmp_out.stat().st_size * 8 if tmp_out.exists() else 0
    except:
        bits = 0
    finally:
        for p in [tmp_in, tmp_out]:
            if p.exists():
                try: p.unlink()
                except: pass
    return bits

def evaluate_models_granularity_flicker():
    print("\n" + "="*70)
    print("2. PIDNET S/M/L, GRANULARITY (K=4,8,19), RESOLUTION, AND FLICKER EVALUATION")
    print("="*70)
    
    # Load 100 contiguous frames from nuScenes scene-0061 CAM_FRONT
    meta_p = Path("results/phase1/cache/sweeps/sweeps_scene-0061_meta.json")
    with open(meta_p, "r") as f:
        meta = json.load(f)
    
    cam_meta = [m for m in meta if m["camera"] == "CAM_FRONT"]
    cam_meta.sort(key=lambda x: x["timestamp"])
    selected_meta = cam_meta[:100]
    
    dataroot = Path("data/nuscenes")
    imgs = [cv2.imread(str(dataroot / m["filename"])) for m in selected_meta]
    print(f"Loaded {len(imgs)} contiguous CAM_FRONT frames from scene-0061.")
    
    architectures = ["pidnet-s", "pidnet-m", "pidnet-l"]
    k_values = [4, 8, 19]
    resolutions = [
        ("Full (900x1600)", 900, 1600),
        ("Half (450x800)", 450, 800)
    ]
    
    summary = []
    
    for arch in architectures:
        print(f"\nProcessing Architecture: {arch.upper()}")
        model = load_pidnet(arch)
        
        for res_name, H, W in resolutions:
            pad_h = ((H + 31) // 32) * 32
            pad_w = ((W + 31) // 32) * 32
            
            # Predict for all 100 frames
            raw_preds = []
            for img in imgs:
                img_res = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR) if (H, W) != (900, 1600) else img
                img_rgb = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                norm = (img_rgb - MEAN) / STD
                padded = np.zeros((pad_h, pad_w, 3), dtype=np.float32)
                padded[:H, :W, :] = norm
                tensor = torch.from_numpy(padded).permute(2, 0, 1).unsqueeze(0).cuda().half()
                
                with torch.no_grad():
                    out = model(tensor)
                    if isinstance(out, (list, tuple)): out = out[0]
                    pred = torch.argmax(out[:, :, :H, :W], dim=1)[0].cpu().numpy().astype(np.uint8)
                raw_preds.append(pred)
                
            raw_preds = np.array(raw_preds, dtype=np.uint8)
            
            for k in k_values:
                if k == 4:
                    maps = CITYSCAPES_TO_K4[raw_preds]
                elif k == 8:
                    maps = CITYSCAPES_TO_K8[raw_preds]
                else: # K = 19
                    maps = raw_preds + 1 # 1..19 for valid palette
                    
                # Compute Yang bits for all 100 frames
                bits_list = [run_yang_single(m) for m in maps]
                mean_bits = float(np.mean(bits_list))
                flicker_pct = compute_flicker(maps)
                
                print(f"[{arch.upper()}] {res_name} | K={k:2d} -> Yang Bits/Frame = {mean_bits:.1f} bpf | 1-Frame Flicker = {flicker_pct:.2f}%")
                summary.append({
                    "arch": arch,
                    "resolution": res_name,
                    "H": H, "W": W,
                    "K": k,
                    "mean_bits_yang": round(mean_bits, 1),
                    "flicker_pct": round(flicker_pct, 2)
                })
                
        del model
        torch.cuda.empty_cache()
        
    out_p = Path("results/phase1/pidnet_granularity_flicker_study.json")
    with open(out_p, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved granularity and flicker results to {out_p}")
    return summary

if __name__ == "__main__":
    benchmark_pidnet_gpu_times()
    evaluate_models_granularity_flicker()
