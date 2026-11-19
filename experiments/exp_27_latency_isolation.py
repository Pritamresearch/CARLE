"""
Phase 1 Audit Fix: Isolated GPU Latency Timing (Median and P95)
==============================================================
Requirements:
- Measure median and p95 of gpu_ms (not mean) with torch.cuda.synchronize().
- 200 frames per model x resolution.
- Models: PIDNet-S, PIDNet-M, PIDNet-L.
- Resolutions: Half (450x800) and Full (900x1600).
- Must run with nothing else running on CPU or GPU.
- Output: results/phase1/gpu_latency_median_p95.csv
"""

import sys
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PIDNET_DIR = Path("legacy_local/CARLE CODE/PIDNet").resolve()
sys.path.insert(0, str(PIDNET_DIR))
from models.pidnet import get_pred_model

OUT_DIR = Path("results/phase1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

def benchmark_latency(num_frames=200):
    print("="*80)
    print(f"ISOLATED GPU LATENCY BENCHMARK ({num_frames} FRAMES PER CONFIGURATION)")
    print("="*80)
    
    archs = ["pidnet-s", "pidnet-m", "pidnet-l"]
    resolutions = [("Half", 450, 800), ("Full", 900, 1600)]
    
    results = []
    
    for arch in archs:
        print(f"\n---> Loading {arch.upper()}...")
        model = load_pidnet(arch)
        
        for res_name, H, W in resolutions:
            pad_h = ((H + 31) // 32) * 32
            pad_w = ((W + 31) // 32) * 32
            
            # Dummy tensor with realistic dimensions
            dummy_input = torch.randn(1, 3, pad_h, pad_w, dtype=torch.float16, device="cuda")
            
            # Warm-up (10 runs)
            for _ in range(10):
                with torch.no_grad():
                    out = model(dummy_input)
                    if isinstance(out, (list, tuple)): out = out[0]
                    out = torch.nn.functional.interpolate(out, size=(pad_h, pad_w), mode="bilinear", align_corners=True)
                    if pad_h != H or pad_w != W:
                        out = out[:, :, :H, :W]
                    torch.cuda.synchronize()
                    
            # Timed runs
            timings = []
            for _ in range(num_frames):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    out = model(dummy_input)
                    if isinstance(out, (list, tuple)): out = out[0]
                    out = torch.nn.functional.interpolate(out, size=(pad_h, pad_w), mode="bilinear", align_corners=True)
                    if pad_h != H or pad_w != W:
                        out = out[:, :, :H, :W]
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                timings.append((t1 - t0) * 1000.0)
                
            timings = np.array(timings)
            med = float(np.median(timings))
            p95 = float(np.percentile(timings, 95))
            mean = float(np.mean(timings))
            p90 = float(np.percentile(timings, 90))
            p99 = float(np.percentile(timings, 99))
            min_v = float(np.min(timings))
            max_v = float(np.max(timings))
            
            print(f"  [{arch:8s} | {res_name:4s} ({H}x{W})] Median: {med:5.2f} ms | P95: {p95:5.2f} ms | Mean: {mean:5.2f} ms | Range: [{min_v:.2f}, {max_v:.2f}] ms")
            
            results.append({
                "Architecture": arch,
                "Resolution": res_name,
                "Height": H,
                "Width": W,
                "Frames_Measured": num_frames,
                "Median_GPU_ms": round(med, 2),
                "P95_GPU_ms": round(p95, 2),
                "Mean_GPU_ms": round(mean, 2),
                "P90_GPU_ms": round(p90, 2),
                "P99_GPU_ms": round(p99, 2),
                "Min_GPU_ms": round(min_v, 2),
                "Max_GPU_ms": round(max_v, 2)
            })
            
        del model
        torch.cuda.empty_cache()
        
    df_res = pd.DataFrame(results)
    out_csv = OUT_DIR / "gpu_latency_median_p95.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"\nSaved isolated GPU latency benchmark to {out_csv}:")
    print(df_res.to_string(index=False))

if __name__ == "__main__":
    benchmark_latency(num_frames=200)
