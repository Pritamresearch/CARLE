"""
Night Speckle Analysis for PIDNet-L Half Resolution (450x800, K=4)
Scenes: scene-1094 and scene-1100
Outputs:
- results/phase1/night_speckle_analysis.csv
"""

import sys
import os
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PIDNET_DIR = Path("legacy_local/CARLE CODE/PIDNet").resolve()
sys.path.insert(0, str(PIDNET_DIR))
from models.pidnet import get_pred_model

YANG_DIR = Path("baselines/yang2026").resolve()
YANG_ENC = str(YANG_DIR / "encoder.exe")

SWEEPS_META_DIR = Path("results/phase1b/cache_pidnet_half/sweeps")
DATAROOT = Path("data/nuscenes")
OUT_DIR = Path("results/phase1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CITYSCAPES_TO_K4 = np.array([
    4, 4, 4, 4, 4, 4, 4,
    1, 1, 4, 4,
    3, 3,
    2, 2, 2, 2, 2, 2
], dtype=np.uint8)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

H, W = 450, 800
pad_h = int(np.ceil(H / 32.0) * 32)
pad_w = int(np.ceil(W / 32.0) * 32)

def encode_yang(frame: np.ndarray, tag: str) -> int:
    t_in = YANG_DIR / f"spk_{tag}_in.yuv"
    t_out = YANG_DIR / f"spk_{tag}_out.bin"
    frame.tofile(t_in)
    cmd = [YANG_ENC, "-i", str(t_in), "-o", str(t_out), "-r", str(H), "-c", str(W), "-f", "1", "-s", "0", "-t", "400"]
    subprocess.run(cmd, capture_output=True, cwd=str(YANG_DIR), stdin=subprocess.DEVNULL)
    b = t_out.stat().st_size if t_out.exists() else 0
    if t_in.exists():
        try: t_in.unlink()
        except: pass
    if t_out.exists():
        try: t_out.unlink()
        except: pass
    return b * 8

def run_speckle_analysis():
    print("="*80)
    print("STARTING NIGHT SPECKLE ANALYSIS: PIDNet-L (Half Res, K=4, Scenes 1094 & 1100)")
    print("="*80)
    
    # 1. Synthetic single isolated pixel test
    arr_uni = np.full((H, W), 4, dtype=np.uint8)
    uni_bits = encode_yang(arr_uni, "uni")
    arr_single = np.full((H, W), 4, dtype=np.uint8)
    arr_single[H//2, W//2] = 1 # road
    single_bits = encode_yang(arr_single, "single")
    print(f"Synthetic uniform map (all-other, 450x800): {uni_bits} bits")
    print(f"Synthetic single isolated pixel at center: {single_bits} bits (+{single_bits - uni_bits} bits)")
    
    # 2. Load PIDNet-L model
    ckpt_path = PIDNET_DIR / "pretrained_models/Cityscapes/PIDNet_L_Cityscapes_val.pt"
    model = get_pred_model("pidnet-l", 19)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = {k[6:]: v for k, v in ckpt.items() if k.startswith("model.")}
    model.load_state_dict(sd, strict=False)
    model.eval().cuda().half()
    
    records = []
    
    for sc_name in ["scene-1094", "scene-1100"]:
        meta_p = SWEEPS_META_DIR / f"sweeps_{sc_name}_meta.json"
        with open(meta_p, "r") as f:
            items = json.load(f)
            
        print(f"\nProcessing {sc_name} ({len(items)} frames)...")
        for idx, r in enumerate(items):
            img_p = DATAROOT / r["filename"]
            img = cv2.imread(str(img_p))
            if img is None: continue
            
            img_half = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
            img_rgb = cv2.cvtColor(img_half, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img_norm = (img_rgb - MEAN) / STD
            if pad_h != H or pad_w != W:
                img_norm = np.pad(img_norm, ((0, pad_h - H), (0, pad_w - W), (0, 0)), mode="constant")
            t_in = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).half().cuda()
            
            with torch.no_grad():
                out = model(t_in)
                if isinstance(out, (list, tuple)): out = out[0]
                out = torch.nn.functional.interpolate(out, size=(pad_h, pad_w), mode="bilinear", align_corners=True)
                if pad_h != H or pad_w != W:
                    out = out[:, :, :H, :W]
                pred_19 = torch.argmax(out, dim=1)[0].byte().cpu().numpy()
                
            pred_k4 = CITYSCAPES_TO_K4[pred_19]
            
            # Non-other pixels (classes 1, 2, 3)
            non_other_mask = (pred_k4 != 4).astype(np.uint8)
            num_non_other = int(np.sum(non_other_mask))
            
            # Connected components (4-connectivity)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(non_other_mask, connectivity=4)
            num_components = num_labels - 1 # exclude background (0)
            
            # Yang bits
            y_bits = encode_yang(pred_k4, f"{sc_name}_{idx}")
            
            records.append({
                "scene": sc_name,
                "camera": r["camera"],
                "frame_idx": idx,
                "yang_bits": y_bits,
                "non_other_pixels": num_non_other,
                "connected_components": num_components,
                "other_pct": round(float(np.mean(pred_k4 == 4) * 100.0), 4)
            })
            
            if (idx + 1) % 100 == 0:
                print(f"  Frame {idx+1}/{len(items)}: yang_bits={y_bits}, non_other_px={num_non_other}, cc={num_components}")
                
    df = pd.DataFrame(records)
    out_csv = OUT_DIR / "night_speckle_analysis.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved night speckle analysis to {out_csv} ({len(df)} rows)")
    
    # Statistical Summary
    corr_bits_cc = df["yang_bits"].corr(df["connected_components"])
    corr_bits_px = df["yang_bits"].corr(df["non_other_pixels"])
    
    # Bits per component (for frames with cc > 0)
    speckled = df[df["connected_components"] > 0].copy()
    speckled["bits_above_baseline"] = speckled["yang_bits"] - 48
    speckled["bits_per_component"] = speckled["bits_above_baseline"] / speckled["connected_components"]
    mean_bpc = speckled["bits_per_component"].mean()
    median_bpc = speckled["bits_per_component"].median()
    
    print("\n" + "="*80)
    print("SPECKLE ANALYSIS RESULTS:")
    print("="*80)
    print(f"Correlation (yang_bits vs connected_components): {corr_bits_cc:.4f}")
    print(f"Correlation (yang_bits vs non_other_pixels):      {corr_bits_px:.4f}")
    print(f"Mean bits per connected component:                {mean_bpc:.2f} bits/component")
    print(f"Median bits per connected component:              {median_bpc:.2f} bits/component")
    print(f"Mean non-other pixels (scene-1094):               {df[df['scene']=='scene-1094']['non_other_pixels'].mean():.2f}")
    print(f"Mean non-other pixels (scene-1100):               {df[df['scene']=='scene-1100']['non_other_pixels'].mean():.2f}")
    print("="*80)

if __name__ == "__main__":
    run_speckle_analysis()
