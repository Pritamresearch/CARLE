"""
Phase 1 Study: nuScenes Sweep Scene Metadata (Day/Night), Night Overlays, and K=4 Class Histograms
==================================================================================================
Requirements:
1. Report whether each of the 10 nuScenes scenes is Day or Night (from scene.json description).
2. For night scenes (scene-1077, scene-1094, scene-1100), save 3 example visual overlays per model
   (PIDNet-S, PIDNet-M, PIDNet-L, half resolution 450x800).
3. Compute the exact K=4 class histogram (% Road, % Vehicle, % VRU, % Other) per model per scene.
Outputs:
- results/phase1/sweep_scenes_day_night_and_histograms.csv
- results/phase1/overlays/night_scene_overlays/*.png
"""

import sys
import os
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PIDNET_DIR = Path("legacy_local/CARLE CODE/PIDNet").resolve()
sys.path.insert(0, str(PIDNET_DIR))
from models.pidnet import get_pred_model

DATAROOT = Path("data/nuscenes")
SWEEPS_META_DIR = Path("results/phase1/cache_pidnet/sweeps")
OUT_DIR = Path("results/phase1")
OVERLAY_DIR = OUT_DIR / "overlays" / "night_scene_overlays"
OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

# Cityscapes 19 to K=4 mapping:
# 0: road -> 1
# 13, 14, 15, 16: vehicles -> 2
# 11, 12, 17, 18: VRU -> 3
# 1..10: other -> 4
CITYSCAPES_TO_K4 = np.zeros(19, dtype=np.uint8)
CITYSCAPES_TO_K4[0] = 1
CITYSCAPES_TO_K4[[13, 14, 15, 16]] = 2
CITYSCAPES_TO_K4[[11, 12, 17, 18]] = 3
CITYSCAPES_TO_K4[list(range(1, 11))] = 4

PALETTE_K4 = np.array([
    [0, 0, 0],          # 0: Void
    [128, 64, 128],     # 1: Road (Purple)
    [142, 0, 0],        # 2: Vehicle (Blue)
    [60, 20, 220],      # 3: VRU (Red)
    [70, 70, 70]        # 4: Other (Gray)
], dtype=np.uint8)

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

def main():
    print("="*80)
    print("SWEEP SCENES DAY/NIGHT AUDIT, K=4 HISTOGRAMS, AND NIGHT OVERLAYS")
    print("="*80)
    
    # 1. Parse scene.json for day/night descriptions
    scene_json_p = DATAROOT / "v1.0-mini" / "scene.json"
    with open(scene_json_p, "r") as f:
        scenes_meta = json.load(f)
        
    scene_env = {}
    for sc in scenes_meta:
        name = sc["name"]
        desc = sc["description"]
        is_night = "night" in desc.lower()
        scene_env[name] = {
            "description": desc,
            "condition": "Night" if is_night else "Day"
        }
        print(f"Scene: {name:12s} | Condition: {scene_env[name]['condition']:5s} | Desc: {desc}")
        
    # 2. Process models: PIDNet-S, PIDNet-M, PIDNet-L at half resolution (450x800)
    H, W = 450, 800
    pad_h = ((H + 31) // 32) * 32
    pad_w = ((W + 31) // 32) * 32
    
    models = ["pidnet-s", "pidnet-m", "pidnet-l"]
    night_scenes = ["scene-1077", "scene-1094", "scene-1100"]
    
    hist_rows = []
    
    for arch in models:
        print(f"\n---> Loading {arch.upper()} for class histogram and overlay generation...")
        model = load_pidnet(arch)
        
        for sc_name in sorted(list(scene_env.keys())):
            meta_p = SWEEPS_META_DIR / f"sweeps_{sc_name}_meta.json"
            if not meta_p.exists():
                continue
            with open(meta_p, "r") as f:
                records = json.load(f)
                
            # Filter CAM_FRONT for consistent representative overlays and evaluate across all frames for histogram
            class_counts = {1: 0, 2: 0, 3: 0, 4: 0}
            total_pixels = 0
            
            overlay_saved = 0
            
            for idx, r in enumerate(records):
                img_p = DATAROOT / r["filename"]
                img = cv2.imread(str(img_p))
                if img is None: continue
                
                # Resize to half res
                img_half = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
                
                # Preprocess
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
                
                for c in range(1, 5):
                    class_counts[c] += int(np.sum(pred_k4 == c))
                total_pixels += (H * W)
                
                # Save 3 example overlays for night scenes on CAM_FRONT
                if sc_name in night_scenes and r["camera"] == "CAM_FRONT" and overlay_saved < 3 and idx % 25 == 0:
                    overlay_saved += 1
                    c_map = PALETTE_K4[pred_k4]
                    blended = cv2.addWeighted(img_half, 0.5, c_map, 0.5, 0)
                    
                    # Side-by-side composite: Original | Segmentation Map | Blended Overlay
                    composite = np.hstack([img_half, c_map, blended])
                    out_p = OVERLAY_DIR / f"{arch}_{sc_name}_overlay_{overlay_saved}.png"
                    cv2.imwrite(str(out_p), composite)
                    print(f"  Saved night overlay: {out_p.name}")
                    
            road_pct = round(class_counts[1] / total_pixels * 100.0, 2)
            veh_pct = round(class_counts[2] / total_pixels * 100.0, 2)
            vru_pct = round(class_counts[3] / total_pixels * 100.0, 2)
            other_pct = round(class_counts[4] / total_pixels * 100.0, 2)
            
            hist_rows.append({
                "Model": arch,
                "Scene": sc_name,
                "Condition": scene_env[sc_name]["condition"],
                "Description": scene_env[sc_name]["description"],
                "Road_Pct": road_pct,
                "Vehicle_Pct": veh_pct,
                "VRU_Pct": vru_pct,
                "Other_Pct": other_pct
            })
            print(f"  [{arch:8s} | {sc_name}] Road: {road_pct:5.2f}% | Veh: {veh_pct:5.2f}% | VRU: {vru_pct:5.2f}% | Other: {other_pct:5.2f}%")
            
        del model
        torch.cuda.empty_cache()
        
    df_out = pd.DataFrame(hist_rows)
    out_csv = OUT_DIR / "sweep_scenes_day_night_and_histograms.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"\nSaved scene condition and class histograms to {out_csv}")
    print(f"Saved all night visual overlays to {OVERLAY_DIR}")

if __name__ == "__main__":
    main()
