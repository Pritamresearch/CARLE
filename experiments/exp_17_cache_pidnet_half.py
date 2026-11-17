"""
Phase 1b: Cache PIDNet-S Semantic Maps at Half-Resolution (450x800)
Projects Cityscapes predictions to:
  K=4: 1 Road, 2 Vehicle, 3 Vulnerable Road User, 4 Other Static
  K=8: 1 Road, 2 Sidewalk, 3 Static Structure, 4 Vegetation/Terrain, 5 Sky, 6 Vehicle, 7 Two-Wheeler, 8 Person/Rider
All frames at 450x800:
- nuScenes Keyframes (all 2,424)
- nuScenes Sweeps (all 6,000 across 10 scenes, all 6 cameras)
- Field Native & 10 Hz (all 5 unflagged clips + flagged Bhubaneswar night)
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
import yaml
from collections import defaultdict

PIDNET_DIR = Path("legacy_local/CARLE CODE/PIDNet").resolve()
sys.path.insert(0, str(PIDNET_DIR))
from models.pidnet import get_pred_model

CACHE_DIR_HALF = Path("results/phase1b/cache_pidnet_half")
CACHE_DIR_HALF.mkdir(parents=True, exist_ok=True)
SWEEPS_DIR_HALF = CACHE_DIR_HALF / "sweeps"
SWEEPS_DIR_HALF.mkdir(parents=True, exist_ok=True)

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

TARGET_H = 450
TARGET_W = 800
PAD_H = 480
PAD_W = 800

def load_pidnet_model():
    print("Loading PIDNet-S Cityscapes model in FP16 on GPU...")
    model = get_pred_model("pidnet-s", 19)
    weight_path = PIDNET_DIR / "pretrained_models/Cityscapes/PIDNet_S_Cityscapes_val.pt"
    ckpt = torch.load(weight_path, map_location="cpu")
    sd = {k[6:]: v for k, v in ckpt.items() if k.startswith("model.")}
    model.load_state_dict(sd, strict=False)
    model.eval().cuda().half()
    print("PIDNet-S loaded successfully.")
    return model

def segment_batch_pidnet_half(model, img_bgr_list, device="cuda"):
    B = len(img_bgr_list)
    batch_tensor = torch.empty((B, 3, PAD_H, PAD_W), dtype=torch.float16, device=device)
    
    for b in range(B):
        img_bgr = img_bgr_list[b]
        res = cv2.resize(img_bgr, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
        rgb_norm = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
        padded = np.zeros((PAD_H, PAD_W, 3), dtype=np.float32)
        padded[:TARGET_H, :TARGET_W] = rgb_norm
        batch_tensor[b] = torch.from_numpy(padded).permute(2, 0, 1).half()
        
    with torch.inference_mode():
        pred = model(batch_tensor)
        if isinstance(pred, (list, tuple)): pred = pred[0]
        pred = torch.nn.functional.interpolate(pred, size=(PAD_H, PAD_W), mode="bilinear", align_corners=True)
        pred = pred[:, :, :TARGET_H, :TARGET_W]
        raw_classes = torch.argmax(pred, dim=1).byte().cpu().numpy()
        
    k4_batch = CITYSCAPES_TO_K4[raw_classes]
    k8_batch = CITYSCAPES_TO_K8[raw_classes]
    return k4_batch, k8_batch

def cache_nuscenes_keyframes(model):
    print("\n--- 1. Caching nuScenes Keyframes (2 Hz) at 450x800 ---")
    out_k4 = CACHE_DIR_HALF / "nuscenes_keyframes_2hz_k4.npz"
    out_k8 = CACHE_DIR_HALF / "nuscenes_keyframes_2hz_k8.npz"
    if out_k4.exists() and out_k8.exists():
        print("nuScenes keyframes already cached at 450x800. Skipping.")
        return
        
    meta_p = Path("results/phase1/cache/nuscenes_keyframes_2hz_meta.json")
    with open(meta_p, "r") as f:
        meta_list = json.load(f)
        
    dataroot = Path("data/nuscenes")
    N = len(meta_list)
    print(f"Total keyframes to segment: {N}")
    
    batch_size = 6
    masks_k4_all = []
    masks_k8_all = []
    
    t0 = time.perf_counter()
    for i in range(0, N, batch_size):
        batch_meta = meta_list[i : i + batch_size]
        batch_imgs = [cv2.imread(str(dataroot / m["filename"])) for m in batch_meta]
        m_k4, m_k8 = segment_batch_pidnet_half(model, batch_imgs)
        for b in range(len(batch_meta)):
            masks_k4_all.append(m_k4[b])
            masks_k8_all.append(m_k8[b])
            
        if (i // batch_size) % 50 == 0 or i + batch_size >= N:
            fps = min(i + batch_size, N) / (time.perf_counter() - t0)
            print(f"Keyframes: {min(i + batch_size, N)}/{N} segmented ({fps:.1f} FPS)")
            
    masks_k4_arr = np.array(masks_k4_all, dtype=np.uint8)
    masks_k8_arr = np.array(masks_k8_all, dtype=np.uint8)
    
    np.savez_compressed(CACHE_DIR_HALF / "nuscenes_keyframes_2hz_k4.npz", masks=masks_k4_arr)
    np.savez_compressed(CACHE_DIR_HALF / "nuscenes_keyframes_2hz_k8.npz", masks=masks_k8_arr)
    with open(CACHE_DIR_HALF / "nuscenes_keyframes_2hz_meta.json", "w") as f:
        json.dump(meta_list, f, indent=2)
        
    print(f"Saved nuScenes keyframes: K4 shape={masks_k4_arr.shape}, K8 shape={masks_k8_arr.shape}")

def cache_nuscenes_sweeps(model):
    print("\n--- 2. Caching nuScenes Sweeps (12 Hz, All 6 Cameras) at 450x800 ---")
    cache_yolo_dir = Path("results/phase1/cache/sweeps")
    sweep_files = sorted(list(cache_yolo_dir.glob("sweeps_scene-*.npz")))
    dataroot = Path("data/nuscenes")
    
    batch_size = 6
    total_sweeps = 0
    
    t0 = time.perf_counter()
    for sf in sweep_files:
        sc_name = sf.stem.replace("sweeps_", "")
        out_s_k4 = SWEEPS_DIR_HALF / f"sweeps_{sc_name}_k4.npz"
        out_s_k8 = SWEEPS_DIR_HALF / f"sweeps_{sc_name}_k8.npz"
        if out_s_k4.exists() and out_s_k8.exists():
            print(f"Scene {sc_name} already cached at 450x800. Skipping.")
            total_sweeps += 600
            continue
            
        meta_p = cache_yolo_dir / f"sweeps_{sc_name}_meta.json"
        with open(meta_p, "r") as f:
            meta = json.load(f)
            
        by_cam = defaultdict(list)
        for i, m in enumerate(meta):
            by_cam[m["camera"]].append(m)
            
        scene_k4 = []
        scene_k8 = []
        scene_meta_out = []
        
        for cam in ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]:
            c_meta = by_cam[cam]
            c_meta.sort(key=lambda x: x["timestamp"])
            selected = c_meta[:100]
            
            for i in range(0, len(selected), batch_size):
                b_meta = selected[i : i + batch_size]
                b_imgs = [cv2.imread(str(dataroot / m["filename"])) for m in b_meta]
                m_k4, m_k8 = segment_batch_pidnet_half(model, b_imgs)
                
                for b in range(len(b_meta)):
                    scene_k4.append(m_k4[b])
                    scene_k8.append(m_k8[b])
                    scene_meta_out.append(b_meta[b])
                    
        sc_k4_arr = np.array(scene_k4, dtype=np.uint8)
        sc_k8_arr = np.array(scene_k8, dtype=np.uint8)
        
        np.savez_compressed(SWEEPS_DIR_HALF / f"sweeps_{sc_name}_k4.npz", masks=sc_k4_arr)
        np.savez_compressed(SWEEPS_DIR_HALF / f"sweeps_{sc_name}_k8.npz", masks=sc_k8_arr)
        with open(SWEEPS_DIR_HALF / f"sweeps_{sc_name}_meta.json", "w") as f:
            json.dump(scene_meta_out, f, indent=2)
            
        total_sweeps += len(scene_k4)
        print(f"Scene {sc_name}: cached {len(scene_k4)} frames across 6 cameras")
        
    print(f"Total nuScenes sweeps cached at 450x800: {total_sweeps} frames")

def cache_field_videos(model):
    print("\n--- 3. Caching Field Videos at 450x800 (Native & 10 Hz) ---")
    cfg_p = Path("configs/fieldtest.yaml")
    with open(cfg_p, "r") as f:
        cfg = yaml.safe_load(f)
        
    clips = cfg["clips"]
    batch_size = 6
    
    for clip in clips:
        cid = clip["clip_id"]
        out_f_k4 = CACHE_DIR_HALF / f"field_{cid}_native_k4.npz"
        out_f_k8 = CACHE_DIR_HALF / f"field_{cid}_native_k8.npz"
        if out_f_k4.exists() and out_f_k8.exists():
            print(f"Clip {cid} already cached at 450x800. Skipping.")
            continue
            
        vpath = Path("data/field_videos") / clip["filename"]
        if not vpath.exists():
            print(f"Video {vpath} not found!")
            continue
            
        cap = cv2.VideoCapture(str(vpath))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frames_all = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frames_all.append(frame)
        cap.release()
        
        N_tot = len(frames_all)
        print(f"\nProcessing {cid}: {N_tot} frames at {fps:.2f} fps -> target 450x800")
        
        k4_all = []
        k8_all = []
        
        t0 = time.perf_counter()
        for i in range(0, N_tot, batch_size):
            b_imgs = frames_all[i : i + batch_size]
            m_k4, m_k8 = segment_batch_pidnet_half(model, b_imgs)
            for b in range(len(b_imgs)):
                k4_all.append(m_k4[b])
                k8_all.append(m_k8[b])
                
        k4_arr = np.array(k4_all, dtype=np.uint8)
        k8_arr = np.array(k8_all, dtype=np.uint8)
        
        # Save Native
        np.savez_compressed(CACHE_DIR_HALF / f"field_{cid}_native_k4.npz", masks=k4_arr)
        np.savez_compressed(CACHE_DIR_HALF / f"field_{cid}_native_k8.npz", masks=k8_arr)
        nat_meta = [{"frame_idx": idx, "fps": fps} for idx in range(N_tot)]
        with open(CACHE_DIR_HALF / f"field_{cid}_native_meta.json", "w") as f:
            json.dump(nat_meta, f, indent=2)
            
        # 10 Hz subsampled (stride = round(fps / 10))
        stride = max(1, int(round(fps / 10.0)))
        k4_10hz = k4_arr[::stride]
        k8_10hz = k8_arr[::stride]
        np.savez_compressed(CACHE_DIR_HALF / f"field_{cid}_10hz_k4.npz", masks=k4_10hz)
        np.savez_compressed(CACHE_DIR_HALF / f"field_{cid}_10hz_k8.npz", masks=k8_10hz)
        meta_10hz = [nat_meta[idx] for idx in range(0, N_tot, stride)]
        with open(CACHE_DIR_HALF / f"field_{cid}_10hz_meta.json", "w") as f:
            json.dump(meta_10hz, f, indent=2)
            
        print(f"{cid}: Saved Native ({len(k4_arr)} frames) & 10 Hz ({len(k4_10hz)} frames) at 450x800")

if __name__ == "__main__":
    model = load_pidnet_model()
    cache_nuscenes_keyframes(model)
    cache_nuscenes_sweeps(model)
    cache_field_videos(model)
    print("\nAll 450x800 half-resolution caches built successfully!")
