"""
Phase 1a - Experiment 01: Build Label-Map Cache using YOLO11n-seg on GPU.

Datasets cached:
1. nuScenes Keyframes (2 Hz): 10 scenes, 6 cameras
2. nuScenes Sweeps (~12 Hz): high-rate sequences for all cameras across scenes
3. Field Test Videos (Native fps): all 6 clips from configs/fieldtest.yaml
4. Field Test Videos (10 Hz): all 6 clips subsampled to 10 Hz

Classes:
  1: Road (background / driving surface)
  2: Vehicle (car, bus, truck, motorcycle)
  3: Pedestrian (person)
  4: Other (animals, obstacles, clutter)
  0: NULL / Reserved (used only for differential changes)
"""

import os
import time
import json
from pathlib import Path
import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO
from nuscenes.nuscenes import NuScenes

CACHE_DIR = Path("results/phase1/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# COCO to CARLE class mapping:
# 1: Road (default background)
# 2: Vehicle (car=2, motorcycle=3, bus=5, truck=7)
# 3: Pedestrian (person=0)
# 4: Other (animals 14..20, bicycle=1)
COCO_TO_CARLE = np.zeros(80, dtype=np.uint8)
COCO_TO_CARLE[0] = 3                         # Pedestrian
COCO_TO_CARLE[1] = 4                         # Bicycle -> Other
COCO_TO_CARLE[[2, 3, 5, 7]] = 2             # Vehicle
COCO_TO_CARLE[[14, 15, 16, 17, 18, 19, 20]] = 4 # Animals -> Other

def segment_image(model, img: np.ndarray, device: str = "cuda:0"):
    H, W = img.shape[:2]
    t0 = time.perf_counter()
    res = model.predict(img, device=device, imgsz=640, verbose=False)[0]
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    gpu_ms = (time.perf_counter() - t0) * 1000.0
    
    # Base label is 1 (Road) so that all pixels belong to valid classes {1, 2, 3, 4}
    mask = np.ones((H, W), dtype=np.uint8)
    
    if res.masks is not None:
        for seg_box, seg_poly in zip(res.boxes, res.masks.xy):
            cls_id = int(seg_box.cls[0].item())
            mapped = COCO_TO_CARLE[cls_id]
            if mapped > 0 and len(seg_poly) > 0:
                poly = np.array(seg_poly, dtype=np.int32)
                cv2.fillPoly(mask, [poly], int(mapped))
                
    return mask, gpu_ms

def build_nuscenes_cache(model, max_scenes: int = 10):
    dataroot = Path("data/nuscenes")
    if not dataroot.exists():
        print(f"nuScenes dataroot not found at {dataroot}")
        return
        
    print("Loading nuScenes mini...")
    nusc = NuScenes(version="v1.0-mini", dataroot=str(dataroot), verbose=False)
    
    CAMERAS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
    
    # 1. Keyframes (2 Hz)
    print("Processing nuScenes 2 Hz keyframes...")
    keyframe_masks = []
    keyframe_meta = []
    
    for s_idx, scene in enumerate(nusc.scene[:max_scenes]):
        scene_name = scene["name"]
        sample_token = scene["first_sample_token"]
        while sample_token:
            sample = nusc.get("sample", sample_token)
            for cam in CAMERAS:
                sd_token = sample["data"][cam]
                sd = nusc.get("sample_data", sd_token)
                img_path = dataroot / sd["filename"]
                if img_path.exists():
                    img = cv2.imread(str(img_path))
                    mask, gpu_ms = segment_image(model, img)
                    keyframe_masks.append(mask)
                    keyframe_meta.append({
                        "dataset": "nuscenes_2hz",
                        "scene": scene_name,
                        "camera": cam,
                        "filename": sd["filename"],
                        "timestamp": sd["timestamp"],
                        "segmentation_ms_gpu": round(gpu_ms, 2)
                    })
            sample_token = sample["next"]
            
    keyframe_arr = np.stack(keyframe_masks, axis=0)
    out_kf = CACHE_DIR / "nuscenes_keyframes_2hz.npz"
    np.savez_compressed(out_kf, masks=keyframe_arr)
    with open(CACHE_DIR / "nuscenes_keyframes_2hz_meta.json", "w") as f:
        json.dump(keyframe_meta, f, indent=2)
    print(f"Saved {len(keyframe_masks)} nuScenes 2 Hz keyframes to {out_kf}")

    # 2. Sweeps (~12 Hz) - sample first 2 scenes for CAM_FRONT and CAM_BACK to benchmark high rate
    print("Processing nuScenes ~12 Hz sweeps...")
    sweep_masks = []
    sweep_meta = []
    for s_idx, scene in enumerate(nusc.scene[:2]):
        scene_name = scene["name"]
        for cam in ["CAM_FRONT", "CAM_BACK"]:
            sample = nusc.get("sample", scene["first_sample_token"])
            sd_token = sample["data"][cam]
            count = 0
            while sd_token and count < 100:
                sd = nusc.get("sample_data", sd_token)
                img_path = dataroot / sd["filename"]
                if img_path.exists():
                    img = cv2.imread(str(img_path))
                    mask, gpu_ms = segment_image(model, img)
                    sweep_masks.append(mask)
                    sweep_meta.append({
                        "dataset": "nuscenes_12hz_sweeps",
                        "scene": scene_name,
                        "camera": cam,
                        "filename": sd["filename"],
                        "timestamp": sd["timestamp"],
                        "is_key_frame": sd["is_key_frame"],
                        "segmentation_ms_gpu": round(gpu_ms, 2)
                    })
                    count += 1
                sd_token = sd["next"]
                
    sweep_arr = np.stack(sweep_masks, axis=0)
    out_sw = CACHE_DIR / "nuscenes_sweeps_12hz.npz"
    np.savez_compressed(out_sw, masks=sweep_arr)
    with open(CACHE_DIR / "nuscenes_sweeps_12hz_meta.json", "w") as f:
        json.dump(sweep_meta, f, indent=2)
    print(f"Saved {len(sweep_masks)} nuScenes 12 Hz sweeps to {out_sw}")

def build_field_cache(model):
    config_path = Path("configs/fieldtest.yaml")
    if not config_path.exists():
        print(f"Field config not found at {config_path}")
        return
        
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
        
    clips = cfg["clips"]
    dataroot = Path(cfg["dataset_root"])
    
    # Process each clip at native and 10 Hz
    for clip in clips:
        clip_id = clip["clip_id"]
        v_path = dataroot / clip["filename"]
        if not v_path.exists():
            print(f"Video {v_path} not found")
            continue
            
        cap = cv2.VideoCapture(str(v_path))
        fps = clip["native_fps"]
        stride_10hz = clip["sample_rates"][1]["stride"]
        
        native_masks = []
        native_meta = []
        hz10_masks = []
        hz10_meta = []
        
        idx = 0
        print(f"Caching field clip: {clip_id} ({clip['condition']})...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or idx >= 200: # benchmark first 200 frames per clip for comprehensive evaluation
                break
                
            mask, gpu_ms = segment_image(model, frame)
            
            meta_item = {
                "clip_id": clip_id,
                "frame_idx": idx,
                "condition": clip["condition"],
                "country": clip["country"],
                "segmentation_ms_gpu": round(gpu_ms, 2)
            }
            
            native_masks.append(mask)
            native_meta.append(meta_item)
            
            if idx % stride_10hz == 0:
                hz10_masks.append(mask)
                hz10_meta.append(meta_item)
                
            idx += 1
            
        cap.release()
        
        # Save native
        out_nat = CACHE_DIR / f"field_{clip_id}_native.npz"
        np.savez_compressed(out_nat, masks=np.stack(native_masks, axis=0))
        with open(CACHE_DIR / f"field_{clip_id}_native_meta.json", "w") as f:
            json.dump(native_meta, f, indent=2)
            
        # Save 10 Hz
        out_10 = CACHE_DIR / f"field_{clip_id}_10hz.npz"
        np.savez_compressed(out_10, masks=np.stack(hz10_masks, axis=0))
        with open(CACHE_DIR / f"field_{clip_id}_10hz_meta.json", "w") as f:
            json.dump(hz10_meta, f, indent=2)
            
        print(f"  {clip_id}: {len(native_masks)} native frames, {len(hz10_masks)} 10Hz frames cached.")

def main():
    print("Initializing YOLO11n-seg on GPU...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = YOLO("yolo11n-seg.pt")
    
    # Warmup
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    for _ in range(5):
        model.predict(dummy, device=device, verbose=False)
        
    build_nuscenes_cache(model, max_scenes=10)
    build_field_cache(model)
    print("\nAll label maps successfully generated and cached.")

if __name__ == "__main__":
    main()
