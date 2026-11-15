"""
Build Full Label Map Caches for Phase 1:
- nuScenes Sweeps: All 10 scenes, all 6 cameras (14,008 frames)
  Processed scene-by-scene to keep RAM strictly below 2 GB.
- Field Test Videos: All frames of all 6 clips (5,484 frames)
Using YOLO11n-seg on GPU (RTX 3050).
"""

import sys
import gc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(line_buffering=True)

import time
import json
import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO
from nuscenes.nuscenes import NuScenes

CACHE_DIR = Path("results/phase1/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
SWEEPS_CACHE_DIR = CACHE_DIR / "sweeps"
SWEEPS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

COCO_TO_CARLE = {
    0: 4,   # person -> Pedestrian (4)
    1: 3,   # bicycle -> Cyclist (3)
    2: 2,   # car -> Vehicle (2)
    3: 3,   # motorcycle -> Cyclist (3)
    5: 2,   # bus -> Vehicle (2)
    7: 2,   # truck -> Vehicle (2)
}

def segment_image(model, img_bgr):
    H, W = img_bgr.shape[:2]
    t0 = time.perf_counter()
    res = model(img_bgr, device="cuda", verbose=False)[0]
    torch.cuda.synchronize()
    gpu_ms = (time.perf_counter() - t0) * 1000.0
    
    mask = np.ones((H, W), dtype=np.uint8) # 1 = Road
    if res.masks is not None:
        for seg_box, seg_poly in zip(res.boxes, res.masks.xy):
            cls_id = int(seg_box.cls[0].item())
            mapped = COCO_TO_CARLE.get(cls_id, 0)
            if mapped > 0 and len(seg_poly) > 0:
                poly = np.array(seg_poly, dtype=np.int32)
                cv2.fillPoly(mask, [poly], int(mapped))
                
    return mask, gpu_ms

def build_full_sweeps_cache(model):
    dataroot = Path("data/nuscenes")
    manifest_path = CACHE_DIR / "nuscenes_sweeps_12hz_manifest.json"
    
    print("Loading nuScenes mini...")
    nusc = NuScenes(version="v1.0-mini", dataroot=str(dataroot), verbose=False)
    CAMERAS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
    
    scene_manifest = []
    total_sweeps_processed = 0
    t_start = time.perf_counter()
    
    for s_idx, scene in enumerate(nusc.scene):
        scene_name = scene["name"]
        scene_npz = SWEEPS_CACHE_DIR / f"sweeps_{scene_name}.npz"
        scene_meta_file = SWEEPS_CACHE_DIR / f"sweeps_{scene_name}_meta.json"
        
        if scene_npz.exists() and scene_meta_file.exists():
            print(f"[{s_idx+1}/10] Scene {scene_name} already cached, skipping.")
            with open(scene_meta_file) as f:
                s_meta = json.load(f)
            total_sweeps_processed += len(s_meta)
            scene_manifest.append({
                "scene": scene_name,
                "frames": len(s_meta),
                "npz_file": str(scene_npz.relative_to(CACHE_DIR)),
                "meta_file": str(scene_meta_file.relative_to(CACHE_DIR))
            })
            continue
            
        print(f"\n[{s_idx+1}/10] Caching sweeps for scene {scene_name} (all 6 cameras)...", flush=True)
        sweep_masks = []
        sweep_meta = []
        
        for cam in CAMERAS:
            sample = nusc.get("sample", scene["first_sample_token"])
            sd_token = sample["data"][cam]
            sd = nusc.get("sample_data", sd_token)
            while sd["prev"]:
                sd = nusc.get("sample_data", sd["prev"])
                
            while sd:
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
                if not sd["next"]:
                    break
                sd = nusc.get("sample_data", sd["next"])
                
        # Save scene cache
        scene_arr = np.stack(sweep_masks, axis=0)
        np.savez_compressed(scene_npz, masks=scene_arr)
        with open(scene_meta_file, "w") as f:
            json.dump(sweep_meta, f, indent=2)
            
        total_sweeps_processed += len(sweep_masks)
        scene_manifest.append({
            "scene": scene_name,
            "frames": len(sweep_masks),
            "npz_file": str(scene_npz.relative_to(CACHE_DIR)),
            "meta_file": str(scene_meta_file.relative_to(CACHE_DIR))
        })
        
        elapsed = time.perf_counter() - t_start
        fps = total_sweeps_processed / elapsed if elapsed > 0 else 0
        print(f"Saved {scene_name} ({len(sweep_masks)} frames, total {total_sweeps_processed}/14008 at {fps:.1f} fps).", flush=True)
        
        # Free memory immediately
        del sweep_masks, scene_arr, sweep_meta
        gc.collect()
        
    with open(manifest_path, "w") as f:
        json.dump(scene_manifest, f, indent=2)
    print(f"All sweeps cached successfully! Total: {total_sweeps_processed} frames.")

def build_full_field_cache(model):
    config_path = Path("configs/fieldtest.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
        
    dataroot = Path(cfg["dataset_root"])
    
    for clip in cfg["clips"]:
        clip_id = clip["clip_id"]
        out_native = CACHE_DIR / f"field_{clip_id}_native_full.npz"
        out_native_meta = CACHE_DIR / f"field_{clip_id}_native_full_meta.json"
        out_10hz = CACHE_DIR / f"field_{clip_id}_10hz_full.npz"
        out_10hz_meta = CACHE_DIR / f"field_{clip_id}_10hz_full_meta.json"
        
        if out_native.exists() and out_10hz.exists():
            print(f"Field cache for {clip_id} already exists, skipping.")
            continue
            
        v_path = dataroot / clip["filename"]
        if not v_path.exists():
            print(f"Video {v_path} not found!")
            continue
            
        cap = cv2.VideoCapture(str(v_path))
        fps = clip["native_fps"]
        stride_10hz = clip["sample_rates"][1]["stride"]
        
        native_masks = []
        native_meta = []
        hz10_masks = []
        hz10_meta = []
        
        idx = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"\nCaching ALL {total_frames} frames of field clip {clip_id} ({clip['condition']})...", flush=True)
        t0 = time.perf_counter()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            mask, gpu_ms = segment_image(model, frame)
            
            meta_item = {
                "clip_id": clip_id,
                "frame_idx": idx,
                "timestamp_s": round(idx / fps, 4),
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
            if idx % 500 == 0 or idx == total_frames:
                elapsed = time.perf_counter() - t0
                print(f"  [{clip_id}] {idx}/{total_frames} frames ({idx/elapsed:.1f} fps)...", flush=True)
                
        cap.release()
        
        # Save native
        arr_native = np.stack(native_masks, axis=0)
        np.savez_compressed(out_native, masks=arr_native)
        with open(out_native_meta, "w") as f:
            json.dump(native_meta, f, indent=2)
            
        # Save 10 Hz
        arr_10hz = np.stack(hz10_masks, axis=0)
        np.savez_compressed(out_10hz, masks=arr_10hz)
        with open(out_10hz_meta, "w") as f:
            json.dump(hz10_meta, f, indent=2)
            
        print(f"Saved {clip_id}: native={len(native_masks)} frames, 10hz={len(hz10_masks)} frames!", flush=True)
        del native_masks, arr_native, hz10_masks, arr_10hz
        gc.collect()

def main():
    print(f"Initializing YOLO11n-seg on GPU: {torch.cuda.get_device_name(0)}...")
    model = YOLO("models/yolo11n-seg.pt")
    
    # 1. Full nuScenes sweeps (14,008 frames, chunked scene-by-scene)
    build_full_sweeps_cache(model)
    
    # 2. Full field videos (5,484 frames, clip-by-clip)
    build_full_field_cache(model)
    
    print("\nAll full caches built successfully!")

if __name__ == "__main__":
    main()
