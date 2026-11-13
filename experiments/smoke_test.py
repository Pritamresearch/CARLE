"""
Smoke test script for CARLE Phase S:
- Loads 5 frames from each available dataset:
  1. nuScenes mini
  2. Cityscapes
  3. IDD
  4. Field-test video (sikim.mp4)
- Runs YOLO11n-seg inference
- Maps detections to priority classes
- Saves overlay images in results/setup/
- Measures and prints inference and overlay timing
"""

import os
import time
from pathlib import Path
import cv2
import numpy as np
import yaml
from ultralytics import YOLO

# 1. Load config paths
CONFIG_PATH = Path(__file__).parent.parent / "configs" / "paths.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

OUTPUT_DIR = Path(__file__).parent.parent / "results" / "setup"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Priority Class Map & Colors (BGR for OpenCV)
# COCO class ID -> CARLE class ID:
# 2: Pedestrian (person = 0)
# 3: Vehicle (car=2, motorcycle=3, bus=5, truck=7)
# 4: Animal (bird=14, cat=15, dog=16, horse=17, sheep=18, cow=19, elephant=20)
COCO_TO_CARLE = np.zeros(80, dtype=np.uint8)
COCO_TO_CARLE[0] = 2
COCO_TO_CARLE[[2, 3, 5, 7]] = 3
COCO_TO_CARLE[[14, 15, 16, 17, 18, 19, 20]] = 4

PALETTE = {
    0: (0, 0, 0),        # Background
    1: (128, 64, 128),   # Road (purple)
    2: (0, 255, 0),      # Pedestrian (green)
    3: (255, 128, 0),    # Vehicle (blue/cyan)
    4: (0, 165, 255),    # Animal (orange)
}
CLASS_NAMES = {
    0: "Background",
    1: "Road",
    2: "Pedestrian",
    3: "Vehicle",
    4: "Animal"
}

def create_overlay(img_bgr, priority_mask, alpha=0.45):
    """Overlay priority mask onto original image."""
    H, W = priority_mask.shape
    if img_bgr.shape[:2] != (H, W):
        img_bgr = cv2.resize(img_bgr, (W, H))

    overlay = np.zeros_like(img_bgr)
    for cls_id, color in PALETTE.items():
        if cls_id == 0:
            continue
        overlay[priority_mask == cls_id] = color

    blended = cv2.addWeighted(img_bgr, 1 - alpha, overlay, alpha, 0)
    
    # Legend
    y_offset = 30
    for cls_id, name in CLASS_NAMES.items():
        if cls_id == 0 or not np.any(priority_mask == cls_id):
            continue
        cv2.putText(blended, f"{name} ({cls_id})", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, PALETTE[cls_id], 2, cv2.LINE_AA)
        y_offset += 30

    return blended

def run_smoke_test():
    print("=" * 60)
    print("CARLE Phase S Smoke Test")
    print("=" * 60)

    # Load YOLO11n-seg model
    print("Loading YOLO11n-seg model...")
    model = YOLO("yolo11n-seg.pt")
    print("Model loaded successfully.\n")

    timing_results = {}

    # --- Dataset 1: nuScenes ---
    print("1. Processing nuScenes (5 frames)...")
    nusc_samples_dir = Path(config["datasets"]["nuscenes"]["root"]) / "samples" / "CAM_FRONT"
    nusc_frames = sorted(list(nusc_samples_dir.glob("*.jpg")))[:5]
    
    nusc_times = []
    for i, frame_path in enumerate(nusc_frames):
        t0 = time.perf_counter()
        img = cv2.imread(str(frame_path))
        H, W = img.shape[:2]
        
        results = model.predict(img, imgsz=640, verbose=False)[0]
        mask = np.zeros((H, W), dtype=np.uint8)
        
        if results.masks is not None:
            seg_masks = results.masks.data.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            priorities = COCO_TO_CARLE[np.clip(classes, 0, 79)]
            for s_idx, p in enumerate(priorities):
                if p > 0:
                    m = cv2.resize(seg_masks[s_idx], (W, H), interpolation=cv2.INTER_NEAREST)
                    mask[m > 0.5] = p

        dt = (time.perf_counter() - t0) * 1000
        nusc_times.append(dt)
        
        vis = create_overlay(img, mask)
        out_path = OUTPUT_DIR / f"smoke_nuscenes_{i+1}.jpg"
        cv2.imwrite(str(out_path), vis)
        print(f"   [nuScenes {i+1}/5] {frame_path.name} -> {dt:.1f}ms | Unique classes: {np.unique(mask).tolist()}")

    timing_results["nuScenes"] = np.median(nusc_times)

    # --- Dataset 2: Cityscapes ---
    print("\n2. Processing Cityscapes (5 frames)...")
    cs_dir = Path(config["datasets"]["cityscapes"]["left_img8bit"]) / "val" / "frankfurt"
    cs_frames = sorted(list(cs_dir.glob("*.png")))[:5]

    cs_times = []
    for i, frame_path in enumerate(cs_frames):
        t0 = time.perf_counter()
        img = cv2.imread(str(frame_path))
        H, W = img.shape[:2]

        results = model.predict(img, imgsz=640, verbose=False)[0]
        mask = np.zeros((H, W), dtype=np.uint8)

        if results.masks is not None:
            seg_masks = results.masks.data.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            priorities = COCO_TO_CARLE[np.clip(classes, 0, 79)]
            for s_idx, p in enumerate(priorities):
                if p > 0:
                    m = cv2.resize(seg_masks[s_idx], (W, H), interpolation=cv2.INTER_NEAREST)
                    mask[m > 0.5] = p

        dt = (time.perf_counter() - t0) * 1000
        cs_times.append(dt)

        vis = create_overlay(img, mask)
        out_path = OUTPUT_DIR / f"smoke_cityscapes_{i+1}.jpg"
        cv2.imwrite(str(out_path), vis)
        print(f"   [Cityscapes {i+1}/5] {frame_path.name} -> {dt:.1f}ms | Unique classes: {np.unique(mask).tolist()}")

    timing_results["Cityscapes"] = np.median(cs_times)

    # --- Dataset 3: IDD (India Driving Dataset) ---
    print("\n3. Processing IDD (5 frames)...")
    idd_dir = Path(config["datasets"]["idd"]["left_img8bit"]) / "val"
    idd_frames = (sorted(list(idd_dir.rglob("*.jpg"))) + sorted(list(idd_dir.rglob("*.png"))))[:5]

    idd_times = []
    for i, frame_path in enumerate(idd_frames):
        t0 = time.perf_counter()
        img = cv2.imread(str(frame_path))
        H, W = img.shape[:2]

        results = model.predict(img, imgsz=640, verbose=False)[0]
        mask = np.zeros((H, W), dtype=np.uint8)

        if results.masks is not None:
            seg_masks = results.masks.data.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            priorities = COCO_TO_CARLE[np.clip(classes, 0, 79)]
            for s_idx, p in enumerate(priorities):
                if p > 0:
                    m = cv2.resize(seg_masks[s_idx], (W, H), interpolation=cv2.INTER_NEAREST)
                    mask[m > 0.5] = p

        dt = (time.perf_counter() - t0) * 1000
        idd_times.append(dt)

        vis = create_overlay(img, mask)
        out_path = OUTPUT_DIR / f"smoke_idd_{i+1}.jpg"
        cv2.imwrite(str(out_path), vis)
        print(f"   [IDD {i+1}/5] {frame_path.name} -> {dt:.1f}ms | Unique classes: {np.unique(mask).tolist()}")

    timing_results["IDD"] = np.median(idd_times)

    # --- Dataset 4: Field-test Video (sikim.mp4) ---
    print("\n4. Processing Field-test Video (5 frames from sikim.mp4)...")
    video_path = Path(config["datasets"]["field_videos"]["root"]) / "sikim.mp4"
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = [int(x) for x in np.linspace(10, total_frames - 10, 5)]

    vid_times = []
    for i, f_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, img = cap.read()
        if not ret:
            continue
        H, W = img.shape[:2]

        t0 = time.perf_counter()
        results = model.predict(img, imgsz=640, verbose=False)[0]
        mask = np.zeros((H, W), dtype=np.uint8)

        if results.masks is not None:
            seg_masks = results.masks.data.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            priorities = COCO_TO_CARLE[np.clip(classes, 0, 79)]
            for s_idx, p in enumerate(priorities):
                if p > 0:
                    m = cv2.resize(seg_masks[s_idx], (W, H), interpolation=cv2.INTER_NEAREST)
                    mask[m > 0.5] = p

        dt = (time.perf_counter() - t0) * 1000
        vid_times.append(dt)

        vis = create_overlay(img, mask)
        out_path = OUTPUT_DIR / f"smoke_field_video_{i+1}.jpg"
        cv2.imwrite(str(out_path), vis)
        print(f"   [Field Video Frame {f_idx}] -> {dt:.1f}ms | Unique classes: {np.unique(mask).tolist()}")

    cap.release()
    timing_results["Field_Video"] = np.median(vid_times)

    print("\n" + "=" * 60)
    print("Smoke Test Summary (Median Latency per frame):")
    for dset, med_ms in timing_results.items():
        print(f"  - {dset:<15}: {med_ms:.1f} ms")
    print(f"Overlays saved to: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    run_smoke_test()
