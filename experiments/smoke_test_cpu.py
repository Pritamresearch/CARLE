"""
CPU-only smoke test with fixed thread count (4 threads).
"""
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

import time
from pathlib import Path
import cv2
import numpy as np
import yaml
import torch
from ultralytics import YOLO

torch.set_num_threads(4)

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "paths.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

OUTPUT_DIR = Path(__file__).parent.parent / "results" / "setup" / "cpu"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COCO_TO_CARLE = np.zeros(80, dtype=np.uint8)
COCO_TO_CARLE[0] = 2
COCO_TO_CARLE[[2, 3, 5, 7]] = 3
COCO_TO_CARLE[[14, 15, 16, 17, 18, 19, 20]] = 4

PALETTE = {
    0: (0, 0, 0),        # Background
    1: (128, 64, 128),   # Road
    2: (0, 255, 0),      # Pedestrian
    3: (255, 128, 0),    # Vehicle
    4: (0, 165, 255),    # Animal
}

def create_overlay(img_bgr, priority_mask, alpha=0.45):
    H, W = priority_mask.shape
    if img_bgr.shape[:2] != (H, W):
        img_bgr = cv2.resize(img_bgr, (W, H))

    overlay = np.zeros_like(img_bgr)
    for cls_id, color in PALETTE.items():
        if cls_id == 0:
            continue
        overlay[priority_mask == cls_id] = color

    blended = cv2.addWeighted(img_bgr, 1 - alpha, overlay, alpha, 0)
    return blended

def run():
    print(f"=== CPU-Only Smoke Test (Threads: {torch.get_num_threads()}) ===")
    model = YOLO("yolo11n-seg.pt")

    # Warmup
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    for _ in range(10):
        model.predict(dummy, device="cpu", imgsz=640, verbose=False)

    results_timing = {}

    # nuScenes
    nusc_dir = Path(config["datasets"]["nuscenes"]["root"]) / "samples" / "CAM_FRONT"
    nusc_files = sorted(list(nusc_dir.glob("*.jpg")))[:5]
    nusc_times = []
    for f in nusc_files:
        img = cv2.imread(str(f))
        H, W = img.shape[:2]
        t0 = time.perf_counter_ns()
        res = model.predict(img, device="cpu", imgsz=640, verbose=False)[0]
        mask = np.zeros((H, W), dtype=np.uint8)
        if res.masks is not None:
            seg_masks = res.masks.data.cpu().numpy()
            classes = res.boxes.cls.cpu().numpy().astype(int)
            priorities = COCO_TO_CARLE[np.clip(classes, 0, 79)]
            for s_idx, p in enumerate(priorities):
                if p > 0:
                    m = cv2.resize(seg_masks[s_idx], (W, H), interpolation=cv2.INTER_NEAREST)
                    mask[m > 0.5] = p
        dt_ms = (time.perf_counter_ns() - t0) / 1e6
        nusc_times.append(dt_ms)
    results_timing["nuScenes (CPU)"] = np.median(nusc_times)

    # Cityscapes
    cs_dir = Path(config["datasets"]["cityscapes"]["left_img8bit"]) / "val" / "frankfurt"
    cs_files = sorted(list(cs_dir.glob("*.png")))[:5]
    cs_times = []
    for f in cs_files:
        img = cv2.imread(str(f))
        H, W = img.shape[:2]
        t0 = time.perf_counter_ns()
        res = model.predict(img, device="cpu", imgsz=640, verbose=False)[0]
        mask = np.zeros((H, W), dtype=np.uint8)
        if res.masks is not None:
            seg_masks = res.masks.data.cpu().numpy()
            classes = res.boxes.cls.cpu().numpy().astype(int)
            priorities = COCO_TO_CARLE[np.clip(classes, 0, 79)]
            for s_idx, p in enumerate(priorities):
                if p > 0:
                    m = cv2.resize(seg_masks[s_idx], (W, H), interpolation=cv2.INTER_NEAREST)
                    mask[m > 0.5] = p
        dt_ms = (time.perf_counter_ns() - t0) / 1e6
        cs_times.append(dt_ms)
    results_timing["Cityscapes (CPU)"] = np.median(cs_times)

    # IDD
    idd_dir = Path(config["datasets"]["idd"]["left_img8bit"]) / "val"
    idd_files = (sorted(list(idd_dir.rglob("*.jpg"))) + sorted(list(idd_dir.rglob("*.png"))))[:5]
    idd_times = []
    for f in idd_files:
        img = cv2.imread(str(f))
        H, W = img.shape[:2]
        t0 = time.perf_counter_ns()
        res = model.predict(img, device="cpu", imgsz=640, verbose=False)[0]
        mask = np.zeros((H, W), dtype=np.uint8)
        if res.masks is not None:
            seg_masks = res.masks.data.cpu().numpy()
            classes = res.boxes.cls.cpu().numpy().astype(int)
            priorities = COCO_TO_CARLE[np.clip(classes, 0, 79)]
            for s_idx, p in enumerate(priorities):
                if p > 0:
                    m = cv2.resize(seg_masks[s_idx], (W, H), interpolation=cv2.INTER_NEAREST)
                    mask[m > 0.5] = p
        dt_ms = (time.perf_counter_ns() - t0) / 1e6
        idd_times.append(dt_ms)
    results_timing["IDD (CPU)"] = np.median(idd_times)

    # Field Video
    vid_path = Path(config["datasets"]["field_videos"]["root"]) / "sikim.mp4"
    cap = cv2.VideoCapture(str(vid_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = [int(x) for x in np.linspace(10, total_frames - 10, 5)]
    vid_times = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, img = cap.read()
        if not ret: continue
        H, W = img.shape[:2]
        t0 = time.perf_counter_ns()
        res = model.predict(img, device="cpu", imgsz=640, verbose=False)[0]
        mask = np.zeros((H, W), dtype=np.uint8)
        if res.masks is not None:
            seg_masks = res.masks.data.cpu().numpy()
            classes = res.boxes.cls.cpu().numpy().astype(int)
            priorities = COCO_TO_CARLE[np.clip(classes, 0, 79)]
            for s_idx, p in enumerate(priorities):
                if p > 0:
                    m = cv2.resize(seg_masks[s_idx], (W, H), interpolation=cv2.INTER_NEAREST)
                    mask[m > 0.5] = p
        dt_ms = (time.perf_counter_ns() - t0) / 1e6
        vid_times.append(dt_ms)
    cap.release()
    results_timing["Field_Video (CPU)"] = np.median(vid_times)

    print("\nCPU-Only Median Latency (Warmup: 10, Threads: 4):")
    for k, v in results_timing.items():
        print(f"  {k:<22}: {v:.1f} ms")

if __name__ == "__main__":
    run()
