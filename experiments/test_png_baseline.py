"""
PNG Baseline Benchmark (OpenCV level 9) vs CARLE on nuScenes 4-class label maps.
"""

import sys
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# Class mapping: COCO -> CARLE 4-class
COCO_TO_CARLE = np.zeros(80, dtype=np.uint8)
COCO_TO_CARLE[0] = 2                         # Pedestrian
COCO_TO_CARLE[[2, 3, 5, 7]] = 3             # Vehicle
COCO_TO_CARLE[[15, 16, 17, 18, 19, 20]] = 4 # Animal / Other

def rle_encode_carle(flat: np.ndarray):
    """Class-preserving CARLE triplets (start, length, class) for non-zero runs."""
    triplets = []
    i = 0
    N = len(flat)
    while i < N:
        if flat[i] == 0:
            i += 1
            continue
        c = int(flat[i])
        start = i
        j = i + 1
        while j < N and flat[j] == c:
            j += 1
        length = j - start
        triplets.append((start, length, c))
        i = j
    return triplets

def benchmark_png_and_carle(nusc_samples_dir: str, num_frames: int = 50):
    model = YOLO("yolo11n-seg.pt")
    
    # Collect sample images
    img_paths = sorted(Path(nusc_samples_dir).glob("CAM_FRONT/*.jpg"))[:num_frames]
    if not img_paths:
        img_paths = sorted(Path(nusc_samples_dir).glob("**/*.jpg"))[:num_frames]
    
    print(f"Found {len(img_paths)} frames for PNG baseline benchmark.")
    
    png_bytes_list = []
    png_times_list = []
    carle_legacy_bytes = []
    carle_canonical_bytes = []
    
    prev_mask = None
    
    for p in img_paths:
        res = model.predict(str(p), imgsz=640, device="cpu", verbose=False)[0]
        H, W = res.orig_shape
        mask = np.zeros((H, W), dtype=np.uint8)
        if res.masks is not None:
            for seg_box, seg_poly in zip(res.boxes, res.masks.xy):
                cls_id = int(seg_box.cls[0].item())
                mapped = COCO_TO_CARLE[cls_id]
                if mapped > 0 and len(seg_poly) > 0:
                    poly = np.array(seg_poly, dtype=np.int32)
                    cv2.fillPoly(mask, [poly], int(mapped))
        
        # 1. OpenCV PNG level 9
        t0 = time.perf_counter()
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
        success, enc = cv2.imencode(".png", mask, encode_params)
        t_png = (time.perf_counter() - t0) * 1000.0
        
        png_size = len(enc)
        png_bytes_list.append(png_size)
        png_times_list.append(t_png)
        
        # 2. CARLE differential
        if prev_mask is None:
            prev_mask = mask
            # For first frame, CARLE transmits full frame RLE or blank diff
            diff_canon = mask.copy()
            diff_xor = mask.copy()
        else:
            # Canonical: F_{t+1} where changed, else 0
            diff_canon = np.where(mask != prev_mask, mask, 0).astype(np.uint8)
            # Legacy: XOR
            diff_xor = np.bitwise_xor(mask, prev_mask)
            prev_mask = mask
            
        triplets_canon = rle_encode_carle(diff_canon.ravel())
        canon_bytes = len(triplets_canon) * 7 # 4+2+1 bytes
        carle_canonical_bytes.append(canon_bytes)
        
        # Legacy XOR RLE (including zeros, 6 bytes each)
        if len(diff_xor) == 0:
            legacy_bytes = 0
        else:
            flat = diff_xor.ravel()
            ch_idx = np.flatnonzero(np.diff(flat)) + 1
            starts = np.concatenate([[0], ch_idx])
            lengths = np.diff(np.concatenate([starts, [len(flat)]]))
            legacy_bytes = len(starts) * 6
        carle_legacy_bytes.append(legacy_bytes)

    H, W = 900, 1600
    raw_8bit = H * W
    raw_2bit = H * W * 2 / 8.0 # 360,000 bytes
    
    mean_png_b = np.mean(png_bytes_list)
    mean_png_bpf = mean_png_b * 8.0 / 1000.0
    mean_png_lat = np.mean(png_times_list)
    
    mean_canon_b = np.mean(carle_canonical_bytes[1:]) if len(carle_canonical_bytes) > 1 else np.mean(carle_canonical_bytes)
    mean_canon_bpf = mean_canon_b * 8.0 / 1000.0
    
    mean_legacy_b = np.mean(carle_legacy_bytes[1:]) if len(carle_legacy_bytes) > 1 else np.mean(carle_legacy_bytes)
    mean_legacy_bpf = mean_legacy_b * 8.0 / 1000.0
    
    print("\n================== PNG BASELINE VS CARLE ==================")
    print(f"Frames evaluated: {len(img_paths)}")
    print(f"Raw 8-bit size:   {raw_8bit:,} bytes ({raw_8bit*8/1000:,.1f} Kb)")
    print(f"Raw 2-bit size:   {raw_2bit:,} bytes ({raw_2bit*8/1000:,.1f} Kb)")
    print(f"PNG (Level 9):    {mean_png_b:,.1f} bytes/frame | {mean_png_bpf:,.2f} Kb/frame | CR_raw8: {raw_8bit/mean_png_b:.1f}x | CR_raw2: {raw_2bit/mean_png_b:.1f}x | Latency: {mean_png_lat:.2f} ms")
    print(f"CARLE Canonical:  {mean_canon_b:,.1f} bytes/frame | {mean_canon_bpf:,.2f} Kb/frame | CR_raw8: {raw_8bit/mean_canon_b:.1f}x | CR_raw2: {raw_2bit/mean_canon_b:.1f}x (real 7B packet)")
    print(f"CARLE Legacy XOR: {mean_legacy_b:,.1f} bytes/frame | {mean_legacy_bpf:,.2f} Kb/frame | CR_raw8: {raw_8bit/mean_legacy_b:.1f}x | CR_raw2: {raw_2bit/mean_legacy_b:.1f}x (len*6 accounting)")
    print("===========================================================\n")
    
    df_out = pd.DataFrame({
        "frame": [p.name for p in img_paths],
        "png_bytes": png_bytes_list,
        "png_latency_ms": png_times_list,
        "carle_canonical_bytes": carle_canonical_bytes,
        "carle_legacy_bytes": carle_legacy_bytes
    })
    df_out.to_csv("results/phase0/png_baseline.csv", index=False)
    print("Saved results to results/phase0/png_baseline.csv")

if __name__ == "__main__":
    benchmark_png_and_carle("data/nuscenes/samples", num_frames=50)
