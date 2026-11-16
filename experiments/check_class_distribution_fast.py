import json
import time
import io
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit
from PIL import Image

cache_dir = Path("results/phase1/cache")

@njit(fastmath=True)
def audit_masks_fast(masks: np.ndarray):
    N, H, W = masks.shape
    total_pix = np.int64(N) * np.int64(H) * np.int64(W)
    frame_pixels = H * W
    near_thresh = int(0.99 * frame_pixels)
    
    counts = np.zeros(5, dtype=np.int64)
    n_has_c1 = 0
    n_single = 0
    n_near_empty = 0
    
    for i in range(N):
        f_counts = np.zeros(5, dtype=np.int64)
        for r in range(H):
            for c in range(W):
                val = masks[i, r, c]
                if val <= 4:
                    f_counts[val] += 1
                    
        for s in range(5):
            counts[s] += f_counts[s]
            
        if f_counts[1] > 0:
            n_has_c1 += 1
            
        # check max class
        max_c = 0
        for s in range(1, 5):
            if f_counts[s] > max_c:
                max_c = f_counts[s]
                
        if max_c == frame_pixels:
            n_single += 1
        if max_c >= near_thresh:
            n_near_empty += 1
            
    return counts, n_has_c1, n_single, n_near_empty, N, total_pix

datasets = {
    "nuScenes_Keyframes_2Hz": cache_dir / "nuscenes_keyframes_2hz.npz",
    "Field_Dhaka_Day_Native": cache_dir / "field_dhaka_day_native_full.npz",
    "Field_Dhaka_Night_Native": cache_dir / "field_dhaka_night_native_full.npz",
    "Field_Bolpur_Native": cache_dir / "field_india_bolpur_native_full.npz",
    "Field_Malaysia_Native": cache_dir / "field_malaysia_native_full.npz",
    "Field_Sikkim_Native": cache_dir / "field_sikkim_native_full.npz",
    "Field_Bhubaneswar_Night_Native": cache_dir / "field_bhubaneswar_night_native_full.npz",
}

stats = []

print("Auditing datasets...")
for name, p in datasets.items():
    if not p.exists():
        p = Path(str(p).replace("_full", ""))
    if not p.exists(): continue
    
    data = np.load(p, mmap_mode="r")["masks"]
    counts, n_has_c1, n_single, n_near, N, tot_pix = audit_masks_fast(data)
    
    stats.append({
        "dataset": name,
        "N_frames": N,
        "class_1_pct": (counts[1] / tot_pix) * 100.0,
        "class_2_pct": (counts[2] / tot_pix) * 100.0,
        "class_3_pct": (counts[3] / tot_pix) * 100.0,
        "class_4_pct": (counts[4] / tot_pix) * 100.0,
        "has_class1_pct": (n_has_c1 / N) * 100.0,
        "single_class_pct": (n_single / N) * 100.0,
        "near_empty_pct": (n_near / N) * 100.0,
    })

# Sweeps
sweep_files = sorted(list((cache_dir / "sweeps").glob("sweeps_*.npz")))
sw_counts = np.zeros(5, dtype=np.int64)
sw_has_c1 = 0
sw_single = 0
sw_near = 0
sw_N = 0
sw_tot_pix = 0

for sf in sweep_files:
    m = np.load(sf, mmap_mode="r")["masks"]
    c, h1, s, ne, N, tp = audit_masks_fast(m)
    sw_counts += c
    sw_has_c1 += h1
    sw_single += s
    sw_near += ne
    sw_N += N
    sw_tot_pix += tp

stats.append({
    "dataset": "nuScenes_Sweeps_12Hz_All",
    "N_frames": sw_N,
    "class_1_pct": (sw_counts[1] / sw_tot_pix) * 100.0,
    "class_2_pct": (sw_counts[2] / sw_tot_pix) * 100.0,
    "class_3_pct": (sw_counts[3] / sw_tot_pix) * 100.0,
    "class_4_pct": (sw_counts[4] / sw_tot_pix) * 100.0,
    "has_class1_pct": (sw_has_c1 / sw_N) * 100.0,
    "single_class_pct": (sw_single / sw_N) * 100.0,
    "near_empty_pct": (sw_near / sw_N) * 100.0,
})

df = pd.DataFrame(stats)
print("\n=== CLASS DISTRIBUTION AND FRAME COMPLEXITY AUDIT ===")
print(df.to_string(index=False))
df.to_csv("results/phase1/class_distribution_audit.csv", index=False)

# -------------------------------------------------------------
# Audit Identical Pairs in nuScenes Keyframes and Sweeps
# -------------------------------------------------------------
print("\n=== IDENTICAL PAIRS AUDIT ===")

# Keyframes
kf_data = np.load(cache_dir / "nuscenes_keyframes_2hz.npz", mmap_mode="r")["masks"]
with open(cache_dir / "nuscenes_keyframes_2hz_meta.json", "r") as f:
    kf_meta = json.load(f)

from collections import defaultdict
by_cam_scene = defaultdict(list)
for i, m in enumerate(kf_meta):
    by_cam_scene[(m["scene"], m["camera"])].append((i, m["timestamp"]))

kf_ident_idx = []
kf_pairs_count = 0
for (sc, cam), items in by_cam_scene.items():
    items.sort(key=lambda x: x[1])
    for j in range(1, len(items)):
        kf_pairs_count += 1
        p_idx = items[j-1][0]
        c_idx = items[j][0]
        if np.array_equal(kf_data[c_idx], kf_data[p_idx]):
            kf_ident_idx.append(c_idx)

kf_ident_single = 0
kf_ident_near = 0
H, W = kf_data[0].shape
for idx in kf_ident_idx:
    f = kf_data[idx]
    fc = np.bincount(f.ravel(), minlength=5)
    max_c = np.max(fc[1:])
    if max_c == H * W:
        kf_ident_single += 1
    if max_c >= 0.99 * H * W:
        kf_ident_near += 1

print(f"nuScenes Keyframes (2 Hz):")
print(f"  Total consecutive pairs = {kf_pairs_count}")
print(f"  Identical pairs = {len(kf_ident_idx)} ({(len(kf_ident_idx)/kf_pairs_count)*100:.2f}%)")
print(f"  Single-class identical pairs (100% background) = {kf_ident_single} ({(kf_ident_single/len(kf_ident_idx))*100:.2f}%)")
print(f"  Near-empty identical pairs (>=99% background)  = {kf_ident_near} ({(kf_ident_near/len(kf_ident_idx))*100:.2f}%)")
print(f"  Active foreground identical pairs (<99% background) = {len(kf_ident_idx) - kf_ident_near} ({((len(kf_ident_idx) - kf_ident_near)/len(kf_ident_idx))*100:.2f}%)")

# Sweeps
sweep_pairs_count = 0
sweep_ident_idx = []
sweep_ident_single = 0
sweep_ident_near = 0

for sf in sweep_files:
    meta_p = Path(str(sf).replace(".npz", "_meta.json"))
    if not meta_p.exists(): continue
    with open(meta_p, "r") as f:
        meta = json.load(f)
    m = np.load(sf, mmap_mode="r")["masks"]
    H, W = m[0].shape
    
    by_cam = defaultdict(list)
    for i, item in enumerate(meta):
        by_cam[item["camera"]].append((i, item["timestamp"]))
        
    for cam, items in by_cam.items():
        items.sort(key=lambda x: x[1])
        for j in range(1, len(items)):
            sweep_pairs_count += 1
            p_idx = items[j-1][0]
            c_idx = items[j][0]
            if np.array_equal(m[c_idx], m[p_idx]):
                sweep_ident_idx.append((sf.name, c_idx))
                fc = np.bincount(m[c_idx].ravel(), minlength=5)
                max_c = np.max(fc[1:])
                if max_c == H * W:
                    sweep_ident_single += 1
                if max_c >= 0.99 * H * W:
                    sweep_ident_near += 1

print(f"\nnuScenes Sweeps (~12 Hz):")
print(f"  Total consecutive pairs = {sweep_pairs_count}")
print(f"  Identical pairs = {len(sweep_ident_idx)} ({(len(sweep_ident_idx)/sweep_pairs_count)*100:.2f}%)")
print(f"  Single-class identical pairs (100% background) = {sweep_ident_single} ({(sweep_ident_single/len(sweep_ident_idx))*100:.2f}%)")
print(f"  Near-empty identical pairs (>=99% background)  = {sweep_ident_near} ({(sweep_ident_near/len(sweep_ident_idx))*100:.2f}%)")
print(f"  Active foreground identical pairs (<99% background) = {len(sweep_ident_idx) - sweep_ident_near} ({((len(sweep_ident_idx) - sweep_ident_near)/len(sweep_ident_idx))*100:.2f}%)")

# -------------------------------------------------------------
# WebP Method 4 vs Level 6 Byte-Identity Verification
# -------------------------------------------------------------
print("\n=== WEBP METHOD 4 vs METHOD 6 BYTE-IDENTITY TEST ===")
test_frames = [kf_data[0], kf_data[10], kf_data[50], kf_data[100]]
all_equal = True
for idx, tf in enumerate(test_frames):
    img = Image.fromarray(tf)
    b4 = io.BytesIO()
    img.save(b4, format="WEBP", lossless=True, quality=100, method=4)
    bytes_m4 = b4.getvalue()
    
    b6 = io.BytesIO()
    img.save(b6, format="WEBP", lossless=True, quality=100, method=6)
    bytes_m6 = b6.getvalue()
    
    is_eq = (bytes_m4 == bytes_m6)
    all_equal = all_equal and is_eq
    print(f"Frame {idx}: len(m4)={len(bytes_m4)}, len(m6)={len(bytes_m6)}, Exact Match={is_eq}")

print(f"WebP method=4 == method=6 Byte-Exact across test frames: {all_equal}")
