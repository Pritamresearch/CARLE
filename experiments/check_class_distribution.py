import json
from pathlib import Path
import numpy as np
import pandas as pd

cache_dir = Path("results/phase1/cache")

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

for name, p in datasets.items():
    if not p.exists():
        p = Path(str(p).replace("_full", ""))
    if not p.exists():
        continue
    
    data = np.load(p, mmap_mode="r")["masks"]
    N, H, W = data.shape
    total_pixels = N * H * W
    
    counts = np.zeros(5, dtype=np.int64)
    n_has_c1 = 0
    n_single_class = 0
    n_near_empty = 0 # >= 99% class 1
    
    for i in range(N):
        frame = data[i]
        fc = np.bincount(frame.ravel(), minlength=5)
        counts += fc
        if fc[1] > 0:
            n_has_c1 += 1
        if fc[1] == H * W or np.max(fc[1:]) == H * W:
            n_single_class += 1
        if np.max(fc[1:]) >= 0.99 * H * W:
            n_near_empty += 1
            
    stats.append({
        "dataset": name,
        "N_frames": N,
        "class_1_pct": (counts[1] / total_pixels) * 100.0,
        "class_2_pct": (counts[2] / total_pixels) * 100.0,
        "class_3_pct": (counts[3] / total_pixels) * 100.0,
        "class_4_pct": (counts[4] / total_pixels) * 100.0,
        "has_class1_pct": (n_has_c1 / N) * 100.0,
        "single_class_pct": (n_single_class / N) * 100.0,
        "near_empty_pct": (n_near_empty / N) * 100.0,
    })

# nuScenes Sweeps across scenes
sweep_files = sorted(list((cache_dir / "sweeps").glob("sweeps_*.npz")))
sweep_n = 0
sweep_pixels = 0
sweep_counts = np.zeros(5, dtype=np.int64)
sweep_single = 0
sweep_near_empty = 0
sweep_has_c1 = 0

for sf in sweep_files:
    m = np.load(sf, mmap_mode="r")["masks"]
    N, H, W = m.shape
    sweep_n += N
    sweep_pixels += N * H * W
    for i in range(N):
        f = m[i]
        fc = np.bincount(f.ravel(), minlength=5)
        sweep_counts += fc
        if fc[1] > 0:
            sweep_has_c1 += 1
        if fc[1] == H * W or np.max(fc[1:]) == H * W:
            sweep_single += 1
        if np.max(fc[1:]) >= 0.99 * H * W:
            sweep_near_empty += 1

stats.append({
    "dataset": "nuScenes_Sweeps_12Hz_All",
    "N_frames": sweep_n,
    "class_1_pct": (sweep_counts[1] / sweep_pixels) * 100.0,
    "class_2_pct": (sweep_counts[2] / sweep_pixels) * 100.0,
    "class_3_pct": (sweep_counts[3] / sweep_pixels) * 100.0,
    "class_4_pct": (sweep_counts[4] / sweep_pixels) * 100.0,
    "has_class1_pct": (sweep_has_c1 / sweep_n) * 100.0,
    "single_class_pct": (sweep_single / sweep_n) * 100.0,
    "near_empty_pct": (sweep_near_empty / sweep_n) * 100.0,
})

df = pd.DataFrame(stats)
print(df.to_string(index=False))
df.to_csv("results/phase1/class_distribution_audit.csv", index=False)

print("\n--- Auditing Identical Pairs on nuScenes ---")

# 1. Keyframes identical pairs
kf_m = np.load(cache_dir / "nuscenes_keyframes_2hz.npz", mmap_mode="r")["masks"]
with open(cache_dir / "nuscenes_keyframes_2hz_meta.json", "r") as f:
    kf_meta = json.load(f)

# Group by scene and camera to find consecutive pairs
from collections import defaultdict
by_cam_scene = defaultdict(list)
for i, m in enumerate(kf_meta):
    by_cam_scene[(m["scene"], m["camera"])].append((i, m["timestamp"]))

kf_identical = []
kf_total_pairs = 0
for (sc, cam), items in by_cam_scene.items():
    items.sort(key=lambda x: x[1])
    for j in range(1, len(items)):
        idx_prev = items[j-1][0]
        idx_curr = items[j][0]
        kf_total_pairs += 1
        if np.array_equal(kf_m[idx_curr], kf_m[idx_prev]):
            kf_identical.append(idx_curr)

kf_ident_single = 0
kf_ident_near_empty = 0
H, W = kf_m[0].shape
for idx in kf_identical:
    f = kf_m[idx]
    fc = np.bincount(f.ravel(), minlength=5)
    if np.max(fc[1:]) == H * W:
        kf_ident_single += 1
    if np.max(fc[1:]) >= 0.99 * H * W:
        kf_ident_near_empty += 1

print(f"nuScenes Keyframes: Total pairs = {kf_total_pairs}, Identical = {len(kf_identical)} ({(len(kf_identical)/kf_total_pairs)*100:.2f}%)")
print(f"  Of identical pairs: Single-Class (100% background) = {kf_ident_single} ({(kf_ident_single/len(kf_identical))*100:.2f}%)")
print(f"  Of identical pairs: Near-Empty (>=99% background)  = {kf_ident_near_empty} ({(kf_ident_near_empty/len(kf_identical))*100:.2f}%)")

# 2. Sweeps identical pairs
sweep_total_pairs = 0
sweep_identical = []
sweep_ident_single = 0
sweep_ident_near_empty = 0

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
            idx_prev = items[j-1][0]
            idx_curr = items[j][0]
            sweep_total_pairs += 1
            if np.array_equal(m[idx_curr], m[idx_prev]):
                sweep_identical.append((sf.name, idx_curr))
                fc = np.bincount(m[idx_curr].ravel(), minlength=5)
                if np.max(fc[1:]) == H * W:
                    sweep_ident_single += 1
                if np.max(fc[1:]) >= 0.99 * H * W:
                    sweep_ident_near_empty += 1

print(f"nuScenes Sweeps: Total pairs = {sweep_total_pairs}, Identical = {len(sweep_identical)} ({(len(sweep_identical)/sweep_total_pairs)*100:.2f}%)")
print(f"  Of identical pairs: Single-Class (100% background) = {sweep_ident_single} ({(sweep_ident_single/len(sweep_identical))*100:.2f}%)")
print(f"  Of identical pairs: Near-Empty (>=99% background)  = {sweep_ident_near_empty} ({(sweep_ident_near_empty/len(sweep_identical))*100:.2f}%)")

