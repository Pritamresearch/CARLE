"""
Fast, memory-safe audit of PIDNet-S Grounded Semantic Caches (K=4 and K=8).
Computes:
  - Class shares
  - % of frames containing road
  - % of frames that are single-class (100% solid background)
  - Mean delta_pix (fraction of pixels changed between consecutive frames)
"""

from pathlib import Path
import numpy as np
import pandas as pd

CACHE_DIR = Path("results/phase1/cache_pidnet")
SWEEPS_DIR = CACHE_DIR / "sweeps"

def audit_dataset_k4(name, p):
    m = np.load(p, mmap_mode="r")["masks"]
    N, H, W = m.shape
    tot_pixels = N * H * W
    frame_pixels = H * W
    
    counts = np.zeros(5, dtype=np.int64)
    n_single = 0
    has_road = 0
    deltas = []
    
    for i in range(N):
        f = m[i]
        fc = np.bincount(f.ravel(), minlength=5)
        counts += fc
        if fc[1] > 0: has_road += 1
        if np.max(fc[1:]) == frame_pixels: n_single += 1
        if i > 0:
            deltas.append(np.mean(m[i] != m[i-1]))
            
    mean_dp = np.mean(deltas) if len(deltas) > 0 else 0.0
    
    return {
        "dataset": name,
        "N_frames": N,
        "Road_c1_pct": round((counts[1] / tot_pixels) * 100.0, 2),
        "Vehicle_c2_pct": round((counts[2] / tot_pixels) * 100.0, 2),
        "VRU_c3_pct": round((counts[3] / tot_pixels) * 100.0, 2),
        "Other_c4_pct": round((counts[4] / tot_pixels) * 100.0, 2),
        "has_road_pct": round((has_road / N) * 100.0, 2),
        "single_class_pct": round((n_single / N) * 100.0, 2),
        "mean_delta_pix": round(mean_dp, 4)
    }

def audit_dataset_k8(name, p):
    m = np.load(p, mmap_mode="r")["masks"]
    N, H, W = m.shape
    tot_pixels = N * H * W
    frame_pixels = H * W
    
    counts = np.zeros(9, dtype=np.int64)
    n_single = 0
    has_road = 0
    deltas = []
    
    for i in range(N):
        f = m[i]
        fc = np.bincount(f.ravel(), minlength=9)
        counts += fc
        if fc[1] > 0: has_road += 1
        if np.max(fc[1:]) == frame_pixels: n_single += 1
        if i > 0:
            deltas.append(np.mean(m[i] != m[i-1]))
            
    mean_dp = np.mean(deltas) if len(deltas) > 0 else 0.0
    
    return {
        "dataset": name,
        "N_frames": N,
        "Road_c1": round((counts[1] / tot_pixels) * 100.0, 2),
        "Sidewalk_c2": round((counts[2] / tot_pixels) * 100.0, 2),
        "Static_c3": round((counts[3] / tot_pixels) * 100.0, 2),
        "Veg_c4": round((counts[4] / tot_pixels) * 100.0, 2),
        "Sky_c5": round((counts[5] / tot_pixels) * 100.0, 2),
        "Vehicle_c6": round((counts[6] / tot_pixels) * 100.0, 2),
        "TwoWheel_c7": round((counts[7] / tot_pixels) * 100.0, 2),
        "Person_c8": round((counts[8] / tot_pixels) * 100.0, 2),
        "has_road_pct": round((has_road / N) * 100.0, 2),
        "single_class_pct": round((n_single / N) * 100.0, 2),
        "mean_delta_pix": round(mean_dp, 4)
    }

print("Auditing K=4 datasets...")
k4_files = {
    "nuScenes_Keyframes_2Hz": CACHE_DIR / "nuscenes_keyframes_2hz_k4.npz",
    "Field_Bhubaneswar_Night_Native": CACHE_DIR / "field_bhubaneswar_night_native_k4.npz",
    "Field_Dhaka_Day_Native": CACHE_DIR / "field_dhaka_day_native_k4.npz",
    "Field_Dhaka_Night_Native": CACHE_DIR / "field_dhaka_night_native_k4.npz",
    "Field_Bolpur_Native": CACHE_DIR / "field_india_bolpur_native_k4.npz",
    "Field_Malaysia_Native": CACHE_DIR / "field_malaysia_native_k4.npz",
    "Field_Sikkim_Native": CACHE_DIR / "field_sikkim_native_k4.npz",
}

stats_k4 = []
for name, p in k4_files.items():
    if p.exists():
        stats_k4.append(audit_dataset_k4(name, p))

# nuScenes Sweeps K=4
sw_k4_files = sorted(list(SWEEPS_DIR.glob("sweeps_*_k4.npz")))
sw_counts = np.zeros(5, dtype=np.int64)
sw_single = 0
sw_has_road = 0
sw_N = 0
sw_tot = 0
sw_deltas = []

for sf in sw_k4_files:
    m = np.load(sf, mmap_mode="r")["masks"]
    N, H, W = m.shape
    sw_N += N
    sw_tot += N * H * W
    for i in range(N):
        fc = np.bincount(m[i].ravel(), minlength=5)
        sw_counts += fc
        if fc[1] > 0: sw_has_road += 1
        if np.max(fc[1:]) == H * W: sw_single += 1
        if i > 0: sw_deltas.append(np.mean(m[i] != m[i-1]))

stats_k4.append({
    "dataset": "nuScenes_Sweeps_12Hz_All6Cams",
    "N_frames": sw_N,
    "Road_c1_pct": round((sw_counts[1] / sw_tot) * 100.0, 2),
    "Vehicle_c2_pct": round((sw_counts[2] / sw_tot) * 100.0, 2),
    "VRU_c3_pct": round((sw_counts[3] / sw_tot) * 100.0, 2),
    "Other_c4_pct": round((sw_counts[4] / sw_tot) * 100.0, 2),
    "has_road_pct": round((sw_has_road / sw_N) * 100.0, 2),
    "single_class_pct": round((sw_single / sw_N) * 100.0, 2),
    "mean_delta_pix": round(np.mean(sw_deltas), 4)
})

df_k4 = pd.DataFrame(stats_k4)
print("\n=== PIDNET K=4 AUDIT RESULTS ===")
print(df_k4.to_string(index=False))
df_k4.to_csv("results/phase1/pidnet_k4_audit.csv", index=False)

print("\nAuditing K=8 datasets...")
k8_files = {
    "nuScenes_Keyframes_2Hz": CACHE_DIR / "nuscenes_keyframes_2hz_k8.npz",
    "Field_Bhubaneswar_Night_Native": CACHE_DIR / "field_bhubaneswar_night_native_k8.npz",
    "Field_Dhaka_Day_Native": CACHE_DIR / "field_dhaka_day_native_k8.npz",
    "Field_Dhaka_Night_Native": CACHE_DIR / "field_dhaka_night_native_k8.npz",
    "Field_Bolpur_Native": CACHE_DIR / "field_india_bolpur_native_k8.npz",
    "Field_Malaysia_Native": CACHE_DIR / "field_malaysia_native_k8.npz",
    "Field_Sikkim_Native": CACHE_DIR / "field_sikkim_native_k8.npz",
}

stats_k8 = []
for name, p in k8_files.items():
    if p.exists():
        stats_k8.append(audit_dataset_k8(name, p))

# nuScenes Sweeps K=8
sw_k8_files = sorted(list(SWEEPS_DIR.glob("sweeps_*_k8.npz")))
sw8_counts = np.zeros(9, dtype=np.int64)
sw8_single = 0
sw8_has_road = 0
sw8_N = 0
sw8_tot = 0
sw8_deltas = []

for sf in sw_k8_files:
    m = np.load(sf, mmap_mode="r")["masks"]
    N, H, W = m.shape
    sw8_N += N
    sw8_tot += N * H * W
    for i in range(N):
        fc = np.bincount(m[i].ravel(), minlength=9)
        sw8_counts += fc
        if fc[1] > 0: sw8_has_road += 1
        if np.max(fc[1:]) == H * W: sw8_single += 1
        if i > 0: sw8_deltas.append(np.mean(m[i] != m[i-1]))

stats_k8.append({
    "dataset": "nuScenes_Sweeps_12Hz_All6Cams",
    "N_frames": sw8_N,
    "Road_c1": round((sw8_counts[1] / sw8_tot) * 100.0, 2),
    "Sidewalk_c2": round((sw8_counts[2] / sw8_tot) * 100.0, 2),
    "Static_c3": round((sw8_counts[3] / sw8_tot) * 100.0, 2),
    "Veg_c4": round((sw8_counts[4] / sw8_tot) * 100.0, 2),
    "Sky_c5": round((sw8_counts[5] / sw8_tot) * 100.0, 2),
    "Vehicle_c6": round((sw8_counts[6] / sw8_tot) * 100.0, 2),
    "TwoWheel_c7": round((sw8_counts[7] / sw8_tot) * 100.0, 2),
    "Person_c8": round((sw8_counts[8] / sw8_tot) * 100.0, 2),
    "has_road_pct": round((sw8_has_road / sw8_N) * 100.0, 2),
    "single_class_pct": round((sw8_single / sw8_N) * 100.0, 2),
    "mean_delta_pix": round(np.mean(sw8_deltas), 4)
})

df_k8 = pd.DataFrame(stats_k8)
print("\n=== PIDNET K=8 AUDIT RESULTS ===")
print(df_k8.to_string(index=False))
df_k8.to_csv("results/phase1/pidnet_k8_audit.csv", index=False)
