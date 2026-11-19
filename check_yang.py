import pandas as pd
for k in (4, 8):
    d = pd.read_csv(f"results/phase1/p1b_all_frames_raw_k{k}.csv")
    d = d[d["dataset"] == "nuScenes_Sweeps_12Hz"]
    print(f"\nK={k}, frames={len(d)}")
    print(d.filter(regex="bits").describe().T[["mean", "min", "50%", "max"]])
    print("yang_bits <= 48 (fallback):", (d["yang_bits"] <= 48).sum())
