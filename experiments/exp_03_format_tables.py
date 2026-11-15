import pandas as pd
import numpy as np

df = pd.read_csv("results/phase1/results_1a.csv")

def format_table(sub_df, title):
    out = []
    out.append(f"### {title}\n")
    out.append("| Codec | Lossless | Mean bpf (Kb) | Median (Kb) | 95% Bootstrap CI (Kb) | CR (vs 2-bit raw) | CR (vs PNG) | Enc Mean (ms) | Enc p95 (ms) | Dec Mean (ms) |")
    out.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for _, row in sub_df.iterrows():
        codec = row['codec']
        lossless = "PASS" if row['lossless_passed'] else "FAIL"
        mean_bpf = f"{row['mean_bpf']:.2f}"
        median_bpf = f"{row['median_bpf']:.2f}"
        ci = f"[{row['ci_low_bpf']:.2f}, {row['ci_high_bpf']:.2f}]"
        cr_raw2 = f"{row['sum_cr_raw2']:.1f}x"
        cr_png = f"{row['sum_cr_png']:.2f}x"
        enc_mean = f"{row['mean_enc_ms']:.2f}"
        enc_p95 = f"{row['p95_enc_ms']:.2f}"
        dec_mean = f"{row['mean_dec_ms']:.2f}"
        out.append(f"| `{codec}` | {lossless} | {mean_bpf} | {median_bpf} | {ci} | {cr_raw2} | {cr_png} | {enc_mean} | {enc_p95} | {dec_mean} |")
    out.append("")
    return "\n".join(out)

# 1. nuScenes Keyframes 2 Hz
sub_kf = df[df['dataset'] == 'nuScenes_Keyframes_2Hz']
print(format_table(sub_kf, "nuScenes Keyframes (2 Hz) - Resolution: 900x1600 (Raw 2-bit: 360.0 KB/frame)"))

# 2. nuScenes Sweeps 12 Hz
sub_sw = df[df['dataset'] == 'nuScenes_Sweeps_12Hz']
print(format_table(sub_sw, "nuScenes Sweeps (~12 Hz) - Resolution: 900x1600 (Raw 2-bit: 360.0 KB/frame)"))

# 3. Aggregate Field Videos - Native fps
field_native = df[df['dataset'].str.startswith('Field_') & df['dataset'].str.endswith('_native')]
# Group by codec
print("### Field Test Videos (Native fps) - Summary across all 6 clips")
grouped_native = field_native.groupby('codec').agg({
    'mean_bpf': 'mean',
    'median_bpf': 'mean',
    'ci_low_bpf': 'mean',
    'ci_high_bpf': 'mean',
    'sum_cr_raw2': 'mean',
    'sum_cr_png': 'mean',
    'mean_enc_ms': 'mean',
    'p95_enc_ms': 'mean',
    'mean_dec_ms': 'mean',
    'lossless_passed': 'all'
}).loc[sub_kf['codec'].values]
grouped_native = grouped_native.reset_index()
print(format_table(grouped_native, "Field Test Videos (Native fps, Average across 6 clips)"))

# 4. Aggregate Field Videos - 10 Hz subsampled
field_10hz = df[df['dataset'].str.startswith('Field_') & df['dataset'].str.endswith('_10hz')]
print("### Field Test Videos (10 Hz subsampled) - Summary across all 6 clips")
grouped_10hz = field_10hz.groupby('codec').agg({
    'mean_bpf': 'mean',
    'median_bpf': 'mean',
    'ci_low_bpf': 'mean',
    'ci_high_bpf': 'mean',
    'sum_cr_raw2': 'mean',
    'sum_cr_png': 'mean',
    'mean_enc_ms': 'mean',
    'p95_enc_ms': 'mean',
    'mean_dec_ms': 'mean',
    'lossless_passed': 'all'
}).loc[sub_kf['codec'].values]
grouped_10hz = grouped_10hz.reset_index()
print(format_table(grouped_10hz, "Field Test Videos (10 Hz subsampled, Average across 6 clips)"))

# Print per-clip breakdowns for field tests
for clip_name in field_native['dataset'].unique():
    clip_df = field_native[field_native['dataset'] == clip_name]
    print(format_table(clip_df, f"{clip_name}"))
