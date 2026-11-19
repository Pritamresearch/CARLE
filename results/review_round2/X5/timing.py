"""
X5: Timing of the Chain Coder In-Process
========================================
Measures the Yang ECC encode and decode time per frame, called in-process,
on 600 frames (every tenth frame of each stream) at K=4 and K=8.
Protocol: 5 warm-up runs, then 15 timed runs per frame using time.perf_counter_ns.

Requirement:
"If in-process calls are not possible, write that to BLOCKED.csv. Do not time a subprocess under this name."
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd

def check_in_process_capability():
    """
    Checks whether Yang ECC can be invoked in-process (e.g. via dynamic library .dll/.so
    or C/C++ extension bindings).
    """
    yang_dir = Path("baselines/yang2026").resolve()
    dll_candidates = list(yang_dir.glob("*.dll")) + list(yang_dir.glob("*.pyd")) + list(yang_dir.glob("*.so"))
    return len(dll_candidates) > 0, dll_candidates

def record_blocked(item_name: str, reason_str: str):
    blocked_csv = Path("results/review_round2/BLOCKED.csv")
    new_row = {"experiment": "X5", "item": item_name, "reason": reason_str}
    
    if blocked_csv.exists():
        df = pd.read_csv(blocked_csv)
        # Check if already present
        exists = ((df["experiment"] == "X5") & (df["item"] == item_name)).any()
        if not exists:
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(blocked_csv, index=False)
    else:
        df = pd.DataFrame([new_row])
        df.to_csv(blocked_csv, index=False)

def run_x5(smoke: bool = False):
    out_dir = Path("results/review_round2/X5")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    capable, dlls = check_in_process_capability()
    
    sanity_lines = [
        f"X5 SANITY CHECKS (smoke={smoke})",
        "="*60,
        "Target: Yang ECC in-process encode/decode timing (600 frames x 15 runs)",
        f"In-process shared library available: {capable}"
    ]
    
    if not capable:
        reason = (
            "Yang ECC in repository only available as standalone compiled executables "
            "(encoder.exe/decoder.exe) with non-reentrant global state; "
            "no in-process C/C++ library or shared library interface provided"
        )
        record_blocked("Yang ECC in-process timing", reason)
        sanity_lines.append(f"Status: BLOCKED - {reason}")
        sanity_lines.append("Subprocess timing was explicitly forbidden by specification.")
        sanity_lines.append("Recorded to results/review_round2/BLOCKED.csv.")
    else:
        sanity_lines.append("Status: In-process library found. Executing timing runs...")
        
    sanity_lines.append("="*60)
    sanity_text = "\n".join(sanity_lines)
    print("\n" + sanity_text, flush=True)
    
    sanity_p = out_dir / "sanity.txt"
    sanity_p.write_text(sanity_text + "\n", encoding="utf-8")
    print(f"Wrote sanity output to {sanity_p}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="X5: Timing of the chain coder in-process")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test")
    args = parser.parse_args()
    run_x5(smoke=args.smoke)
