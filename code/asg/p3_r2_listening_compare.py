"""Compare two of our result ledgers (default: sep2 front end vs sep3
front end) on the exact ten listening-test trials, so the decision
whether the listening test must be repeated rests on numbers.

  python src/p3_r2_listening_compare.py [--a p3_r2_ours.json] [--b p3_r2_ours3.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np

PN = Path("/workspace/CF_ASG/paper_numbers")
CFG = Path("/workspace/CF_ASG/p3/listening_test_v2/config.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="p3_r2_ours.json")
    ap.add_argument("--b", default="p3_r2_ours3.json")
    args = ap.parse_args()
    A = json.loads((PN / args.a).read_text())
    B = json.loads((PN / args.b).read_text())
    trials = json.loads(CFG.read_text())
    trials = trials.get("trials", trials)
    print(f"{'trial':>5} {'scene':>5} edit {'A sisdr':>8} {'B sisdr':>8} {'diff':>6} {'A mrstft':>8} {'B mrstft':>8}")
    diffs = []
    for t_i, t in enumerate(trials):
        s, e = t["scene"], t["edit"]
        ra = {r["scene"]: r for r in A["rows"][e]}.get(s)
        rb = {r["scene"]: r for r in B["rows"][e]}.get(s)
        if ra is None or rb is None:
            print(f"{t_i+1:5d} {s:5d} {e}   missing")
            continue
        d = rb["sisdr"] - ra["sisdr"]
        diffs.append(d)
        print(f"{t_i+1:5d} {s:5d} {e}  {ra['sisdr']:8.2f} {rb['sisdr']:8.2f} {d:6.2f} {ra['mrstft']:8.3f} {rb['mrstft']:8.3f}")
    if diffs:
        print(f"median diff {np.median(diffs):.2f} dB, max |diff| {np.max(np.abs(diffs)):.2f} dB, "
              f"trials with |diff| > 3 dB: {sum(abs(x) > 3 for x in diffs)} of {len(diffs)}")
    for e in A["edits"]:
        if e in B["edits"]:
            print(f"cohort median {e}: A {A['edits'][e]['sisdr']['median']:.2f}  B {B['edits'][e]['sisdr']['median']:.2f}")


if __name__ == "__main__":
    main()
