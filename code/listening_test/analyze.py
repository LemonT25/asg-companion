"""Aggregate result files: python analyze.py results/*.json
Per edit and condition: median agreement over listeners (each listener's
mean over trials), paired Wilcoxon against ours and against the hidden
ground truth (gt). Screening: a listener is dropped when the hidden
ground truth is rated below the untouched original in more than 30% of
trials (they could not follow the instruction)."""
import json, sys
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon

R = defaultdict(lambda: defaultdict(list))
kept = 0
subs = {}
for f in sys.argv[1:]:
    d = json.load(open(f))
    k = d["listener"].strip().lower()
    if k in subs and subs[k]["started"] <= d["started"]:
        print("duplicate listener id", d["listener"], "later submission ignored (first attempt counts)")
        continue
    if k in subs:
        print("duplicate listener id", d["listener"], "later submission ignored (first attempt counts)")
    subs[k] = d
for d in subs.values():
    bad = [a["ratings"].get("gt", 100) < a["ratings"].get("orig", 0) for a in d["answers"]]
    if np.mean(bad) > 0.3:
        print("excluded", d["listener"]); continue
    kept += 1
    for a in d["answers"]:
        for c, v in a["ratings"].items():
            R[(a["edit"], c)][d["listener"]].append(v)
print("listeners kept", kept)
for e in sorted({e for e, _ in R}):
    print("==", e)
    for c in ("gt", "orig", "ours", "x1", "x3", "x6"):
        if (e, c) not in R: continue
        x = [np.mean(v) for v in R[(e, c)].values()]
        line = f"{c:5s} median {np.median(x):5.1f} (n={len(x)})"
        for ref in ("ours", "gt"):
            if ref != c and (e, ref) in R and len(x) >= 5:
                y = [np.mean(R[(e, ref)][l]) for l in R[(e, c)]]
                line += f"  vs {ref} p={wilcoxon(x, y).pvalue:.3g}" if any(np.array(x) != np.array(y)) else f"  vs {ref} p=1"
        print(line)
