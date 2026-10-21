"""Listening test v2: bootstrap confidence intervals for the paired
median differences reported in the paper. Same screening as
p3/listening_test_v2/analyze.py (first submission per listener id;
drop listeners who rate the hidden ground truth below the untouched
original in more than 30% of trials). Per listener and (edit,
condition) the mean over that listener's trials is used, as in the
table. 10 000 bootstrap resamples over listeners, percentile 95% CI of
the median paired difference.
-> paper_numbers/p3_r2_listening_ci.json
"""
import glob
import json
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

RES = "/workspace/CF_ASG/p3/listening_test_v2/results"
PN = "/workspace/CF_ASG/paper_numbers"
NAME = {"gt": "Ground truth", "ours": "Ours", "x1": "AudioSep",
        "x3": "SoloAudio", "x6": "Audio-Omni", "orig": "Unedited mixture"}
PAIRS = [("ours", "gt"), ("ours", "x1"), ("ours", "x3"), ("ours", "x6"),
         ("ours", "orig"), ("gt", "orig")]


def main():
    subs = {}
    for f in sorted(glob.glob(f"{RES}/*.json")):
        d = json.load(open(f))
        k = d["listener"].strip().lower()
        if k in subs and subs[k]["started"] <= d["started"]:
            continue
        subs[k] = d
    kept = []
    for d in subs.values():
        bad = [a["ratings"].get("gt", 100) < a["ratings"].get("orig", 0)
               for a in d["answers"]]
        if np.mean(bad) <= 0.3:
            kept.append(d)
    per = defaultdict(lambda: defaultdict(dict))   # edit -> cond -> listener -> mean
    for li, d in enumerate(kept):
        acc = defaultdict(lambda: defaultdict(list))
        for a in d["answers"]:
            for c, v in a["ratings"].items():
                acc[a["edit"]][c].append(v)
        for e, cs in acc.items():
            for c, v in cs.items():
                per[e][c][li] = float(np.mean(v))
    rng = np.random.default_rng(0)
    out = {"meta": {"n_listeners_total": len(subs), "n_kept": len(kept),
                    "bootstrap": 10000, "ci": "percentile 95% of the median "
                    "paired difference over listeners"},
           "medians": {}, "pairs": {}}
    for e in sorted(per):
        out["medians"][e] = {c: {"median": float(np.median(list(v.values()))),
                                 "n": len(v)} for c, v in per[e].items()}
        for a, b in PAIRS:
            if a not in per[e] or b not in per[e]:
                continue
            com = sorted(set(per[e][a]) & set(per[e][b]))
            d = np.array([per[e][a][l] - per[e][b][l] for l in com])
            if len(d) < 5:
                continue
            boots = [float(np.median(d[rng.integers(0, len(d), len(d))]))
                     for _ in range(10000)]
            p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
            out["pairs"].setdefault(e, {})[f"{a}_vs_{b}"] = {
                "median_diff": float(np.median(d)),
                "ci95": [float(np.percentile(boots, 2.5)),
                         float(np.percentile(boots, 97.5))],
                "p": p, "n": len(d)}
    json.dump(out, open(f"{PN}/p3_r2_listening_ci.json", "w"), indent=1)
    for e in out["pairs"]:
        print(e, {k: (round(v["median_diff"], 1),
                      [round(x, 1) for x in v["ci95"]], "%.2g" % v["p"])
                  for k, v in out["pairs"][e].items()})


if __name__ == "__main__":
    main()
