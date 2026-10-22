"""A12.1 (R-C) confirmation-cohort verdict: tests EXACTLY the
pre-registered claims on the 250-scene conf1 cohort (seed 1, disjoint
from the main cohort):

(a) E3: paired superiority of ours (S3 system-of-record) vs EACH of
    X1..X4, scene-paired Wilcoxon + Holm over the four comparisons,
    p < 0.05, ours better.
(b) E1: superiority vs X2/X3/X4 (scene-paired Wilcoxon, p < 0.05,
    ours better) and parity vs X1 (TOST, +-3 dB margin on the paired
    difference, one-sided Wilcoxons, p < 0.05 both sides).
(c) E2: no significant separation-family difference expected --
    reported as-is (paired medians, CIs, p), no pass criterion.

X5/X6 excluded per A12.1 (main-cohort E2/E3 deficits are 20-60 dB; no
plausible flip; disclosed).
-> paper_numbers/p3_confirm_verdict_v0.json
"""
import argparse
import datetime
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

PN = Path("/workspace/CF_ASG/paper_numbers")
XFILES = {"X1": ("p3_baselines_x1_c1.json", "S5"),
          "X2": ("p3_baselines_x2_c1.json", "X2"),
          "X3": ("p3_baselines_x3_c1.json", "X3"),
          "X4": ("p3_baselines_x4_c1.json", "X4")}
MARGIN = 3.0


def rows_of(d, key, e):
    r = d[key].get("rows", {}).get(e, [])
    return {row["scene"]: row["sisdr"] for row in r}


def paired_diffs(a, b):
    common = sorted(set(a) & set(b))
    return [a[s] - b[s] for s in common]


def boot_ci(diffs, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(diffs)
    m = np.median(rng.choice(v, (5000, len(v))), axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def cmp_stats(a, b):
    d = paired_diffs(a, b)
    if len(d) < 8:
        return None
    d = np.asarray(d)
    tost_lo = float(wilcoxon(d + MARGIN, alternative="greater").pvalue)
    tost_hi = float(wilcoxon(d - MARGIN, alternative="less").pvalue)
    return {"n_paired": len(d),
            "diff_median": float(np.median(d)),
            "diff_ci95": boot_ci(d),
            "wilcoxon_p": float(wilcoxon(d).pvalue),
            "ours_better": bool(np.median(d) > 0),
            "tost_3db": {"margin_db": MARGIN,
                         "p_lower": tost_lo, "p_upper": tost_hi,
                         "p": max(tost_lo, tost_hi),
                         "parity": bool(max(tost_lo, tost_hi) < 0.05)}}


def holm(ps):
    """{name: p} -> {name: {p, threshold, pass}} step-down."""
    order = sorted(ps, key=lambda k: ps[k])
    m = len(order)
    out, ok = {}, True
    for i, k in enumerate(order):
        thr = 0.05 / (m - i)
        passed = ok and ps[k] <= thr
        out[k] = {"p": ps[k], "threshold": thr, "pass": passed}
        if not passed:
            ok = False
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default="p3_edit_eval_c1.json")
    ap.add_argument("--out", default="p3_confirm_verdict_v0.json")
    args = ap.parse_args()
    ours_led = json.loads((PN / args.ledger).read_text())
    o_rows = {e: {r["scene"]: r["sisdr"]
                  for r in ours_led["rows"]["S3"][e]}
              for e in ("E1", "E2", "E3", "E4")}
    out = {"meta": {"date": str(datetime.date.today()),
                    "script": "src/p3_confirm_verdict.py",
                    "cohort": ours_led["meta"].get("split"),
                    "ours_ledger": args.ledger,
                    "criteria": "PROTOCOL_P3.md A12.1 (a)(b)(c), "
                                "pre-registered 2026-08-16 before the "
                                "cohort was rendered",
                    "x_excluded": "X5/X6 per A12.1 (20-60 dB main-"
                                  "cohort deficits, no plausible flip)"},
           "ours_median": {e: float(np.median(list(o_rows[e].values())))
                           for e in o_rows if o_rows[e]},
           "vs": {}, "verdict": {}}
    for xn, (f, key) in XFILES.items():
        d = json.loads((PN / f).read_text())
        out["vs"][xn] = {}
        for e in ("E1", "E2", "E3"):
            xr = rows_of(d, key, e)
            s = cmp_stats(o_rows[e], xr) if xr else None
            if s is not None:
                s["x_median"] = float(np.median(list(xr.values())))
            out["vs"][xn][e] = s

    # (a) E3 superiority vs each of X1..X4, Holm
    e3_ps = {xn: out["vs"][xn]["E3"]["wilcoxon_p"] for xn in XFILES}
    e3_holm = holm(e3_ps)
    for xn in e3_holm:
        e3_holm[xn]["pass"] = bool(e3_holm[xn]["pass"]
                                   and out["vs"][xn]["E3"]["ours_better"])
    out["verdict"]["a_E3_holm"] = e3_holm
    out["verdict"]["a_E3_pass"] = all(v["pass"] for v in e3_holm.values())

    # (b) E1: superiority vs X2/X3/X4 (raw p<0.05 as registered; Holm
    # reported as sensitivity), parity (TOST 3 dB) vs X1
    b = {}
    for xn in ("X2", "X3", "X4"):
        v = out["vs"][xn]["E1"]
        b[xn] = {"superior": bool(v["ours_better"]
                                  and v["wilcoxon_p"] < 0.05),
                 "wilcoxon_p": v["wilcoxon_p"],
                 "diff_median": v["diff_median"]}
    b["holm_sensitivity"] = holm({xn: out["vs"][xn]["E1"]["wilcoxon_p"]
                                  for xn in ("X2", "X3", "X4")})
    v1 = out["vs"]["X1"]["E1"]
    b["X1_parity_tost"] = v1["tost_3db"]
    b["X1_diff_median"] = v1["diff_median"]
    out["verdict"]["b_E1"] = b
    out["verdict"]["b_E1_pass"] = bool(
        all(b[xn]["superior"] for xn in ("X2", "X3", "X4"))
        and v1["tost_3db"]["parity"])

    # (c) E2 reported as-is
    out["verdict"]["c_E2_reported"] = {xn: out["vs"][xn]["E2"]
                                       for xn in XFILES}

    (PN / args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out["verdict"], indent=1)[:3000])


if __name__ == "__main__":
    main()
