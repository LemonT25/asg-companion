"""A9 final verdict: ours (S3 system-of-record) vs X1..X6 on the main
ledger's shared edit specs.  Scene-paired Wilcoxon; Holm correction
over the six E2 comparisons (C-A); E1 vs separation family (C-B);
E3 reported per system (C-C).  E4: no external adapter implements a
path (documented).  -> paper_numbers/p3_sixsys_verdict_v0.json
"""
import datetime
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

PN = Path("/workspace/CF_ASG/paper_numbers")
XFILES = {"X1": ("p3_baselines_x1_v0.json", "S5"),
          "X2": ("p3_baselines_x2_v0.json", "X2"),
          "X3": ("p3_baselines_x3_v0.json", "X3"),
          "X4": ("p3_baselines_x4_v0.json", "X4"),
          "X5": ("p3_baselines_x5_v0.json", "X5"),
          "X6": ("p3_baselines_x6_v0.json", "X6")}
SEP_FAMILY = ("X1", "X2", "X3", "X4")


def rows_of(d, key, e):
    r = d[key].get("rows", {}).get(e, [])
    return {row["scene"]: row["sisdr"] for row in r}


def paired(a, b):
    common = sorted(set(a) & set(b))
    diffs = [a[s] - b[s] for s in common]
    if len(diffs) < 8:
        return None, len(diffs), None
    stat = wilcoxon(diffs)
    return float(stat.pvalue), len(diffs), float(np.median(diffs))


def main():
    ours = json.loads((PN / "p3_edit_eval_v0.json").read_text())
    o_rows = {e: {r["scene"]: r["sisdr"] for r in ours["rows"]["S3"][e]}
              for e in ("E1", "E2", "E3", "E4")}
    med = {e: float(np.median(list(o_rows[e].values())))
           for e in o_rows if o_rows[e]}
    out = {"meta": {"date": str(datetime.date.today()),
                    "script": "src/p3_sixsys_verdict.py",
                    "criteria": "PROTOCOL_P3.md A9 (C-A/C-B/C-C)"},
           "ours_median": med, "vs": {}, "verdict": {}}
    e2_ps = {}
    for xn, (f, key) in XFILES.items():
        d = json.loads((PN / f).read_text())
        out["vs"][xn] = {}
        for e in ("E1", "E2", "E3"):
            xr = rows_of(d, key, e)
            if not xr:
                out["vs"][xn][e] = None
                continue
            p, n, dmed = paired(o_rows[e], xr)
            xmed = float(np.median(list(xr.values())))
            out["vs"][xn][e] = {"x_median": xmed, "n_paired": n,
                                "diff_median": dmed, "wilcoxon_p": p,
                                "ours_better": (dmed or 0) > 0}
            if e == "E2" and p is not None:
                e2_ps[xn] = p
    # Holm over the six E2 comparisons (C-A)
    order = sorted(e2_ps, key=lambda k: e2_ps[k])
    m = len(order)
    holm = {}
    ok = True
    for i, xn in enumerate(order):
        thr = 0.05 / (m - i)
        passed = ok and e2_ps[xn] <= thr \
            and out["vs"][xn]["E2"]["ours_better"]
        holm[xn] = {"p": e2_ps[xn], "threshold": thr, "pass": passed}
        if not passed:
            ok = False
    out["verdict"]["C-A_E2_holm"] = holm
    out["verdict"]["C-A_E2_all_pass"] = all(v["pass"] for v in holm.values())
    out["verdict"]["C-A_E4"] = ("no external adapter implements an E4 "
                                "path (separation family: no re-render "
                                "machinery of their own; regeneration "
                                "family: E4 not expressible in their "
                                "instruction interface) -- documented")
    cb = {}
    for xn in SEP_FAMILY:
        v = out["vs"][xn]["E1"]
        cb[xn] = {"win": v["ours_better"] and (v["wilcoxon_p"] or 1) < 0.05,
                  "parity_within_3db": abs(v["diff_median"] or 99) <= 3.0
                  or v["ours_better"], **v}
    out["verdict"]["C-B_E1"] = cb
    out["verdict"]["C-B_pass"] = all(
        c["win"] or c["parity_within_3db"] for c in cb.values())
    out["verdict"]["C-C_E3"] = {xn: out["vs"][xn]["E3"] for xn in XFILES}
    (PN / "p3_sixsys_verdict_v0.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["verdict"], indent=1)[:2400])
    print()
    print("%-4s %8s %8s %8s" % ("sys", "E1", "E2", "E3"))
    print("%-4s %8.2f %8.2f %8.2f" % ("ours", med["E1"], med["E2"],
                                      med["E3"]))
    for xn in XFILES:
        v = out["vs"][xn]
        print("%-4s %8s %8s %8s" % (
            xn, *[("%.2f" % v[e]["x_median"]) if v[e] else "--"
                  for e in ("E1", "E2", "E3")]))


if __name__ == "__main__":
    main()
