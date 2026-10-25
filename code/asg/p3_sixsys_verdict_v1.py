"""R-E (fix plan L9 / A12): verdict v2 on the _v1 main ledgers.
Extends p3_sixsys_verdict.py with (i) paired-difference bootstrap CIs
for every comparison, (ii) TOST-style parity tests at the
pre-registered 3 dB margin wherever the paper claims parity (C-B: E1
vs the separation family), (iii) the C4 locality gate report from the
new per-row loc metric.  Criteria roster unchanged (A9: C-A/C-B/C-C).
-> paper_numbers/p3_sixsys_verdict_v1.json
"""
import datetime
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

PN = Path("/workspace/CF_ASG/paper_numbers")
OURS_LEDGER = "p3_edit_eval_v1.json"
XFILES = {"X1": ("p3_baselines_x1_v1.json", "S5"),
          "X2": ("p3_baselines_x2_v1.json", "X2"),
          "X3": ("p3_baselines_x3_v1.json", "X3"),
          "X4": ("p3_baselines_x4_v1.json", "X4"),
          "X5": ("p3_baselines_x5_v1.json", "X5"),
          "X6": ("p3_baselines_x6_v1.json", "X6")}
SEP_FAMILY = ("X1", "X2", "X3", "X4")
MARGIN = 3.0


def rows_of(d, key, e, metric="sisdr"):
    r = d[key].get("rows", {}).get(e, [])
    return {row["scene"]: row[metric] for row in r if metric in row}


def boot_ci(diffs, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(diffs)
    m = np.median(rng.choice(v, (5000, len(v))), axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def paired(a, b):
    common = sorted(set(a) & set(b))
    d = np.asarray([a[s] - b[s] for s in common])
    if len(d) < 8:
        return None
    tost_lo = float(wilcoxon(d + MARGIN, alternative="greater").pvalue)
    tost_hi = float(wilcoxon(d - MARGIN, alternative="less").pvalue)
    return {"n_paired": len(d),
            "diff_median": float(np.median(d)),
            "diff_ci95": boot_ci(d),
            "wilcoxon_p": float(wilcoxon(d).pvalue),
            "ours_better": bool(np.median(d) > 0),
            "tost_3db": {"margin_db": MARGIN, "p_lower": tost_lo,
                         "p_upper": tost_hi,
                         "p": max(tost_lo, tost_hi),
                         "parity": bool(max(tost_lo, tost_hi) < 0.05)}}


def main():
    ours = json.loads((PN / OURS_LEDGER).read_text())
    o_rows = {e: {r["scene"]: r["sisdr"] for r in ours["rows"]["S3"][e]}
              for e in ("E1", "E2", "E3", "E4")}
    med = {e: float(np.median(list(o_rows[e].values())))
           for e in o_rows if o_rows[e]}
    out = {"meta": {"date": str(datetime.date.today()),
                    "script": "src/p3_sixsys_verdict_v1.py",
                    "ours_ledger": OURS_LEDGER,
                    "criteria": ("PROTOCOL_P3.md A9 (C-A/C-B/C-C) + "
                                 "A12/R-E: paired-diff bootstrap CIs "
                                 "and TOST parity at the 3 dB margin "
                                 "for all parity claims (C-B)")},
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
            s = paired(o_rows[e], xr)
            if s is not None:
                s["x_median"] = float(np.median(list(xr.values())))
            out["vs"][xn][e] = s
            if e == "E2" and s is not None:
                e2_ps[xn] = s["wilcoxon_p"]
    # C-A: Holm over the six E2 comparisons
    order = sorted(e2_ps, key=lambda k: e2_ps[k])
    m = len(order)
    holm, ok = {}, True
    for i, xn in enumerate(order):
        thr = 0.05 / (m - i)
        passed = ok and e2_ps[xn] <= thr \
            and out["vs"][xn]["E2"]["ours_better"]
        holm[xn] = {"p": e2_ps[xn], "threshold": thr, "pass": passed}
        if not passed:
            ok = False
    out["verdict"]["C-A_E2_holm"] = holm
    out["verdict"]["C-A_E2_all_pass"] = all(v["pass"]
                                            for v in holm.values())
    out["verdict"]["C-A_E4"] = ("no external adapter implements an E4 "
                                "path (separation family: no re-render "
                                "machinery of their own; regeneration "
                                "family: E4 not expressible in their "
                                "instruction interface) -- documented")
    # C-B: E1 win-or-parity vs the separation family; parity now a
    # TOST decision at the pre-registered 3 dB margin (L9)
    cb = {}
    for xn in SEP_FAMILY:
        v = out["vs"][xn]["E1"]
        cb[xn] = {"win": bool(v["ours_better"]
                              and v["wilcoxon_p"] < 0.05),
                  "parity_tost_3db": v["tost_3db"], **v}
    out["verdict"]["C-B_E1"] = cb
    out["verdict"]["C-B_pass"] = all(
        c["win"] or c["parity_tost_3db"]["parity"] for c in cb.values())
    out["verdict"]["C-C_E3"] = {xn: out["vs"][xn]["E3"] for xn in XFILES}

    # C4 locality gate (pre-registered: identity-based edits >= 30 dB
    # on non-target content).  Two operationalizations, both reported:
    #   loc       (A12.2 proxy, every system, node error charged)
    #   loc_clean (identity systems only: si_sdr(y - s_hat, y - gts),
    #              the actual by-construction quantity C4 refers to)
    # Gate decided on loc_clean; proxy reported alongside.
    loc, locc = {}, {}
    for sysk in ours["rows"]:
        loc[sysk], locc[sysk] = {}, {}
        for e in ours["rows"][sysk]:
            for metric, dst in (("loc", loc), ("loc_clean", locc)):
                v = [r[metric] for r in ours["rows"][sysk][e]
                     if metric in r]
                if v:
                    dst[sysk][e] = {"median": float(np.median(v)),
                                    "n": len(v)}
    for xn, (f, key) in XFILES.items():
        d = json.loads((PN / f).read_text())
        loc[xn], locc[xn] = {}, {}
        for e in ("E1", "E2", "E3"):
            for metric, dst in (("loc", loc), ("loc_clean", locc)):
                xr = rows_of(d, key, e, metric)
                if xr:
                    dst[xn][e] = {"median": float(np.median(
                        list(xr.values()))), "n": len(xr)}
    ours_locc = {e: locc.get("S3", {}).get(e) for e in ("E2", "E3")}
    out["verdict"]["C4_locality"] = {
        "gate": "identity-based edits >= 30 dB median non-target "
                "locality (E2/E3, loc_clean; E1 loc==sisdr by "
                "construction, excluded from the gate; E4 has no "
                "surviving bed, proxy only)",
        "metric": "loc_clean (see main-ledger meta); A12.2 proxy loc "
                  "reported alongside (charges node-render error; "
                  "only definition available for X5/X6)",
        "ours_S3_loc_clean": ours_locc,
        "pass": all(v and v["median"] >= 30.0
                    for v in ours_locc.values()),
        "all_systems_median_loc": loc,
        "all_systems_median_loc_clean": locc}
    (PN / "p3_sixsys_verdict_v1.json").write_text(json.dumps(out,
                                                             indent=1))
    print(json.dumps(out["verdict"], indent=1)[:2600])
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
