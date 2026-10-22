"""A14 aggregate: chains, CLAP, RT60 across systems -> one ledger and
LaTeX fragments.

Inputs: paper_numbers/p3_r2_<sys>.json (rows per edit, SI-SDR/mrSTFT/
locality) and p3_r2_metrics_<sys>.json (CLAP, RT60, EDC) for
sys in ours, x1..x6 (missing systems skipped). Paired tests are
scene-paired Wilcoxon signed-rank on the common scenes, Holm-corrected
over the external systems per (edit, metric).
-> paper_numbers/p3_r2_verdict.json, paper/gen/tab_r2_*.tex
"""
import datetime
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

PN = Path("/workspace/CF_ASG/paper_numbers")
GEN = Path("/workspace/CF_ASG/paper/gen")
SYS = [("ours", "Ours"), ("x1", "AudioSep"), ("x2", "FlowSep"),
       ("x3", "SoloAudio"), ("x4", "OmniSep"), ("x5", "ZETA"),
       ("x6", "Audio-Omni")]
EDITS = ["E1", "E2", "E3", "E4", "C1", "C2", "C3", "C4"]
CHAINS = ["C1", "C2", "C3", "C4"]


def J(p):
    return json.loads(p.read_text()) if p.exists() else None


def holm(ps):
    """Holm step-down; returns adjusted p in the input order."""
    idx = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(idx):
        run = max(run, (m - r) * ps[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def rows_by_scene(rows, key):
    return {r["scene"]: r[key] for r in rows if key in r and r[key] is not None}


def paired(a, b, higher_better=True):
    """a, b: {scene: value}. Returns median diff (a-b), p, n."""
    common = sorted(set(a) & set(b))
    if len(common) < 8:
        return None
    x = np.array([a[s] for s in common])
    y = np.array([b[s] for s in common])
    d = x - y
    p = float(wilcoxon(x, y).pvalue) if np.any(d != 0) else 1.0
    return {"median_diff": float(np.median(d)), "p": p, "n": len(common),
            "ours_better": bool(np.median(d) > 0) == higher_better}


def f2(x):
    return "--" if x is None else f"{x:.2f}"


def main():
    L = {k: J(PN / f"p3_r2_{k}.json") for k, _ in SYS}
    M = {k: J(PN / f"p3_r2_metrics_{k}.json") for k, _ in SYS}
    out = {"meta": {"date": str(datetime.date.today()),
                    "script": "src/p3_r2_verdict.py",
                    "test": "scene-paired Wilcoxon signed-rank, Holm over"
                            " the external systems per (edit, metric)",
                    "systems_present": [k for k, _ in SYS if L[k]]},
           "medians": {}, "paired": {}, "regen_forms": {}}
    # ---------------- medians per system / edit / metric
    metrics = {"sisdr": ("rows", True), "mrstft": ("rows", False),
               "loc": ("rows", True), "clap_ref": ("m", True),
               "drop": ("m", True), "keep": ("m", None),
               "rt60_off_err": ("m", False), "edc_err": ("m", False)}
    for k, _ in SYS:
        if not L[k]:
            continue
        out["medians"][k] = {}
        for e in EDITS + [c + "seq" for c in CHAINS]:
            rows = L[k]["rows"].get(e, [])
            mrows = (M[k] or {}).get("rows", {}).get(e, []) if M[k] else []
            d = {}
            for mname, (src, _) in metrics.items():
                rr = rows if src == "rows" else mrows
                v = [r[mname] for r in rr if mname in r and r[mname] is not None]
                if v:
                    d[mname] = {"median": float(np.median(v)), "n": len(v)}
            if d:
                out["medians"][k][e] = d
    # anchor and reference rungs (from ours' metrics file)
    if M["ours"]:
        out["anchor"] = M["ours"]["anchor"]
        out["reference"] = M["ours"]["reference"]
        out["rt60_validation"] = M["ours"].get("rt60_validation")
    # ---------------- paired ours vs each external
    for e in EDITS:
        for mname, (src, hb) in metrics.items():
            if hb is None:
                continue
            res = {}
            ps, keys = [], []
            for k, _ in SYS[1:]:
                if not L[k]:
                    continue
                if src == "rows":
                    a = rows_by_scene(L["ours"]["rows"].get(e, []), mname)
                    b = rows_by_scene(L[k]["rows"].get(e, []), mname)
                else:
                    if not (M["ours"] and M[k]):
                        continue
                    a = rows_by_scene(M["ours"]["rows"].get(e, []), mname)
                    b = rows_by_scene(M[k]["rows"].get(e, []), mname)
                r = paired(a, b, hb)
                if r:
                    res[k] = r
                    ps.append(r["p"]); keys.append(k)
            if ps:
                for k, pa in zip(keys, holm(ps)):
                    res[k]["p_holm"] = pa
                out["paired"].setdefault(e, {})[mname] = res
    # ---------------- regen: combined vs sequential
    for k in ("x5", "x6"):
        if not L[k]:
            continue
        for c in CHAINS:
            a = rows_by_scene(L[k]["rows"].get(c, []), "sisdr")
            b = rows_by_scene(L[k]["rows"].get(c + "seq", []), "sisdr")
            r = paired(a, b, True)
            if r:
                out["regen_forms"].setdefault(k, {})[c] = r
    if L["ours"] and "order_test" in L["ours"]:
        out["order_test"] = L["ours"]["order_test"]
    (PN / "p3_r2_verdict.json").write_text(json.dumps(out, indent=1))

    # ---------------- LaTeX: chains table (SI-SDR medians, bold best)
    GEN.mkdir(parents=True, exist_ok=True)
    present = [k for k, _ in SYS if L[k]]
    names = dict(SYS)

    def table(edits, metric, fname, caption_note="", seq=False):
        lines = ["\\begin{tabular}{l" + "r" * len(edits) + "}", "\\toprule",
                 "System & " + " & ".join(edits) + "\\\\", "\\midrule"]
        best = {}
        for e in edits:
            vals = [(out["medians"][k].get(e, {}).get(metric, {}).get("median"), k)
                    for k in present]
            vals = [(v, k) for v, k in vals if v is not None]
            if vals:
                hb = metrics[metric][1]
                if hb is None:          # keep: best is closest to zero
                    best[e] = min((abs(v), k) for v, k in vals)[1]
                else:
                    best[e] = (max if hb else min)(vals)[1]
        for k in present:
            cells = []
            for e in edits:
                v = out["medians"][k].get(e, {}).get(metric, {}).get("median")
                s = f2(v)
                if v is not None and best.get(e) == k:
                    s = "\\textbf{" + s + "}"
                cells.append(s)
            lines.append(names[k] + " & " + " & ".join(cells) + "\\\\")
            if seq and k in ("x5", "x6"):
                cells = [f2(out["medians"][k].get(e + "seq", {}).get(metric, {}).get("median"))
                         for e in edits]
                lines.append(names[k] + " (sequential) & " + " & ".join(cells) + "\\\\")
        if "anchor" in out:
            cells = [f2(out["anchor"].get(e, {}).get(metric, {}).get("median"))
                     for e in edits]
            if any(c != "--" for c in cells):
                lines.append("\\midrule")
                lines.append("Unedited mixture & " + " & ".join(cells) + "\\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        (GEN / fname).write_text("\n".join(lines) + "\n")
        print("wrote", fname)

    table(CHAINS, "sisdr", "tab_r2_chains.tex", seq=True)
    table(CHAINS, "loc", "tab_r2_chains_loc.tex", seq=True)
    table(["E1", "E2", "E3", "E4"] + CHAINS, "clap_ref", "tab_r2_clap.tex")
    table(["E1", "C1", "C3", "C4"], "drop", "tab_r2_drop.tex")
    table(["E1", "C1", "C3", "C4"], "keep", "tab_r2_keep.tex")
    table(["E4", "C3"], "rt60_off_err", "tab_r2_rt60.tex")
    table(["E4", "C3"], "edc_err", "tab_r2_edc.tex")
    # anchor SI-SDR for chains from the ours ledger is not stored; the
    # no-edit anchor for chains is computed by p3_r2_metrics (clap only)
    print(json.dumps({e: {m: {k: (round(v["median_diff"], 2), round(v["p_holm"], 4))
                              for k, v in d.items()}
                          for m, d in out["paired"][e].items() if m == "sisdr"}
                      for e in out["paired"]}, indent=0))


if __name__ == "__main__":
    main()
