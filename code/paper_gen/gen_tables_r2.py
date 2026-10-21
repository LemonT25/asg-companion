"""Revision tables (T-ASL-13354-2026 R1) from the A14 ledgers.
Inputs: paper_numbers/p3_r2_main_table.json (medians, anchors, paired),
        paper_numbers/p3_r2_verdict.json (CLAP, RT60, EDC, regen forms),
        paper_numbers/p3_r2_x5.json / x6 (sequential chain rows),
        p3/listening_test_v2/results/*.json (listening test).
Outputs: paper/gen/tab_main.tex, tab_chains.tex, tab_clap.tex,
         tab_keep.tex, tab_room.tex, tab_listen.tex, tab_locality.tex,
         macros_r2.tex.  Run with /workspace/venvs/audiocf python.
"""
import glob
import json
from pathlib import Path

import numpy as np

PN = Path("/workspace/CF_ASG/paper_numbers")
OUT = Path("/workspace/CF_ASG/paper/gen")
RES = Path("/workspace/CF_ASG/p3/listening_test_v2/results")
SYS = [("ours", "Ours"), ("x1", "AudioSep"), ("x2", "FlowSep"),
       ("x3", "SoloAudio"), ("x4", "OmniSep"), ("x5", "ZETA"),
       ("x6", "Audio-Omni")]
NAME = dict(SYS)
EDITS = ["E1", "E2", "E3", "E4"]
CHAINS = ["C1", "C2", "C3", "C4"]


def J(name):
    return json.loads((PN / name).read_text())


def w(name, s):
    (OUT / name).write_text(s)
    print("wrote", name)


def f2(x, nd=2):
    if x is None:
        return "--"
    t = f"{x:.{nd}f}"
    return "$-$" + t[1:] if t.startswith("-") else t


def fp(p):
    """p-value for text: 0.049, or 2\\times10^{-4} below 1e-3."""
    if p >= 1e-3:
        return f"{p:.2g}"
    m, e = f"{p:.0e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


SHARED = ("x1", "x2", "x3", "x4")   # stem systems share our E4 path
DAG = "$^{\\dagger}$"


def shared_col(cells, higher):
    """E4 column: the stem systems reproduce our output exactly (the
    room edit uses no stems), so their cells show our value marked with
    a dagger and are excluded from the bold comparison."""
    ours = cells[0]
    own = [None if k in SHARED else v for (k, _), v in zip(SYS, cells)]
    b = bold_best(own, higher)
    return [f2(ours) + DAG if k in SHARED and ours is not None else b[j]
            for j, (k, _) in enumerate(SYS)]


def bold_best(cells, higher):
    """cells: list of (value or None). Returns list of strings with the best
    value bold. Ties keep both bold."""
    vals = [v for v in cells if v is not None]
    if not vals:
        return ["--"] * len(cells)
    rv = [None if v is None else round(v, 2) for v in cells]
    rvals = [v for v in rv if v is not None]
    best = max(rvals) if higher else min(rvals)
    return ["\\textbf{" + f2(v) + "}" if r is not None and abs(r - best) < 1e-9
            else f2(v) for v, r in zip(cells, rv)]


def main():
    M = J("p3_r2_main_table.json")
    V = J("p3_r2_verdict.json")
    macros = {}

    # ---------------- Table: main (SI-SDR | mrSTFT), eight columns
    def med(sysk, e, met):
        d = M["medians"].get(e, {}).get(met, {}).get(sysk)
        return None if d is None else d["median"]
    cols_si = {e: [med(k, e, "sisdr") for k, _ in SYS] for e in EDITS}
    cols_mr = {e: [med(k, e, "mrstft") for k, _ in SYS] for e in EDITS}
    si_b = {e: (shared_col if e == "E4" else bold_best)(cols_si[e], True)
            for e in EDITS}
    mr_b = {e: (shared_col if e == "E4" else bold_best)(cols_mr[e], False)
            for e in EDITS}
    lines = ["\\begin{tabular}{lrrrrrrrr}", "\\toprule",
             "& \\multicolumn{4}{c}{SI-SDR (dB, higher is better)} & "
             "\\multicolumn{4}{c}{mrSTFT (lower is better)}\\\\",
             "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}",
             "System & E1 & E2 & E3 & E4 & E1 & E2 & E3 & E4\\\\", "\\midrule"]
    for j, (k, nm) in enumerate(SYS):
        lines.append(nm + " & " + " & ".join(si_b[e][j] for e in EDITS) + " & "
                     + " & ".join(mr_b[e][j] for e in EDITS) + "\\\\")
    lines.append("\\midrule")
    anc = [M["anchor"][e]["sisdr_median"] for e in EDITS]
    anc_mr = [M["anchor"][e].get("mrstft_median") for e in EDITS]
    lines.append("Unedited mixture & " + " & ".join(f2(a) for a in anc) + " & "
                 + " & ".join(f2(a) for a in anc_mr) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    w("tab_main.tex", "\n".join(lines) + "\n")
    for e in EDITS:
        macros[f"anchor{e}"] = f2(M["anchor"][e]["sisdr_median"])
        macros[f"anchormr{e}"] = f2(M["anchor"][e].get("mrstft_median"))
        macros[f"n{e}"] = str(M["medians"][e]["sisdr"]["ours"]["n"])
        for k, _ in SYS:
            d = M["medians"].get(e, {}).get("sisdr", {}).get(k)
            if d is not None:
                macros[f"n{e}{k}"] = str(d["n"])
            v = med(k, e, "sisdr")
            if v is not None:
                macros[f"si{e}{k}"] = f2(v)
            v = med(k, e, "mrstft")
            if v is not None:
                macros[f"mr{e}{k}"] = f2(v)
    # paired macros (median diff, Holm p) for sisdr
    for e in EDITS + CHAINS:
        for k, d in M["paired"].get(e, {}).get("sisdr", {}).items():
            macros[f"pd{e}{k}"] = f2(d["median_diff"])
            macros[f"pp{e}{k}"] = fp(d["p_holm"])
        for k, d in M["paired"].get(e, {}).get("mrstft", {}).items():
            macros[f"pdmr{e}{k}"] = f2(d["median_diff"])
            macros[f"ppmr{e}{k}"] = fp(d["p_holm"])

    # ---------------- Table: chains (SI-SDR, with sequential rows)
    lines = ["\\begin{tabular}{lrrrr}", "\\toprule",
             "System & C1: E1+E3 & C2: E2+E3 & C3: E1+E4 & C4: E1+E2\\\\", "\\midrule"]
    cols = {c: [med(k, c, "sisdr") for k, _ in SYS] for c in CHAINS}
    cb = {c: bold_best(cols[c], True) for c in CHAINS}
    for j, (k, nm) in enumerate(SYS):
        lines.append(nm + " & " + " & ".join(cb[c][j] for c in CHAINS) + "\\\\")
        if k in ("x5", "x6"):
            seq = [V["medians"].get(k, {}).get(c + "seq", {}).get("sisdr", {}).get("median") for c in CHAINS]
            lines.append(nm + " (two steps) & " + " & ".join(f2(s) for s in seq) + "\\\\")
    lines.append("\\midrule")
    lines.append("Unedited mixture & " + " & ".join(f2(M["anchor"][c]["sisdr_median"]) for c in CHAINS) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    w("tab_chains.tex", "\n".join(lines) + "\n")
    for c in CHAINS:
        macros[f"n{c}"] = str(M["medians"][c]["sisdr"]["ours"]["n"])
        macros[f"anchor{c}"] = f2(M["anchor"][c]["sisdr_median"])
        for k, _ in SYS:
            v = med(k, c, "sisdr")
            if v is not None:
                macros[f"si{c}{k}"] = f2(v)
        for k in ("x5", "x6"):
            s = V["medians"].get(k, {}).get(c + "seq", {}).get("sisdr", {}).get("median")
            if s is not None:
                macros[f"si{c}{k}seq"] = f2(s)
            r = V["regen_forms"].get(k, {}).get(c)
            if r:
                macros[f"rf{c}{k}d"] = f2(r["median_diff"])
                macros[f"rf{c}{k}p"] = fp(r["p"])

    # ---------------- Table: CLAP similarity to the reference (E1..E4, C1..C4)
    allE = EDITS + CHAINS

    def vmed(k, e, met):
        d = V["medians"].get(k, {}).get(e, {}).get(met)
        return None if d is None else d["median"]
    lines = ["\\begin{tabular}{lrrrrrrrr}", "\\toprule",
             "System & E1 & E2 & E3 & E4 & C1 & C2 & C3 & C4\\\\", "\\midrule"]
    cols = {e: [vmed(k, e, "clap_ref") for k, _ in SYS] for e in allE}
    cb = {e: (shared_col if e == "E4" else bold_best)(cols[e], True)
          for e in allE}
    for j, (k, nm) in enumerate(SYS):
        lines.append(nm + " & " + " & ".join(cb[e][j] for e in allE) + "\\\\")
    lines.append("\\midrule")
    lines.append("Unedited mixture & " + " & ".join(f2(V["anchor"].get(e, {}).get("clap_ref", {}).get("median")) for e in allE) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    w("tab_clap.tex", "\n".join(lines) + "\n")
    for e in allE:
        for k, _ in SYS:
            v = vmed(k, e, "clap_ref")
            if v is not None:
                macros[f"clap{e}{k}"] = f2(v)
        macros[f"clapanchor{e}"] = f2(V["anchor"].get(e, {}).get("clap_ref", {}).get("median"))
        for k, d in V["paired"].get(e, {}).get("clap_ref", {}).items():
            macros[f"clapp{e}{k}"] = fp(d["p_holm"])

    # ---------------- Table: CLAP removal (drop) and damage to the kept source (keep), E1 and C1
    lines = ["\\begin{tabular}{lrrrr}", "\\toprule",
             "& \\multicolumn{2}{c}{E1} & \\multicolumn{2}{c}{C1}\\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
             "System & removed & kept & removed & kept\\\\", "\\midrule"]
    for k, nm in SYS:
        cells = []
        for e in ("E1", "C1"):
            cells.append(f2(vmed(k, e, "drop"), 3))
            cells.append(f2(vmed(k, e, "keep"), 3))
        if any(c != "--" for c in cells):
            lines.append(nm + " & " + " & ".join(cells) + "\\\\")
    lines.append("\\midrule")
    ref = V["reference"]
    lines.append("Ground truth & " + " & ".join(
        f2(ref.get(e, {}).get(m, {}).get("median"), 3) for e in ("E1", "C1") for m in ("drop", "keep")) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    w("tab_keep.tex", "\n".join(lines) + "\n")
    for e in ("E1", "C1"):
        for k, _ in SYS:
            for m in ("drop", "keep"):
                v = vmed(k, e, m)
                if v is not None:
                    macros[f"{m}{e}{k}"] = f2(v, 3)
        macros[f"dropref{e}"] = f2(ref.get(e, {}).get("drop", {}).get("median"), 3)

    # ---------------- Table: room (RT60 error and EDC error), E4 and C3
    lines = ["\\begin{tabular}{lrrrr}", "\\toprule",
             "& \\multicolumn{2}{c}{E4 ($n$=" + str(V["medians"]["ours"]["E4"]["rt60_off_err"]["n"]) + ")} & "
             "\\multicolumn{2}{c}{C3 ($n$=" + str(V["medians"]["ours"]["C3"]["rt60_off_err"]["n"]) + ")}\\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
             "System & RT60 err (s) & EDC err (dB) & RT60 err (s) & EDC err (dB)\\\\", "\\midrule"]
    colsr = {(e, m): [vmed(k, e, m) for k, _ in SYS] for e in ("E4", "C3") for m in ("rt60_off_err", "edc_err")}
    cbr = {key: (shared_col if key[0] == "E4" else bold_best)(v, False)
           for key, v in colsr.items()}
    for j, (k, nm) in enumerate(SYS):
        cells = [cbr[(e, m)][j] for e in ("E4", "C3") for m in ("rt60_off_err", "edc_err")]
        if any(c != "--" for c in cells):
            lines.append(nm + " & " + " & ".join(cells) + "\\\\")
    W = J("p3_r2_metrics_wpe.json")["system"]
    WL = J("p3_r2_wpe.json")["edits"]
    lines.append("WPE + target response & " + " & ".join(
        f2(W.get(e, {}).get(m, {}).get("median")) for e in ("E4", "C3") for m in ("rt60_off_err", "edc_err")) + "\\\\")
    lines.append("\\midrule")
    lines.append("Unedited mixture & " + " & ".join(
        f2(V["anchor"].get(e, {}).get(m, {}).get("median")) for e in ("E4", "C3") for m in ("rt60_off_err", "edc_err")) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    w("tab_room.tex", "\n".join(lines) + "\n")
    for e in ("E4", "C3"):
        macros[f"si{e}wpe"] = f2(WL[e]["sisdr"]["median"])
        macros[f"mr{e}wpe"] = f2(WL[e]["mrstft"]["median"])
        macros[f"edc{e}wpe"] = f2(W[e]["edc_err"]["median"])
        macros[f"rt{e}wpe"] = f2(W[e]["rt60_off_err"]["median"])
        for k in ("x5", "x6", "ours"):
            d = V["medians"].get(k, {}).get(e, {}).get("rt60_off_err")
            if d is not None:
                macros[f"nrt{e}{k}"] = str(d["n"])
    for e in ("E4", "C3"):
        for m, short in (("rt60_off_err", "rt"), ("edc_err", "edc")):
            macros[f"{short}anchor{e}"] = f2(V["anchor"].get(e, {}).get(m, {}).get("median"))
            for k, _ in SYS:
                v = vmed(k, e, m)
                if v is not None:
                    macros[f"{short}{e}{k}"] = f2(v)
    rv = V.get("rt60_validation") or {}
    macros["rtvalid"] = json.dumps(rv)[:0]  # placeholder, text written by hand from the ledger

    # ---------------- Table: locality (E1..E3 and chains), identity systems only
    lines = ["\\begin{tabular}{lrrrrrrr}", "\\toprule",
             "System & E1 & E2 & E3 & C1 & C2 & C3 & C4\\\\", "\\midrule"]
    locE = ["E1", "E2", "E3"] + CHAINS
    cols = {e: [med(k, e, "loc") for k, _ in SYS[:5]] for e in locE}
    cb = {e: ([f2(v) for v in cols[e]] if e in ("C1", "C4")
              else bold_best(cols[e], True)) for e in locE}
    for j, (k, nm) in enumerate(SYS[:5]):
        lines.append(nm + " & " + " & ".join(cb[e][j] for e in locE) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    w("tab_locality.tex", "\n".join(lines) + "\n")
    for e in locE:
        for k, _ in SYS[:5]:
            v = med(k, e, "loc")
            if v is not None:
                macros[f"loc{e}{k}"] = f2(v)

    # ---------------- Table: listening test (kept listeners, median of listener means)
    subs = {}
    for f in sorted(glob.glob(str(RES / "*.json"))):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        key = d["listener"].strip().lower()
        if key in subs and subs[key]["started"] <= d["started"]:
            continue
        subs[key] = d
    kept = []
    for d in subs.values():
        bad = [a["ratings"].get("gt", 100) < a["ratings"].get("orig", 0) for a in d["answers"]]
        if np.mean(bad) <= 0.3:
            kept.append(d)
    R = {}
    for d in kept:
        for a in d["answers"]:
            for c, v in a["ratings"].items():
                R.setdefault((a["edit"], c), {}).setdefault(d["listener"], []).append(v)
    conds = [("gt", "Ground truth"), ("ours", "Ours"), ("x1", "AudioSep"), ("x3", "SoloAudio"),
             ("x6", "Audio-Omni"), ("orig", "Unedited mixture")]
    from scipy.stats import wilcoxon
    lines = ["\\begin{tabular}{lrrrr}", "\\toprule",
             "Clip & E1 & E2 & E3 & E4\\\\", "\\midrule"]
    for c, nm in conds:
        cells = []
        for e in EDITS:
            if (e, c) not in R:
                cells.append("--"); continue
            x = [np.mean(v) for v in R[(e, c)].values()]
            cells.append(f"{np.median(x):.0f}")
        lines.append(nm + " & " + " & ".join(cells) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    w("tab_listen.tex", "\n".join(lines) + "\n")
    macros["nListeners"] = str(len(subs))
    macros["nKept"] = str(len(kept))
    for e in EDITS:
        for c, _ in conds:
            if (e, c) in R:
                x = [np.mean(v) for v in R[(e, c)].values()]
                macros[f"lt{e}{c}"] = f"{np.median(x):.0f}"
        for c, _ in conds:
            for ref in ("ours", "gt", "orig"):
                if c != ref and (e, c) in R and (e, ref) in R:
                    ls = [l for l in R[(e, c)] if l in R[(e, ref)]]
                    x = np.array([np.mean(R[(e, c)][l]) for l in ls])
                    y = np.array([np.mean(R[(e, ref)][l]) for l in ls])
                    p = wilcoxon(x, y).pvalue if np.any(x != y) else 1.0
                    macros[f"ltp{e}{c}vs{ref}"] = fp(p)
    # bootstrap intervals of the paired median differences (p3_r2_listening_ci.py)
    CI = J("p3_r2_listening_ci.json")["pairs"]
    for e, d in CI.items():
        for key, v in d.items():
            a, b = key.split("_vs_")
            lo, hi = v["ci95"]
            macros[f"lc{e}{a}vs{b}"] = (f"{f2(v['median_diff'], 0)}, 95\\% interval "
                                       f"{f2(lo, 0)} to {f2(hi, 0)}")

    # ---------------- order test
    ot = V.get("order_test", {})
    if ot:
        macros["orderTest"] = f"{max(float(v['max_abs_diff']) for v in ot.values()):.0e}".replace("e-", "\\times10^{-").replace("+", "") + "}"

    def texname(k):
        digits = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
                  "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"}
        return "".join(digits.get(ch, ch) for ch in k if ch.isalnum())
    body = "\n".join(f"\\newcommand{{\\r{texname(k)}}}{{{v}}}" for k, v in macros.items())
    w("macros_r2.tex", body + "\n")


if __name__ == "__main__":
    main()
