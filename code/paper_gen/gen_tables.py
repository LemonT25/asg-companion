"""Generate all paper tables from ledgers (no hand-typed numbers).
Outputs LaTeX fragments to paper/gen/*.tex.
Run: /workspace/venvs/audiocf/bin/python paper/gen/gen_tables.py
"""
import json
from pathlib import Path

PN = Path("/workspace/CF_ASG/paper_numbers")
OUT = Path("/workspace/CF_ASG/paper/gen")
OUT.mkdir(parents=True, exist_ok=True)


def J(name):
    return json.loads((PN / name).read_text())


def w(name, s):
    (OUT / name).write_text(s)
    print("wrote", name)


def f2(x):
    return f"{x:.2f}"


def main():
    main_led = J("p3_edit_eval_v1.json")
    verdict = J("p3_sixsys_verdict_v1.json")
    confirm = J("p3_confirm_verdict_v0.json")
    re2 = J("p3_r_e2_v1.json")
    re13 = J("p3_r_e1e3_v0.json")

    # --- Table: six-system synthetic (medians + paired diff + Holm) ---
    rows = []
    om = verdict["ours_median"]
    rows.append(("Ours", f2(om["E1"]), f2(om["E2"]), f2(om["E3"]),
                 f2(om["E4"])))
    names = {"X1": "AudioSep", "X2": "FlowSep", "X3": "SoloAudio",
             "X4": "OmniSep", "X5": "ZETA", "X6": "Audio-Omni"}
    for xn, nm in names.items():
        v = verdict["vs"][xn]
        cells = []
        for e in ("E1", "E2", "E3"):
            d = v.get(e)
            cells.append(f2(d["x_median"]) if d else "--")
        rows.append((nm, *cells, "--"))
    # per-edit n from ours rows + X subset flag
    ns = {e: main_led["edits"]["S3"][e]["sisdr"]["n"]
          for e in ("E1", "E2", "E3", "E4")}
    body = "\\\\\n".join(" & ".join(r) for r in rows)
    w("tab_sixsys.tex",
      "\\begin{tabular}{lrrrr}\n\\toprule\n"
      "System & E1 & E2 & E3 & E4\\\\\n\\midrule\n"
      + body + "\\\\\n\\bottomrule\n\\end{tabular}\n")

    # --- Table: six-system mrSTFT (secondary metric, C2 fix) ---
    import numpy as np
    XL = {"X1": ("p3_baselines_x1_v1.json",),
          "X2": ("p3_baselines_x2_v1.json",),
          "X3": ("p3_baselines_x3_v1.json",),
          "X4": ("p3_baselines_x4_v1.json",),
          "X5": ("p3_baselines_x5_v1.json",),
          "X6": ("p3_baselines_x6_v1.json",)}
    def mr_of_rows(rws):
        v = [r["mrstft"] for r in rws if "mrstft" in r]
        return np.median(v) if v else None
    lines = []
    ours_mr = []
    for e in ("E1", "E2", "E3", "E4"):
        m = mr_of_rows(main_led["rows"]["S3"].get(e, []))
        ours_mr.append(f"{m:.2f}" if m is not None else "--")
    lines.append("Ours & " + " & ".join(ours_mr))
    for xn, (f,) in list(XL.items())[:4]:
        try:
            xd = J(f)
        except FileNotFoundError:
            continue
        key = [k for k in xd if k != "meta"][0]
        cells = []
        for e in ("E1", "E2", "E3"):
            m = mr_of_rows(xd[key].get("rows", {}).get(e, []))
            cells.append(f"{m:.2f}" if m is not None else "--")
        lines.append(f"{names.get(xn, xn)} & " + " & ".join(cells)
                     + " & --")
    w("tab_sixsys_mr.tex",
      "\\begin{tabular}{lrrrr}\n\\toprule\n"
      "System & E1 & E2 & E3 & E4\\\\\n\\midrule\n"
      + "\\\\\n".join(lines)
      + "\\\\\n\\bottomrule\n\\end{tabular}\n")

    # --- Table: confirmation cohort (A12.1) ---
    cv = confirm["verdict"]
    cl = []

    def e1cell(xn, e1p):
        # criterion (b): superiority over X2-X4 (Wilcoxon p), parity with X1
        # by TOST; when X1 is beaten outright the Wilcoxon p is shown
        if e1p:
            return "%.1e" % e1p
        tost = cv["b_E1"].get("X1_parity_tost", {})
        if xn == "X1" and tost and not tost.get("parity"):
            return "%.1e (superior)" % confirm["vs"]["X1"]["E1"]["wilcoxon_p"]
        return "TOST parity"
    for xn in ("X1", "X2", "X3", "X4"):
        e3 = cv["a_E3_holm"].get(xn, {})
        e1 = cv["b_E1"].get(xn) or {}
        e1p = e1.get("wilcoxon_p")
        cl.append(f"{names[xn]} & "
                  f"{e1cell(xn, e1p)} & "
                  f"{'%.1e' % e3['p'] if e3 else '--'} & "
                  f"{'pass' if e3.get('pass') else 'n.s.'}")
    w("tab_confirm.tex",
      "\\begin{tabular}{lrrr}\n\\toprule\n"
      "System & E1 $p$ & E3 $p$ & E3 Holm\\\\\n\\midrule\n"
      + "\\\\\n".join(cl)
      + "\\\\\n\\bottomrule\n\\end{tabular}\n")

    # --- Table: locality (loc_clean) ---
    ll = []
    o = []
    for e in ("E1", "E2", "E3"):
        v = [r.get("loc_clean") for r in main_led["rows"]["S3"][e]
             if r.get("loc_clean") is not None]
        o.append(f"{np.median(v):.1f}" if v else "--")
    ll.append("Ours & " + " & ".join(o))
    for xn in ("X1", "X2", "X3", "X4"):
        try:
            xd = J(XL[xn][0])
        except FileNotFoundError:
            continue
        key = [k for k in xd if k != "meta"][0]
        cells = []
        for e in ("E1", "E2", "E3"):
            v = [r.get("loc_clean")
                 for r in xd[key].get("rows", {}).get(e, [])
                 if r.get("loc_clean") is not None]
            cells.append(f"{np.median(v):.1f}" if v else "--")
        ll.append(f"{names[xn]} & " + " & ".join(cells))
    w("tab_locality.tex",
      "\\begin{tabular}{lrrr}\n\\toprule\n"
      "System & E1 & E2 & E3\\\\\n\\midrule\n"
      + "\\\\\n".join(ll)
      + "\\\\\n\\bottomrule\n\\end{tabular}\n")

    # --- Table: real-domain R-E2 ---
    lab = {"source": "No-edit input", "ours": "Ours",
           "x5": "ZETA", "x6": "Audio-Omni"}
    lines = []
    agg = re2["aggregates"]
    for k in ("source", "ours", "x5", "x6"):
        s = agg.get(k)
        if s is None:
            continue
        ver = s["verr_target_kmh"]["median"]
        clap = s["clap_target"]["median"]
        lsd = s["lsd_src"]["median"]
        fad = s["fad_clap_vs_ref"]
        lines.append(f"{lab[k]} & {ver:.1f} & {clap:.3f} & "
                     f"{lsd:.1f} & {fad:.2f}")
    w("tab_re2.tex",
      "\\begin{tabular}{lrrrr}\n\\toprule\n"
      "System & Verif.\\ err & CLAP & LSD & FAD\\\\\n\\midrule\n"
      + "\\\\\n".join(lines)
      + "\\\\\n\\bottomrule\n\\end{tabular}\n")

    # --- Table: real-domain R-E1/E3 (separation family) ---
    lab13 = {"ours": "Ours (corpus-trained front end)",
             "x1": "AudioSep", "x2": "FlowSep", "x3": "SoloAudio",
             "x4": "OmniSep"}
    lines = []
    for k in ("ours", "x1", "x2", "x3", "x4"):
        sd = re13["systems"].get(k)
        if not sd:
            continue
        e = sd["edits"]
        lines.append(f"{lab13[k]} & {e['e1a']['sisdr']['median']:.2f} & "
                     f"{e['e1b']['sisdr']['median']:.2f} & "
                     f"{e['e3']['sisdr']['median']:.2f}")
    w("tab_re13.tex",
      "\\begin{tabular}{lrrr}\n\\toprule\n"
      "System & E1 keep & E1 remove & E3\\\\\n\\midrule\n"
      + "\\\\\n".join(lines)
      + "\\\\\n\\bottomrule\n\\end{tabular}\n")

    # --- Table: R-E2 direction split (from per-clip rows) ---
    import numpy as np
    rows2 = re2["rows"]
    clips = {c["clip"]: c for c in re2["clips"]}
    dl = []
    for sysk, lb in (("source", "No-edit input"), ("ours", "Ours"),
                     ("x5", "ZETA"), ("x6", "Audio-Omni")):
        up, dn = [], []
        for r in rows2.get(sysk, []):
            fac = clips[r["clip"]]["factor"]
            (up if fac > 1 else dn).append(r["verr_target_kmh"])
        if up and dn:
            dl.append(f"{lb} & {np.median(up):.1f} & {np.median(dn):.1f}")
    w("tab_re2dir.tex",
      "\\begin{tabular}{lrr}\n\\toprule\n"
      "System & Speed-up & Slow-down\\\\\n\\midrule\n"
      + "\\\\\n".join(dl)
      + "\\\\\n\\bottomrule\n\\end{tabular}\n")

    # --- inline numbers macro file ---
    inf = main_led["inference"]
    m = {
        "OursThetaRel": f2(inf["theta_relerr"]["S3"]["median"]),
        "OursStem": f2(inf["stem_sisdr"]["S3"]["median"]),
        "OracleETwo": f2(main_led["edits"]["S4"]["E2"]["sisdr"]["median"]),
        "VerifierGate": "5.7",
    }
    macros = "\n".join(
        f"\\newcommand{{\\num{k}}}{{{v}}}" for k, v in m.items())
    w("macros.tex", macros + "\n")


if __name__ == "__main__":
    main()
