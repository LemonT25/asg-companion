"""Main-table ledger for the revision (T-ASL-13354-2026 R1).

Reads paper_numbers/p3_r2_<sys>.json rows for ours and X1..X6, the
listening stats (source count per scene) and the saved reference wavs,
and writes paper_numbers/p3_r2_main_table.json with
  anchor  : SI-SDR and mrSTFT of the unedited mixture to the reference
            (E1..E4, C1..C4; E3 restricted to K >= 2)
  medians : per edit / metric / system (E3 restricted to K >= 2)
  paired  : scene-paired Wilcoxon vs ours, Holm over external systems
Run with /workspace/venvs/audiocf python.
"""
import json
import sys

import numpy as np
import soundfile as sf
import torch
from scipy.stats import wilcoxon

sys.path.insert(0, "/workspace/CF_ASG/src")
import p3_r2_common as C  # noqa: E402

PN = C.PN
SYS = [("ours", "Ours"), ("x1", "AudioSep"), ("x2", "FlowSep"),
       ("x3", "SoloAudio"), ("x4", "OmniSep"), ("x5", "ZETA"),
       ("x6", "Audio-Omni")]
EDITS = ["E1", "E2", "E3", "E4"]
CHAINS = ["C1", "C2", "C3", "C4"]


# Standard multi-resolution STFT loss (Yamamoto et al., ICASSP 2020;
# official kan-bayashi implementation): spectral convergence + log STFT
# magnitude L1, FFT 1024/2048/512, hop 120/240/50, win 600/1200/240,
# Hann windows, magnitude clamped at sqrt(1e-7). Replaces the training
# loss variant (linear + log L1, floor 1e-5) used in the first ledgers;
# see PROTOCOL_P3.md amendment (e).
STD_RES = [(1024, 120, 600), (2048, 240, 1200), (512, 50, 240)]


def _mag(x, n, h, w):
    S = torch.stft(x, n, h, w, window=torch.hann_window(w),
                   return_complex=True)
    return torch.sqrt(torch.clamp(S.real ** 2 + S.imag ** 2, min=1e-7))


def mrstft_std(a, b):
    x = torch.tensor(a, dtype=torch.float32)[None]
    y = torch.tensor(b, dtype=torch.float32)[None]
    L = min(x.shape[-1], y.shape[-1])
    x, y = x[:, :L], y[:, :L]
    sc = lm = 0.0
    for n, h, w in STD_RES:
        X, Y = _mag(x, n, h, w), _mag(y, n, h, w)
        sc += (torch.norm(Y - X, p="fro") / torch.norm(Y, p="fro")).item()
        lm += torch.nn.functional.l1_loss(torch.log(Y), torch.log(X)).item()
    return (sc + lm) / 3


def sisdr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    s = (a @ b) / (b @ b + 1e-12) * b
    return float(10 * np.log10((s @ s) / (((a - s) @ (a - s)) + 1e-12)))


def holm(ps):
    idx = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(idx):
        run = max(run, (m - r) * ps[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def main():
    stats = {int(k): v for k, v in json.load(
        open(PN / "p3_r2_listening_stats.json")).items()}
    L = {k: json.load(open(PN / f"p3_r2_{k}.json"))["rows"] for k, _ in SYS}
    torch.set_num_threads(4)
    # standard mrSTFT per scene and system, from the saved waveforms
    std = {}
    for e in EDITS + CHAINS:
        for k, _ in SYS:
            for r in L[k].get(e, []):
                i = r["scene"]
                po = C.R2_OUT / k / f"{i:05d}_{e}.wav"
                pr = C.R2_REF / f"{i:05d}_{e}_ref.wav"
                if po.exists() and pr.exists():
                    std.setdefault(e, {}).setdefault(k, {})[i] = mrstft_std(
                        sf.read(po)[0], sf.read(pr)[0])
        print("std mrSTFT", e, {k: len(v) for k, v in std.get(e, {}).items()},
              flush=True)
    out = {"meta": {"date": "2026-10-21", "script": "src/p3_r2_main_table.py",
                    "note": "E3 restricted to K>=2 scenes (single-source "
                            "distance edit is a pure gain, SI-SDR scale "
                            "invariant); mrSTFT = standard Parallel WaveGAN "
                            "definition (spectral convergence + log "
                            "magnitude, three resolutions), recomputed from "
                            "the saved waveforms; anchors = SI-SDR and mrSTFT of "
                            "the unedited mixture to the reference; "
                            "paired = scene-paired Wilcoxon vs ours, Holm "
                            "over external systems per (edit, metric)"},
           "anchor": {}, "medians": {}, "paired": {}}
    for e in EDITS + CHAINS:
        v, vm = [], []
        for r in L["ours"][e]:
            i = r["scene"]
            if e == "E3" and stats[i]["K"] < 2:
                continue
            pm = C.R2_REF / f"{i:05d}_mix.wav"
            pr = C.R2_REF / f"{i:05d}_{e}_ref.wav"
            if pm.exists() and pr.exists():
                a, b = sf.read(pm)[0], sf.read(pr)[0]
                v.append(sisdr(a, b))
                vm.append(mrstft_std(a, b))
        out["anchor"][e] = {"sisdr_median": float(np.median(v)),
                            "mrstft_median": float(np.median(vm)),
                            "n": len(v)}
    print("anchors", {e: (round(d["sisdr_median"], 2),
                          round(d["mrstft_median"], 2), d["n"])
                      for e, d in out["anchor"].items()})
    for e in EDITS + CHAINS:
        for met, hb in (("sisdr", True), ("mrstft", False), ("loc", True)):
            if met == "mrstft":
                rows = {k: {i: v for i, v in std.get(e, {}).get(k, {}).items()
                            if not (e == "E3" and stats[i]["K"] < 2)}
                        for k, _ in SYS}
            else:
                rows = {k: {r["scene"]: r[met] for r in L[k].get(e, [])
                            if r.get(met) is not None
                            and not (e == "E3" and stats[r["scene"]]["K"] < 2)}
                        for k, _ in SYS}
            med = {k: {"median": float(np.median(list(v.values()))),
                       "n": len(v)} for k, v in rows.items() if v}
            out["medians"].setdefault(e, {})[met] = med
            res, ps, ks = {}, [], []
            for k, _ in SYS[1:]:
                com = sorted(set(rows["ours"]) & set(rows[k]))
                if len(com) < 8:
                    continue
                x = np.array([rows["ours"][s] for s in com])
                y = np.array([rows[k][s] for s in com])
                d = x - y
                p = float(wilcoxon(x, y).pvalue) if np.any(d != 0) else 1.0
                res[k] = {"median_diff": float(np.median(d)), "p": p,
                          "n": len(com)}
                ps.append(p)
                ks.append(k)
            for k, pa in zip(ks, holm(ps)):
                res[k]["p_holm"] = pa
            out["paired"].setdefault(e, {})[met] = res
            if met == "sisdr":
                print(e, {k: (round(v["median"], 2), v["n"])
                          for k, v in med.items()})
                print("   paired vs ours:",
                      {k: (round(v["median_diff"], 2), "%.2g" % v["p_holm"],
                           v["n"]) for k, v in res.items()})
    # source-count breakdown kept from the previous ledger if present
    old = PN / "p3_r2_main_table.json"
    if old.exists():
        prev = json.load(open(old))
        if "K_breakdown" in prev:
            out["K_breakdown"] = prev["K_breakdown"]
    out["mrstft_std_rows"] = {e: {k: {str(i): v for i, v in d.items()}
                                  for k, d in sd.items()}
                              for e, sd in std.items()}
    json.dump(out, open(old, "w"), indent=1)
    print("wrote", old)


if __name__ == "__main__":
    main()
