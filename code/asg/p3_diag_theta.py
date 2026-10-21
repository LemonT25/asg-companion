"""Diagnostic: can a physics-inversion line-search beat the learned
theta heads?  theta_hat = argmin_v || Phi^-1_v(stem) - dry_prior ||.
Wrong v leaves a residual Doppler chirp in the inverted dry, which
mismatches the (chirp-free) prior.  Variants:
  A  GT stem,        GT dry prior   (sanity upper bound)
  B  GT stem,        learned d_hat  (graph slot decoder)
  C  posterior stem, learned d_hat  (deployable)
  D  posterior stem, GT dry         (stem-quality ceiling for C)
Baselines: est head on sep stems (0.331 rel), graph head (0.247 rel).
-> paper_numbers/p3_diag_theta_v0.json
"""
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/workspace/CF_ASG/src")
from p3_train import (SceneSet, SceneModel, theta_denorm_t, SEG)  # noqa
from p3_edit_eval import (si_sdr, mrstft_np, inverse_passby,  # noqa
                          posterior_stems, pit_by_sisdr, graph_infer)

ROOT = Path("/workspace/CF_ASG")
CK = ROOT / "p3" / "ckpt"
PN = ROOT / "paper_numbers"
rng = np.random.default_rng(1)
V_GRID = np.linspace(5.0, 40.0, 36)


def linesearch(stem, prior):
    costs = []
    for v in V_GRID:
        d = inverse_passby(stem, float(v))
        costs.append(mrstft_np(d[:SEG], prior[:SEG]))
    return float(V_GRID[int(np.argmin(costs))])


def centroid_flatness(d, sr=16000, nfft=1024, hop=256):
    """Energy-weighted variance of the log-spectral-centroid track.
    Doppler rescales the whole spectrum by a time-varying factor, so a
    residual (wrong-v) warp adds a systematic drift to the log-centroid;
    the correct inverse minimises it.  Gain modulation is invariant."""
    w = np.hanning(nfft)
    n = 1 + (len(d) - nfft) // hop
    f = np.fft.rfftfreq(nfft, 1.0 / sr) + 1e-6
    cents, ens = [], []
    for t in range(n):
        seg = d[t * hop:t * hop + nfft] * w
        P = np.abs(np.fft.rfft(seg)) ** 2
        e = P.sum()
        if e <= 0:
            continue
        cents.append(np.log((P * f).sum() / e))
        ens.append(e)
    cents, ens = np.array(cents), np.array(ens)
    if len(cents) < 4:
        return np.inf
    wgt = ens / ens.sum()
    mu = (wgt * cents).sum()
    return float((wgt * (cents - mu) ** 2).sum())


def linesearch_flat(stem):
    costs = [centroid_flatness(inverse_passby(stem, float(v))[:SEG])
             for v in V_GRID]
    return float(V_GRID[int(np.argmin(costs))])


def spectral_selfconsistency(d, nfft=1024, hop=256, norm=False):
    """Full-spectrum generalisation of centroid flatness: the dry
    source's spectral ENVELOPE is time-invariant, so the correct
    inverse minimises the energy-weighted distance between each
    frame's log-spectrum and the weighted mean log-spectrum.
    norm=True removes per-frame gain (unit-energy frames)."""
    w = np.hanning(nfft)
    n = 1 + (len(d) - nfft) // hop
    S, ens = [], []
    for t in range(n):
        seg = d[t * hop:t * hop + nfft] * w
        P = np.abs(np.fft.rfft(seg)) ** 2
        e = P.sum()
        if e <= 1e-14:
            continue
        S.append(P / e if norm else P)
        ens.append(e)
    if len(S) < 4:
        return np.inf
    S = np.log(np.array(S) + 1e-12)
    ens = np.array(ens)
    wgt = ens / ens.sum()
    mu = (wgt[:, None] * S).sum(0)
    return float((wgt * ((S - mu) ** 2).mean(1)).sum())


def linesearch_selfcon(stem, norm=False):
    costs = [spectral_selfconsistency(inverse_passby(stem, float(v))[:SEG],
                                      norm=norm) for v in V_GRID]
    return float(V_GRID[int(np.argmin(costs))])


def frame_purity(stem, yn, nfft=1024, hop=256):
    """Per-frame energy share of the target stem in the mixture."""
    w = np.hanning(nfft)
    n = 1 + (min(len(stem), len(yn)) - nfft) // hop
    pur = np.zeros(n)
    for t in range(n):
        es = np.sum((stem[t * hop:t * hop + nfft] * w) ** 2)
        em = np.sum((yn[t * hop:t * hop + nfft] * w) ** 2)
        pur[t] = es / (em + 1e-14)
    return np.clip(pur, 0.0, 1.0)


def selfcon_pure(d, pur, nfft=1024, hop=256):
    """Self-consistency restricted to target-dominated frames
    (purity>0.5, weight=purity*energy); gain-normalised frames."""
    w = np.hanning(nfft)
    n = 1 + (len(d) - nfft) // hop
    S, wts = [], []
    for t in range(n):
        p = pur[t] if t < len(pur) else 0.0
        if p <= 0.5:
            continue
        seg = d[t * hop:t * hop + nfft] * w
        P = np.abs(np.fft.rfft(seg)) ** 2
        e = P.sum()
        if e <= 1e-14:
            continue
        S.append(P / e)
        wts.append(p * e)
    if len(S) < 4:
        return np.inf
    S = np.log(np.array(S) + 1e-12)
    wts = np.array(wts)
    wgt = wts / wts.sum()
    mu = (wgt[:, None] * S).sum(0)
    return float((wgt * ((S - mu) ** 2).mean(1)).sum())


def linesearch_pure(stem, yn):
    pur = frame_purity(stem, yn)
    costs = [selfcon_pure(inverse_passby(stem, float(v))[:SEG], pur)
             for v in V_GRID]
    return float(V_GRID[int(np.argmin(costs))])


def posterior_stems_hard(yn, stems, p=2.0):
    """Hardened Wiener masks (exponent p on the power ratio): more
    cross-talk suppression, more distortion -- fine for parameter
    estimation, not meant for listening."""
    t = torch.tensor(np.stack([yn[:SEG]] + [s[:SEG] for s in stems]),
                     dtype=torch.float32)
    w = torch.hann_window(1024)
    Z = torch.stft(t, 1024, 256, window=w, return_complex=True)
    P = (Z[1:].abs() ** 2) ** p
    M = P / (P.sum(0, keepdim=True) + 1e-12)
    post = torch.istft(M * Z[0][None], 1024, 256, window=w, length=SEG)
    return [x.numpy().astype(np.float64) for x in post]


def main():
    torch.set_num_threads(8)
    dev = "cuda"
    gmodel = SceneModel("graph").to(dev)
    gmodel.load_state_dict(torch.load(CK / "graph_latest.pt",
                                      map_location=dev,
                                      weights_only=False)["model"])
    gmodel.eval()
    smodel = SceneModel("sep").to(dev)
    smodel.load_state_dict(torch.load(CK / "sep_latest.pt",
                                      map_location=dev,
                                      weights_only=False)["model"])
    smodel.eval()

    ds = SceneSet("val")
    idxs = rng.choice(len(ds), size=200, replace=False)
    R = {k: [] for k in "ABCDEFGHIJKLMN"}
    n_used = 0
    for i in idxs:
        if n_used >= 40:
            break
        y, stems, dries, exist, traj, thn, alpha, env, K = ds[int(i)]
        K = int(K)
        pb = [k for k in range(K) if int(traj[k]) == 0]
        if not pb or float(env) > 0.5:   # open scenes (clean passby)
            continue
        n_used += 1
        yn = y.numpy().astype(np.float64)
        gts = stems.numpy().astype(np.float64)
        gtd = dries.numpy().astype(np.float64)
        # graph dry prior + sep posterior stems
        d1, e1, trj1, tn1, al1, env1, st1 = graph_infer(gmodel, y, dev)
        with torch.no_grad():
            dsep, esep, _, _, _, _ = smodel.slots(y[None].to(dev))
            sepst = dsep[0, ..., :SEG].cpu().numpy().astype(np.float64)
        po = posterior_stems(yn, [sepst[k] for k in range(3)])
        poh = posterior_stems_hard(yn, [sepst[k] for k in range(3)])
        pg = pit_by_sisdr([st1[k].numpy() for k in range(3)], gts, K)
        pp = pit_by_sisdr(po, gts, K)
        for k in pb:
            gt_v = float(theta_denorm_t(traj[k][None], thn[k][None]))
            dh = d1[pg[k]].numpy().astype(np.float64)
            for tag, stem, prior in (
                    ("A", gts[k], gtd[k]),
                    ("B", gts[k], dh),
                    ("C", po[pp[k]], dh),
                    ("D", po[pp[k]], gtd[k])):
                vh = linesearch(stem, prior)
                R[tag].append({"gt": gt_v, "vh": vh,
                               "rel": abs(vh - gt_v) / gt_v})
            for tag, stem in (("E", po[pp[k]]), ("F", gts[k])):
                vh = linesearch_flat(stem)
                R[tag].append({"gt": gt_v, "vh": vh,
                               "rel": abs(vh - gt_v) / gt_v})
            # I/J: prior-free spectral self-consistency (I raw, J
            # gain-normalised frames); K/L: same on GT stems
            for tag, stem, nr in (("I", po[pp[k]], False),
                                  ("J", po[pp[k]], True),
                                  ("K", gts[k], False),
                                  ("L", gts[k], True),
                                  ("M", poh[pp[k]], True),
                                  ("N", poh[pp[k]], False)):
                vh = linesearch_selfcon(stem, norm=nr)
                R[tag].append({"gt": gt_v, "vh": vh,
                               "rel": abs(vh - gt_v) / gt_v})
            # P: self-consistency on the RAW sep stem (network's own
            # nonlinear interference suppression, no mask bleed)
            vh = linesearch_selfcon(sepst[pp[k]], norm=True)
            R.setdefault("P", []).append(
                {"gt": gt_v, "vh": vh, "rel": abs(vh - gt_v) / gt_v})
            vh = linesearch_selfcon(sepst[pp[k]], norm=False)
            R.setdefault("P2", []).append(
                {"gt": gt_v, "vh": vh, "rel": abs(vh - gt_v) / gt_v})
            # O: purity-weighted self-consistency on posterior stems
            vh = linesearch_pure(po[pp[k]], yn)
            R.setdefault("O", []).append(
                {"gt": gt_v, "vh": vh, "rel": abs(vh - gt_v) / gt_v})
            # G/H: prior-quality sensitivity -- GT dry corrupted with
            # white noise to 5 / 10 dB SNR (proxy for a learned dry
            # predictor of that quality)
            for tag, snr in (("G", 5.0), ("H", 10.0)):
                nz = rng.standard_normal(len(gtd[k]))
                nz *= np.linalg.norm(gtd[k]) / (10 ** (snr / 20)
                                                * np.linalg.norm(nz))
                vh = linesearch(po[pp[k]], gtd[k] + nz)
                R[tag].append({"gt": gt_v, "vh": vh,
                               "rel": abs(vh - gt_v) / gt_v})
    out = {"meta": {"date": str(datetime.date.today()),
                    "script": "src/p3_diag_theta.py",
                    "grid": [float(V_GRID[0]), float(V_GRID[-1]),
                             len(V_GRID)],
                    "n_slots": len(R["A"]), "n_scenes": n_used},
           "rel_median": {t: float(np.median([r["rel"] for r in R[t]]))
                          for t in R if R[t]},
           "rel_mean": {t: float(np.mean([r["rel"] for r in R[t]]))
                        for t in R if R[t]},
           "rows": R}
    (PN / "p3_diag_theta_v0.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("rel_median", "rel_mean")},
                     indent=1))


if __name__ == "__main__":
    main()
