"""A14 metrics on the saved waveforms (p3/r2_out/<system>/, p3/r2_ref/).

Per system ledger p3_r2_<system>.json (rows + specs) this adds:
  * CLAP full-reference audio similarity cos(a(out), a(ref)) for every
    edit and chain (semantic companion to SI-SDR)
  * E1 semantic removal: drop = s_rm(mix) - s_rm(out), keep = s_keep(out)
    - s_keep(mix); reference ceiling and unedited anchor recorded
  * E4 (and C3) reverberation: |RT60(out) - RT60(ref)| by (a) the
    offset-isolated Schroeder T20 on the free decay after the last dry
    offset and (b) a blind decay-rate estimate; validation of (a)/(b)
    on the references against the RIR Schroeder RT60 of alpha'
Also scores the unedited mixture (anchor) and the reference (ceiling)
with the same functions so every table has the two rungs.
-> paper_numbers/p3_r2_metrics_<system>.json
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from scipy.signal import stft

sys.path.insert(0, "/workspace/CF_ASG/src")
import p3_r2_common as C  # noqa: E402
from p3_r2_common import SceneSet, load_scene, q_of, agg  # noqa: E402

PN = C.PN
SR = C.SR
CLAP_ID = "laion/clap-htsat-unfused"
RIR_RT60 = {}


# ------------------------------------------------------------ RT60
def rir_rt60(alpha):
    """Schroeder T20 of the renderer's own RIR for alpha (cached)."""
    if alpha not in RIR_RT60:
        from p2_common import phi_reverb, _RIR_CACHE
        phi_reverb(np.zeros(10), alpha)
        RIR_RT60[alpha] = schroeder_t20(_RIR_CACHE[alpha], SR)
    return RIR_RT60[alpha]


def schroeder_t20(x, sr, lo=-5.0, hi=-25.0):
    e = np.asarray(x, np.float64) ** 2
    if e.sum() <= 0:
        return None
    edc = np.cumsum(e[::-1])[::-1]
    db = 10 * np.log10(edc / edc[0] + 1e-30)
    t = np.arange(len(x)) / sr
    m = (db <= lo) & (db >= hi)
    if m.sum() < 8:
        return None
    p = np.polyfit(t[m], db[m], 1)
    if p[0] >= 0:
        return None
    return float(-60.0 / p[0])


def last_offset(gtd, K, thr_db=-40.0):
    """Last sample (over all dry sources) above thr_db re each source's
    own peak -> start of the scene's free decay."""
    off = 0
    for k in range(K):
        d = np.abs(gtd[k])
        pk = d.max() + 1e-12
        idx = np.nonzero(d > pk * 10 ** (thr_db / 20))[0]
        if len(idx):
            off = max(off, int(idx[-1]))
    return off


def rt60_offset(x, off, sr=SR, min_tail_s=0.3):
    """Offset-isolated T10 (-5..-15 dB, extrapolated to 60 dB) on the
    free decay after the last dry offset. None if the tail is shorter
    than min_tail_s."""
    tail = x[off:]
    if len(tail) < int(min_tail_s * sr):
        return None
    return schroeder_t20(tail, sr, lo=-5.0, hi=-15.0)


def edc_db(x, off, sr=SR, span_s=0.3):
    """Schroeder energy decay curve (dB, normalised at the offset) over
    the first span_s of the free decay; None if the tail is too short."""
    tail = np.asarray(x[off:], np.float64)
    n = int(span_s * sr)
    if len(tail) < n:
        return None
    e = tail ** 2
    edc = np.cumsum(e[::-1])[::-1]
    if edc[0] <= 0:
        return None
    return 10 * np.log10(edc[:n] / edc[0] + 1e-30)


def edc_err(a, b, off):
    """RMS difference (dB) between two decay curves on the same window."""
    ea, eb = edc_db(a, off), edc_db(b, off)
    if ea is None or eb is None:
        return None
    return float(np.sqrt(np.mean((ea - eb) ** 2)))


def rt60_blind(x, sr=SR, nfft=512, hop=128, fmin=200.0, fmax=6000.0,
               win_s=0.25):
    """Blind decay-rate estimate: per-band log-energy envelopes, sliding
    windows of win_s, keep windows whose envelope decreases
    monotonically (free decays), fit slopes, take the 10th percentile of
    the implied RT60s (the fastest consistent decays belong to the room,
    slower ones to source envelopes). Deterministic."""
    f, t, Z = stft(x, sr, nperseg=nfft, noverlap=nfft - hop)
    P = np.abs(Z) ** 2
    band = (f >= fmin) & (f <= fmax)
    L = 10 * np.log10(P[band] + 1e-12)
    W = max(4, int(win_s * sr / hop))
    dt = hop / sr
    rts = []
    for b in range(L.shape[0]):
        row = L[b]
        for s in range(0, len(row) - W, W // 2):
            seg = row[s:s + W]
            if seg[0] - seg[-1] < 10.0:        # need >= 10 dB of decay
                continue
            d = np.diff(seg)
            if (d > 1.0).sum() > W // 8:         # not a free decay
                continue
            slope = np.polyfit(np.arange(W) * dt, seg, 1)[0]
            if slope < -5.0:
                rts.append(-60.0 / slope)
    if len(rts) < 20:
        return None
    return float(np.percentile(rts, 10))


# ------------------------------------------------------------ CLAP
class Clap:
    def __init__(self, dev):
        from transformers import ClapModel, AutoProcessor
        self.dev = dev
        self.model = ClapModel.from_pretrained(CLAP_ID).to(dev).eval()
        self.proc = AutoProcessor.from_pretrained(CLAP_ID)
        self.tcache = {}

    @torch.no_grad()
    def audio(self, x):
        w48 = torchaudio.functional.resample(
            torch.tensor(np.asarray(x, np.float32)), SR, 48000).numpy()
        inp = self.proc(audio=w48, sampling_rate=48000, return_tensors="pt")
        inp = {k: v.to(self.dev) for k, v in inp.items()}
        e = self.model.get_audio_features(**inp)
        e = e.pooler_output if hasattr(e, "pooler_output") else e
        return torch.nn.functional.normalize(e, dim=-1)[0].cpu().numpy()

    @torch.no_grad()
    def text(self, s):
        if s not in self.tcache:
            inp = self.proc(text=[s], return_tensors="pt", padding=True)
            inp = {k: v.to(self.dev) for k, v in inp.items()}
            e = self.model.get_text_features(**inp)
            e = e.pooler_output if hasattr(e, "pooler_output") else e
            self.tcache[s] = torch.nn.functional.normalize(
                e, dim=-1)[0].cpu().numpy()
        return self.tcache[s]


def prompt(cls):
    return f"the sound of {q_of(cls)}"


# ------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--ledger", default=None)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no_clap", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(4)
    dev = "cuda"
    led = json.loads((PN / (args.ledger or f"p3_r2_{args.system}.json"))
                     .read_text())
    specs = led["specs"]
    ds = SceneSet(args.split)
    clap = None if args.no_clap else Clap(dev)
    odir = Path(led["meta"]["wav_dir"])

    def rd(p):
        x, sr = sf.read(str(p), dtype="float64")
        assert sr == SR
        return x

    out_rows = {e: [] for e in C.EDITS}
    anchor_rows = {e: [] for e in C.EDITS}     # unedited mixture
    ref_rows = {e: [] for e in C.EDITS}        # reference ceiling
    rt_valid = []                               # reference RT60 validation
    for spec in specs:
        i = spec["scene"]
        sc = load_scene(ds, i)
        yn, K = sc["yn"], sc["K"]
        refs = C.references(sc, spec)
        a_mix = clap.audio(yn) if clap else None
        for e, (ref, ned) in refs.items():
            p = odir / f"{i:05d}_{e}.wav"
            if not p.exists():
                continue
            est = rd(p)
            row = {"scene": i}
            # waveform rungs: unedited mixture (anchor) and reference
            arow = {"scene": i, "sisdr": C.si_sdr(yn, ref),
                    "mrstft": C.mrstft_np(yn, ref),
                    "loc": C.locality(yn, ref, ned)}
            rrow = {"scene": i, "sisdr": C.si_sdr(ref, ref), "mrstft": 0.0,
                    "loc": C.locality(ref, ref, ned)}
            if clap:
                a_ref = clap.audio(ref)
                a_est = clap.audio(est)
                row["clap_ref"] = float(a_est @ a_ref)
                arow["clap_ref"] = float(a_mix @ a_ref)
                rrow["clap_ref"] = 1.0
                rm = None
                if e == "E1":
                    rm, keep = spec["E1_slot"], None
                elif e == "C1":
                    rm = spec["C1_rm"]
                elif e == "C3":
                    rm = spec["C3_rm"]
                elif e == "C4":
                    rm = spec["C4_rm"]
                if rm is not None:
                    t_rm = clap.text(prompt(sc["classes"][rm]))
                    t_keep = [clap.text(prompt(c)) for j, c in
                              enumerate(sc["classes"]) if j != rm]
                    for name, a_ in (("out", a_est), ("mix", a_mix),
                                     ("ref", a_ref)):
                        s_rm = float(a_ @ t_rm)
                        s_keep = float(np.mean([a_ @ t for t in t_keep]))
                        (row if name == "out" else arow if name == "mix"
                         else rrow)[f"s_rm_{name}"] = s_rm
                        (row if name == "out" else arow if name == "mix"
                         else rrow)[f"s_keep_{name}"] = s_keep
                    s_rm_mix = float(a_mix @ t_rm)
                    s_keep_mix = float(np.mean([a_mix @ t for t in t_keep]))
                    row["drop"] = s_rm_mix - float(a_est @ t_rm)
                    row["keep"] = float(np.mean([a_est @ t for t in t_keep])) \
                        - s_keep_mix
                    arow["drop"], arow["keep"] = 0.0, 0.0
                    rrow["drop"] = s_rm_mix - float(a_ref @ t_rm)
                    rrow["keep"] = float(np.mean([a_ref @ t for t in t_keep])) \
                        - s_keep_mix
            if e in ("E4", "C3"):
                off = last_offset(sc["gtd"], K)
                na = spec["E4_newa"] if e == "E4" else spec["C3_newa"]
                r_ref_o, r_ref_b = rt60_offset(ref, off), rt60_blind(ref)
                r_est_o, r_est_b = rt60_offset(est, off), rt60_blind(est)
                r_mix_o, r_mix_b = rt60_offset(yn, off), rt60_blind(yn)
                row["rt60_off_err"] = (abs(r_est_o - r_ref_o)
                                       if r_est_o and r_ref_o else None)
                row["rt60_blind_err"] = (abs(r_est_b - r_ref_b)
                                         if r_est_b and r_ref_b else None)
                arow["rt60_off_err"] = (abs(r_mix_o - r_ref_o)
                                        if r_mix_o and r_ref_o else None)
                arow["rt60_blind_err"] = (abs(r_mix_b - r_ref_b)
                                          if r_mix_b and r_ref_b else None)
                rrow["rt60_off_err"], rrow["rt60_blind_err"] = 0.0, 0.0
                row["edc_err"] = edc_err(est, ref, off)
                arow["edc_err"] = edc_err(yn, ref, off)
                rrow["edc_err"] = 0.0
                row["rt60_off_out"], row["rt60_off_ref"] = r_est_o, r_ref_o
                row["rt60_blind_out"], row["rt60_blind_ref"] = r_est_b, r_ref_b
                row["free_decay_s"] = (len(yn) - off) / SR
                if e == "E4":
                    rt_valid.append({"scene": i, "alpha_new": na,
                                     "alpha_old": sc["alpha"],
                                     "rir_rt60_new": rir_rt60(na),
                                     "rir_rt60_old": rir_rt60(sc["alpha"]),
                                     "ref_off": r_ref_o, "ref_blind": r_ref_b,
                                     "mix_off": r_mix_o, "mix_blind": r_mix_b,
                                     "free_decay_s": (len(yn) - off) / SR})
            out_rows[e].append(row)
            anchor_rows[e].append(arow)
            ref_rows[e].append(rrow)

    keys = ("sisdr", "mrstft", "loc", "clap_ref", "drop", "keep",
            "rt60_off_err", "rt60_blind_err", "edc_err")

    def summ(rows):
        return {e: {k: agg(rows[e], k) for k in keys
                    if any(k in r and r[k] is not None for r in rows[e])}
                for e in rows if rows[e]}

    # validation of the RT60 estimators on references
    val = {}
    if rt_valid:
        for tag in ("off", "blind"):
            pairs = [(r["rir_rt60_new"], r[f"ref_{tag}"]) for r in rt_valid
                     if r[f"ref_{tag}"] is not None and r["rir_rt60_new"]]
            if len(pairs) >= 5:
                a, b = np.array(pairs).T
                val[tag] = {"n": len(pairs),
                            "spearman": float(np.corrcoef(
                                np.argsort(np.argsort(a)),
                                np.argsort(np.argsort(b)))[0, 1]),
                            "median_abs_err_vs_rir_s": float(
                                np.median(np.abs(a - b))),
                            "by_alpha": {str(al): float(np.median(
                                [bb for aa, bb, r in zip(a, b, [p for p in
                                 rt_valid if p[f"ref_{tag}"] is not None])
                                 if abs(r["alpha_new"] - al) < 1e-6]))
                                for al in sorted(set(r["alpha_new"]
                                                     for r in rt_valid))},
                            "coverage": len(pairs) / len(rt_valid)}
            # does the metric separate ref-vs-mixture (alpha changed)?
            d = [abs(r[f"mix_{tag}"] - r[f"ref_{tag}"]) for r in rt_valid
                 if r[f"mix_{tag}"] is not None and r[f"ref_{tag}"] is not None]
            if d:
                val.setdefault(tag, {})["anchor_median_err_s"] = float(
                    np.median(d))
    out = {"meta": {"date": str(datetime.date.today()),
                    "script": "src/p3_r2_metrics.py", "system": args.system,
                    "clap": None if args.no_clap else CLAP_ID,
                    "prompt": "the sound of <class>",
                    "rt60": {"offset": "Schroeder T10 (-5..-15 dB, x6) on"
                             " the free decay after the last dry offset"
                             " (-40 dB re peak), tails >= 0.3 s only",
                             "edc": "RMS dB distance of Schroeder decay"
                             " curves over the first 0.3 s of free decay",
                             "blind": "decay-rate distribution, 200-6000 Hz,"
                             " 0.25 s monotone windows, 10th percentile"},
                    "notes": "A14.1/A14.2; anchor = unedited mixture, ref ="
                             " closed-form reference"},
           "system": summ(out_rows), "anchor": summ(anchor_rows),
           "reference": summ(ref_rows), "rt60_validation": val,
           "rows": out_rows, "anchor_rows": anchor_rows,
           "rt60_validation_rows": rt_valid}
    (PN / (args.out or f"p3_r2_metrics_{args.system}.json")).write_text(
        json.dumps(out, indent=1))
    for tag in ("system", "anchor", "reference"):
        print(tag, {e: {k: (round(v["median"], 3), v["n"]) for k, v in
                        out[tag][e].items()} for e in out[tag]})
    print("rt60_validation", json.dumps(val))


if __name__ == "__main__":
    main()
