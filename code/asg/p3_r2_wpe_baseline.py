"""Simple room-edit baseline for the revision (Reviewer 3 follow-up):
blind WPE dereverberation (nara_wpe, single channel) followed by
convolution with the target absorption response. It is our E4 path
with the learned absorption estimate and the Wiener inverse replaced by
a blind dereverberator, on the same scenes and references.

  E4: WPE(y)            -> Phi_{alpha'}   (mixture level, as ours)
  C3: WPE(y - s_hat_a)  -> Phi_{alpha'}   (our E1 output, then room)

Writes p3/r2_out/wpe/<scene>_{E4,C3}.wav and
paper_numbers/p3_r2_wpe.json (same layout as the other R2 ledgers), so
p3_r2_metrics.py --system wpe scores CLAP, RT60 and EDC.
CPU only. Run with /workspace/venvs/audiocf python.
"""
import argparse
import datetime
import json
import sys

import numpy as np
import soundfile as sf
import torch
from nara_wpe.utils import istft, stft
from nara_wpe.wpe import wpe

sys.path.insert(0, "/workspace/CF_ASG/src")
import p3_r2_common as C  # noqa: E402
from p3_r2_common import (SceneSet, SEG, PN, R2_OUT, load_scene,  # noqa
                          chain_specs, references, score, save_wav, agg)
from p3_train import ALPHA_NODES  # noqa: E402
from p2_common import phi_reverb  # noqa: E402

STFT_SIZE, STFT_SHIFT = 512, 128
TAPS, DELAY, ITERS = 10, 3, 3


def wpe_dereverb(x):
    """Single-channel WPE on a float64 signal, same length out."""
    Y = stft(x[None, :], size=STFT_SIZE, shift=STFT_SHIFT)   # (1, T, F)
    Y = Y.transpose(2, 0, 1)                                  # (F, D, T)
    Z = wpe(Y, taps=TAPS, delay=DELAY, iterations=ITERS,
            statistics_mode="full")
    z = istft(Z.transpose(1, 2, 0), size=STFT_SIZE, shift=STFT_SHIFT)[0]
    if len(z) < len(x):
        z = np.pad(z, (0, len(x) - len(z)))
    return z[:len(x)]


def reroom(d, na):
    nn_ = min(ALPHA_NODES, key=lambda t: abs(t - na))
    return phi_reverb(d, nn_)[:SEG]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default="p3_edit_eval_v1.json")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n_limit", type=int, default=0)
    ap.add_argument("--out", default="p3_r2_wpe.json")
    args = ap.parse_args()
    torch.set_num_threads(4)
    led = json.loads((PN / args.ledger).read_text())
    ds = SceneSet(args.split)
    specs = chain_specs(ds, led["edit_specs"])
    if args.n_limit:
        specs = specs[:args.n_limit]
    odir = R2_OUT / "wpe"
    rows = {}
    n_done = 0
    for si, spec in enumerate(specs):
        if "E4_newa" not in spec and "C3_newa" not in spec:
            continue
        i = spec["scene"]
        sc = load_scene(ds, i)
        refs = references(sc, spec)
        if "E4" in refs:
            ref, ned = refs["E4"]
            est = reroom(wpe_dereverb(sc["yn"]), spec["E4_newa"])
            rows.setdefault("E4", []).append({"scene": i,
                                              **score(est, ref, ned)})
            save_wav(odir / f"{i:05d}_E4.wav", est)
        if "C3" in refs:
            p = R2_OUT / "ours" / f"{i:05d}_E1.wav"
            if p.exists():
                ref, ned = refs["C3"]
                e1 = sf.read(str(p), dtype="float64")[0]
                est = reroom(wpe_dereverb(e1), spec["C3_newa"])
                rows.setdefault("C3", []).append({"scene": i,
                                                  **score(est, ref, ned)})
                save_wav(odir / f"{i:05d}_C3.wav", est)
        n_done += 1
        if n_done % 10 == 0:
            print(f"{n_done} room scenes, E4 {len(rows.get('E4', []))}, "
                  f"C3 {len(rows.get('C3', []))}", flush=True)
    out = {"meta": {"date": str(datetime.date.today()),
                    "script": "src/p3_r2_wpe_baseline.py",
                    "system": "WPE + target response (blind dereverb, "
                              "nara_wpe taps 10 delay 3 iters 3, STFT 512/128)",
                    "split": args.split, "ledger": args.ledger,
                    "wav_dir": str(odir),
                    "notes": "Post hoc baseline added in revision R1 after "
                             "the first audit (not pre-registered). Same "
                             "references and scenes as the R2 ledgers; C3 "
                             "uses our E1 output (y minus our stem) before "
                             "the blind dereverberation."},
           "specs": specs, "rows": rows,
           "edits": {e: {k: agg(rows[e], k) for k in ("sisdr", "mrstft", "loc")}
                     for e in rows}}
    (PN / args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({e: (round(out["edits"][e]["sisdr"]["median"], 2),
                          out["edits"][e]["sisdr"]["n"]) for e in rows}))


if __name__ == "__main__":
    main()
