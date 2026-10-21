"""External top-venue baselines on the SAME edits as p3_edit_eval (replayed
from edit_specs in the main ledger):

S5 AudioSep-pipeline (ICLR'24 text-queried separation): AudioSep stems
   (GT class-name queries, least-squares scale fit to the mixture --
   both favor the baseline, disclosed) -> our est heads -> semi-analytic
   inversion -> edit identity. E1-E3.
S6 ZETA (ICML'24 DDPM-inversion editing): direct prompt edit for E1
   (remove-the-<class>); regeneration family representative; subset of
   scenes (compute-bound). P1 established the family's codec floor.
-> paper_numbers/p3_baselines_ext_v0.json
"""
import argparse
import datetime
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, "/workspace/CF_AQA/AudioSep")
_orig = torch.load
torch.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": False})
os.chdir("/workspace/CF_AQA/AudioSep")
from pipeline import build_audiosep, separate_audio  # noqa: E402

sys.path.insert(0, "/workspace/CF_ASG/src")
sys.path.insert(0, "/workspace/CF_PhysAudio/src")
sys.path.insert(0, "/workspace/audiocf_render")
from p3_train import (SceneSet, Estimator, theta_denorm_t, SEG,  # noqa: E402
                      ALPHA_NODES)
from p3_edit_eval import (si_sdr, mrstft_np, render_node,  # noqa: E402
                          pipeline_dry, posterior_stems,
                          locality)

ROOT = Path("/workspace/CF_ASG")
PN = ROOT / "paper_numbers"
CK = ROOT / "p3" / "ckpt"
SR = 16000
CKPT = ("/workspace/hf_cache/hub/models--audo--AudioSep/snapshots/"
        "891454f4cb60363dd67fdef052508d4866ad42f3/audiosep_base_4M_steps.ckpt")


def q_of(cls):
    m = {"Vehicle_horn_and_car_horn_and_honking": "car horn",
         "Gunshot_and_gunfire": "gunshot", "Thump_and_thud": "thump"}
    return m.get(cls, cls.replace("_", " ").lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default="p3_edit_eval_v0.json")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default="p3_baselines_ext_v0.json")
    args = ap.parse_args()
    torch.set_num_threads(8)
    dev = "cuda"

    led = json.loads((PN / args.ledger).read_text())
    specs = led["edit_specs"]
    model = build_audiosep("config/audiosep_base.yaml", CKPT, dev)
    from p3_train import Est2
    e2model = Est2().to(dev)
    e2model.load_state_dict(torch.load(CK / "est2_latest.pt",
                                       map_location=dev)["model"])
    e2model.eval()
    emodel = Estimator().to(dev)
    emodel.load_state_dict(torch.load(CK / "est_latest.pt",
                                      map_location=dev)["model"])
    emodel.eval()
    ds = SceneSet(args.split)

    R = {"S5": {e: [] for e in ("E1", "E2", "E3")}}
    stem_sisdr = []
    with tempfile.TemporaryDirectory() as td:
        for si, spec in enumerate(specs):
            i = spec["scene"]
            y, stems, dries, exist, traj, thn, alpha, env, K = ds[int(i)]
            yn = y.numpy().astype(np.float64)
            gts = stems.numpy().astype(np.float64)
            alpha, env, K = float(alpha), float(env), int(K)
            meta = json.loads(str(np.load(ds.files[int(i)],
                                          allow_pickle=True)["meta"]))
            mp = f"{td}/mix.wav"
            sf.write(mp, y.numpy().astype(np.float32), SR)
            ests = []
            for m in meta:
                op = f"{td}/est.wav"
                separate_audio(model, mp, q_of(m["cls"]), op, dev)
                w, sr = torchaudio.load(op)
                w16 = torchaudio.functional.resample(
                    w.mean(0), sr, SR).numpy().astype(np.float64)[:SEG]
                if len(w16) < SEG:
                    w16 = np.pad(w16, (0, SEG - len(w16)))
                a = float(np.dot(w16, yn) / (np.dot(w16, w16) + 1e-12))
                ests.append(a * w16)   # LS scale fit to the mixture
            # A5.1: same Wiener consistency projection as S1-S3
            # (concession -- mixture phase + exact mixture consistency)
            ests = posterior_stems(yn, ests)
            for k in range(K):
                stem_sisdr.append(si_sdr(ests[k], gts[k]))

            # E1 (A5.3: specs only contain E1 for K>=2 scenes)
            if "E1_slot" in spec:
                k = spec["E1_slot"]
                ref = yn - gts[k]
                est = yn - ests[k]
                R["S5"]["E1"].append({"scene": i, "sisdr": si_sdr(est, ref),
                                      "mrstft": mrstft_np(est, ref),
                                      "loc": locality(est, ref, ref),
                                      "loc_clean": si_sdr(est, ref)})
            # E2
            if "E2_slot" in spec:
                k, newv = spec["E2_slot"], spec["E2_newv"]
                with torch.no_grad():
                    _, pv2 = e2model(torch.tensor(
                        np.stack(ests), dtype=torch.float32).to(dev))
                vh = float(theta_denorm_t(torch.tensor([0]),
                                          pv2[k][None].cpu().clamp(0, 1)))
                gt_v = float(theta_denorm_t(traj[k][None], thn[k][None]))
                ref = yn - gts[k] + render_node(
                    dries[k].numpy().astype(np.float64), "passby", newv,
                    alpha, env)
                d5 = pipeline_dry(ests[k], "passby", vh, alpha, env)
                node = render_node(d5, "passby", newv, alpha, env)
                est = yn - ests[k] + node
                R["S5"]["E2"].append({"scene": i, "sisdr": si_sdr(est, ref),
                                      "mrstft": mrstft_np(est, ref),
                                      "loc": locality(est, ref, yn - gts[k]),
                                      "loc_clean": si_sdr(yn - ests[k],
                                                         yn - gts[k])})
            # E3
            if "E3_slot" in spec:
                k, fm = spec["E3_slot"], spec["E3_factor"]
                ref = yn - gts[k] + gts[k] / fm
                est = yn - ests[k] + ests[k] / fm
                R["S5"]["E3"].append({"scene": i, "sisdr": si_sdr(est, ref),
                                      "mrstft": mrstft_np(est, ref),
                                      "loc": locality(est, ref, yn - gts[k]),
                                      "loc_clean": si_sdr(yn - ests[k],
                                                         yn - gts[k])})
            if (si + 1) % 25 == 0:
                print(f"{si+1}/{len(specs)}", flush=True)

    rng = np.random.default_rng(0)

    def agg(rows, key="sisdr"):
        v = ([r[key] for r in rows if key in r]
             if rows and isinstance(rows[0], dict)
             else list(rows))
        if not v:
            return None
        m = np.median(rng.choice(v, (5000, len(v))), axis=1)
        return {"median": float(np.median(v)),
                "ci95": [float(np.percentile(m, 2.5)),
                         float(np.percentile(m, 97.5))], "n": len(v)}

    out = {"meta": {"date": str(datetime.date.today()),
                    "script": "src/p3_baselines_ext.py",
                    "split": args.split,
                    "locality": ("A12.2: loc = si_sdr((est - ref) + noedit_ref, noedit_ref), noedit_ref = y - gts[k]; for E1 ref == noedit_ref so loc == sisdr (recorded); measured, not granted"),
                    "loc_clean": ("A12.2 correction: loc_clean = si_sdr(y - s_hat_k, y - gts_k) = si_sdr(est - node_prime, noedit_ref) -- exact non-target damage of the identity edit, node-render error excluded (est - node_prime == y - s_hat exactly); for E1 node_prime = 0 so loc_clean == loc == sisdr"),
                    "notes": ("S5=AudioSep(ICLR24) + our est/inversion + "
                              "edit identity; GT class-name queries and LS "
                              "scale fit favor the baseline (disclosed); "
                              "same edits as " + args.ledger)},
           "S5": {"stem_sisdr": agg(stem_sisdr),
                  "rows": R["S5"],
                  "edits": {e: {"sisdr": agg(R["S5"][e]),
                                "mrstft": agg(R["S5"][e], "mrstft"),
                                "loc": agg(R["S5"][e], "loc"),
                                "loc_clean": agg(R["S5"][e],
                                                 "loc_clean")}
                            for e in R["S5"] if R["S5"][e]}}}
    (PN / args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out["S5"]["edits"], indent=1))


if __name__ == "__main__":
    main()
