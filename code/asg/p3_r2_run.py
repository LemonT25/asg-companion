"""A14 runner for identity systems: ours (S3) and X1..X4.

Replays the pre-registered edit_specs of the main ledger, derives the
chain specs (p3_r2_common.chain_specs), applies every edit with the
node identities on the system's stems, scores SI-SDR / mrSTFT /
locality and SAVES every edited waveform (p3/r2_out/<system>/).
Ours additionally ledgers the chain order test.

  --system ours   : /workspace/venvs/audiocf   (sep2 + est heads)
  --system x1     : /workspace/venvs/audiocf   (AudioSep)
  --system x2     : /workspace/venvs/flowsep   (FlowSep)
  --system x3     : /workspace/venvs/soloaudio (SoloAudio)
  --system x4     : /workspace/venvs/omnisep   (OmniSep)
-> paper_numbers/p3_r2_<system>.json
"""
import argparse
import datetime
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, "/workspace/CF_ASG/src")
import p3_r2_common as C  # noqa: E402
from p3_r2_common import (SceneSet, SEG, SR, PN, CK, R2_OUT,  # noqa: E402
                          load_scene, chain_specs, references, Heads,
                          identity_edits, score, save_wav, save_refs,
                          agg, q_of, posterior_stems, posterior_stems_own,
                          pit_by_sisdr, MAXK)


def stems_ours(dev):
    from p3_train import SceneModel
    smodel = SceneModel("sep2").to(dev)
    smodel.load_state_dict(torch.load(CK / "sep2_latest.pt",
                                      map_location=dev)["model"])
    smodel.eval()

    def fn(sc, heads):
        y = sc["y"]
        with torch.no_grad():
            dsep, esep, _, _, _, evsep = smodel.slots(y[None].to(dev))
            dsep = dsep[0, ..., :SEG].cpu()
            esep = torch.sigmoid(esep[0]).cpu()
            env3 = float(torch.sigmoid(evsep[0]) > 0.5)
        sepst = dsep.numpy().astype(np.float64)
        po3 = posterior_stems_own(smodel, sc["yn"],
                                  [sepst[k] for k in range(MAXK)], esep, dev)
        pp = pit_by_sisdr(po3, sc["gts"], sc["K"])   # slot -> source map
        ests = [po3[pp[k]] for k in range(sc["K"])]
        al3 = heads.alpha(sc["yn"])
        raw = [sepst[pp[k]] for k in range(sc["K"])]   # ledger convention
        return ests, al3, env3, raw
    return fn


def stems_x1(dev):
    sys.path.insert(0, "/workspace/CF_AQA/AudioSep")
    _orig = torch.load
    torch.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": False})
    # transformers >= 5 dropped the RoBERTa position_ids buffer; the
    # CLAP checkpoint still carries it. Loader shim only (no repo edit).
    _lsd = torch.nn.Module.load_state_dict

    def _lsd_shim(self, sd, strict=True, **kw):
        sd = {k: v for k, v in sd.items()
              if not k.endswith("embeddings.position_ids")}
        return _lsd(self, sd, strict=strict, **kw)
    torch.nn.Module.load_state_dict = _lsd_shim
    from pipeline import build_audiosep, separate_audio
    import torchaudio
    snaps = sorted(Path("/workspace/hf_cache/hub/models--audo--AudioSep/"
                        "snapshots").glob("*/audiosep_base_4M_steps.ckpt"))
    model = build_audiosep("config/audiosep_base.yaml", str(snaps[0]), dev)
    td = tempfile.mkdtemp()

    def fn(sc, heads):
        mp = f"{td}/mix.wav"
        sf.write(mp, sc["yn"].astype(np.float32), SR)
        ests = []
        for cls in sc["classes"]:
            op = f"{td}/est.wav"
            separate_audio(model, mp, q_of(cls), op, dev)
            w, sr = sf.read(op, dtype="float32", always_2d=True)
            w = torch.tensor(w.T)
            w16 = torchaudio.functional.resample(
                w.mean(0), sr, SR).numpy().astype(np.float64)[:SEG]
            if len(w16) < SEG:
                w16 = np.pad(w16, (0, SEG - len(w16)))
            a = float(np.dot(w16, sc["yn"]) / (np.dot(w16, w16) + 1e-12))
            ests.append(a * w16)
        return posterior_stems(sc["yn"], ests), None, None, None
    return fn


def stems_x3(dev):
    from p3_x3_soloaudio import build_soloaudio, solo_extract
    M = build_soloaudio(dev)

    def fn(sc, heads):
        ests = []
        for cls in sc["classes"]:
            w16 = solo_extract(M, sc["yn"], q_of(cls), dev)
            a = float(np.dot(w16, sc["yn"]) / (np.dot(w16, w16) + 1e-12))
            ests.append(a * w16)
        return posterior_stems(sc["yn"], ests), None, None, None
    return fn


def stems_x4(dev):
    from p3_x4_omnisep import build_omnisep, omni_extract
    M = build_omnisep(dev)

    def fn(sc, heads):
        ests = []
        for cls in sc["classes"]:
            w16 = omni_extract(M, sc["yn"], q_of(cls), dev)
            a = float(np.dot(w16, sc["yn"]) / (np.dot(w16, w16) + 1e-12))
            ests.append(a * w16)
        return posterior_stems(sc["yn"], ests), None, None, None
    return fn


def stems_x2(dev, steps=20):
    from p3_x2_flowsep import build_flowsep, separate
    model, ds_feat = build_flowsep(dev)
    td = tempfile.mkdtemp()

    def fn(sc, heads):
        mp = f"{td}/mix.wav"
        sf.write(mp, sc["yn"].astype(np.float32), SR)
        ests = []
        # FlowSep samples from noise; the main-ledger harness set no
        # seed (medians move by up to 1 dB between runs). Fixed per-scene
        # seed here so the A14 replay is reproducible.
        torch.manual_seed(1000 + int(sc["scene_idx"]))
        for cls in sc["classes"]:
            w = separate(model, ds_feat, mp, q_of(cls), dev, steps, td)
            w16 = w.astype(np.float64)[:SEG]
            if len(w16) < SEG:
                w16 = np.pad(w16, (0, SEG - len(w16)))
            a = float(np.dot(w16, sc["yn"]) / (np.dot(w16, w16) + 1e-12))
            ests.append(a * w16)
        return posterior_stems(sc["yn"], ests), None, None, None
    return fn


NAMES = {"ours": "Ours (sep2 + est + semi-analytic inverse)",
         "x1": "X1 AudioSep", "x2": "X2 FlowSep", "x3": "X3 SoloAudio",
         "x4": "X4 OmniSep"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=list(NAMES))
    ap.add_argument("--ledger", default="p3_edit_eval_v1.json")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n_limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.set_num_threads(4)
    dev = "cuda"
    out_name = args.out or f"p3_r2_{args.system}.json"

    led = json.loads((PN / args.ledger).read_text())
    ds = SceneSet(args.split)
    specs = chain_specs(ds, led["edit_specs"])
    if args.n_limit > 0:
        specs = specs[:args.n_limit]
    heads = Heads(dev)
    get_stems = {"ours": stems_ours, "x1": stems_x1, "x2": stems_x2,
                 "x3": stems_x3, "x4": stems_x4}[args.system](dev)

    rows = {e: [] for e in C.EDITS}
    order = {e: [] for e in C.CHAIN_DEF}
    stem_sisdr = []
    odir = R2_OUT / args.system
    for si, spec in enumerate(specs):
        i = spec["scene"]
        sc = load_scene(ds, i)
        refs = references(sc, spec)
        save_refs(sc, spec, refs)
        ests, al_hat, env_hat, vel_in = get_stems(sc, heads)
        for k in range(sc["K"]):
            stem_sisdr.append(C.si_sdr(ests[k], sc["gts"][k]))
        E, D = identity_edits(sc, spec, ests, heads, al_hat, env_hat,
                              order_test=(args.system == "ours"),
                              vel_stems=vel_in)
        for e, est in E.items():
            ref, ned = refs[e]
            r = {"scene": i, **score(est, ref, ned)}
            rows[e].append(r)
            save_wav(odir / f"{i:05d}_{e}.wav", est)
        for e, d in D.items():
            order[e].append(d)
        if (si + 1) % 25 == 0:
            print(f"{si+1}/{len(specs)}", flush=True)

    out = {"meta": {"date": str(datetime.date.today()),
                    "script": "src/p3_r2_run.py", "system": NAMES[args.system],
                    "split": args.split, "ledger": args.ledger,
                    "n_specs": len(specs), "chains": C.CHAIN_DEF,
                    "wav_dir": str(odir),
                    "alpha_env": ("own est head / sep head" if
                                  args.system == "ours" else
                                  "GT alpha and env granted (main-ledger"
                                  " convention for external stems)"),
                    "notes": "A14: same specs as the main ledger; chains"
                             " derived by p3_r2_common.chain_specs (seed"
                             " 14); every edited waveform saved"},
           "specs": specs,
           "stem_sisdr": agg(stem_sisdr),
           "rows": rows,
           "edits": {e: {k: agg(rows[e], k) for k in ("sisdr", "mrstft",
                                                      "loc")}
                     for e in rows if rows[e]}}
    if args.system == "ours":
        out["order_test"] = {e: {"n": len(v), "max_abs_diff": max(v)
                                 if v else None} for e, v in order.items()}
    (PN / out_name).write_text(json.dumps(out, indent=1))
    print(json.dumps({e: {"sisdr": out["edits"][e]["sisdr"]["median"],
                          "n": out["edits"][e]["sisdr"]["n"]}
                      for e in out["edits"]}, indent=1))
    if args.system == "ours":
        print("order_test", out["order_test"])


if __name__ == "__main__":
    main()
