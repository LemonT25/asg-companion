"""External top-venue baseline X3 on the SAME edits as p3_edit_eval
(replayed from edit_specs in the main ledger):

X3 SoloAudio (ICASSP'25, language-oriented target sound extraction with
   an audio diffusion transformer): one stem per source via GT class-name
   text queries (CLAP text embedding -> UDiT + DDIM, repo-default sampler
   settings) -> least-squares scale fit to the mixture -> our Wiener
   posterior projection -> our est heads -> semi-analytic inversion ->
   edit identity. E1-E3. All concessions favor the baseline (disclosed).

Sample-rate adapter: scenes are 16 kHz mono / SEG samples; SoloAudio's
VAE is 24 kHz. torchaudio resample 16k->24k in, 24k->16k out.

Runs under /workspace/venvs/soloaudio/bin/python (NOT the audiocf venv).
-> paper_numbers/p3_baselines_x3_v0.json
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio
import yaml

SOLO = "/workspace/ext_baselines/SoloAudio"
sys.path.insert(0, SOLO)
_orig = torch.load
torch.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": False})
from model.udit import UDiT  # noqa: E402
from vae_modules.stable_vae import load_vae  # noqa: E402  (bypasses the
# dac/audiotools import chain pulled in by vae_modules.autoencoder_wrapper)
from diffusers import DDIMScheduler  # noqa: E402
from transformers import AutoProcessor, ClapModel  # noqa: E402

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
SOLO_SR = 24000          # SoloAudio VAE native rate (pretrained config.json)
VAE_CKPT = f"{SOLO}/pretrained_models/audio-vae.pt"
DIT_CKPT = f"{SOLO}/pretrained_models/soloaudio_v2.pt"
DIT_CFG = f"{SOLO}/config/SoloAudio.yaml"
CLAP_ID = "laion/larger_clap_general"
DDIM_STEPS = 50          # repo default (tse_languageTSE.py)
DDIM_ETA = 0
DDIM_SEED = 2024         # repo default --random-seed


def q_of(cls):
    m = {"Vehicle_horn_and_car_horn_and_honking": "car horn",
         "Gunshot_and_gunfire": "gunshot", "Thump_and_thud": "thump"}
    return m.get(cls, cls.replace("_", " ").lower())


def build_soloaudio(dev):
    """Load CLAP text encoder, 24k stable-audio VAE and the UDiT diffusion
    model with the repo's DDIM scheduler (v-prediction, zero-SNR,
    trailing spacing) -- mirrors tse_languageTSE.py."""
    cfg = yaml.safe_load(open(DIT_CFG))
    clap = ClapModel.from_pretrained(CLAP_ID).to(dev).eval()
    proc = AutoProcessor.from_pretrained(CLAP_ID)
    vae = load_vae(VAE_CKPT).to(dev).eval()
    unet = UDiT(**cfg["diffwrap"]["UDiT"]).to(dev)
    unet.load_state_dict(torch.load(DIT_CKPT, map_location=dev)["model"])
    unet.eval()
    sched = DDIMScheduler(**cfg["ddim"]["diffusers"])
    # repo quirk kept verbatim: dummy add_noise resets scheduler dtypes
    lat = torch.randn((1, 128, 128), device=dev)
    noi = torch.randn(lat.shape).to(dev)
    ts = torch.randint(0, sched.config.num_train_timesteps, (1,),
                       device=dev).long()
    _ = sched.add_noise(lat, noi, ts)
    return {"clap": clap, "proc": proc, "vae": vae, "unet": unet,
            "sched": sched}


@torch.no_grad()
def solo_extract(M, y16, text, dev):
    """(16 kHz mixture np.float64, text query) -> 16 kHz stem np.float64.
    Resample 16k->24k, VAE-encode (quantization_first: bottleneck sample),
    DDIM sample conditioned on mixture latent + CLAP text feature,
    VAE-decode, resample back, trim/pad to SEG."""
    w24 = torchaudio.functional.resample(
        torch.tensor(y16, dtype=torch.float32), SR, SOLO_SR).to(dev)
    z = M["vae"].encoder(w24[None, None, :])
    mix = M["vae"].bottleneck.encode(z)
    ti = M["proc"](text=[text], max_length=10, padding="max_length",
                   truncation=True, return_tensors="pt")
    ti = {k: v.to(dev) for k, v in ti.items()}
    timbre = M["clap"].get_text_features(**ti)

    sched = M["sched"]
    sched.set_timesteps(DDIM_STEPS)
    gen = torch.Generator(device=dev).manual_seed(DDIM_SEED)
    pred = torch.randn(mix.shape, generator=gen, device=dev)
    for t in sched.timesteps:
        pred = sched.scale_model_input(pred, t)
        mo = M["unet"](x=pred, timesteps=t, mixture=mix, timbre=timbre)
        pred = sched.step(model_output=mo, timestep=t, sample=pred,
                          eta=DDIM_ETA, generator=gen).prev_sample
    wav = M["vae"].decoder(pred).squeeze()
    w16 = torchaudio.functional.resample(
        wav.cpu(), SOLO_SR, SR).numpy().astype(np.float64)[:SEG]
    if len(w16) < SEG:
        w16 = np.pad(w16, (0, SEG - len(w16)))
    return w16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default="p3_edit_eval_v0.json")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default="p3_baselines_x3_v0.json")
    ap.add_argument("--n_limit", type=int, default=0,
                    help="truncate edit_specs for smoke tests (0 = all)")
    args = ap.parse_args()
    torch.set_num_threads(8)
    dev = "cuda"

    led = json.loads((PN / args.ledger).read_text())
    specs = led["edit_specs"]
    if args.n_limit > 0:
        specs = specs[:args.n_limit]
    M = build_soloaudio(dev)
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

    R = {"X3": {e: [] for e in ("E1", "E2", "E3")}}
    stem_sisdr = []
    for si, spec in enumerate(specs):
        i = spec["scene"]
        y, stems, dries, exist, traj, thn, alpha, env, K = ds[int(i)]
        yn = y.numpy().astype(np.float64)
        gts = stems.numpy().astype(np.float64)
        alpha, env, K = float(alpha), float(env), int(K)
        meta = json.loads(str(np.load(ds.files[int(i)],
                                      allow_pickle=True)["meta"]))
        ests = []
        for m in meta:
            w16 = solo_extract(M, yn, q_of(m["cls"]), dev)
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
            R["X3"]["E1"].append({"scene": i, "sisdr": si_sdr(est, ref),
                                  "mrstft": mrstft_np(est, ref),
                                  "loc": locality(est, ref, ref),
                                      "loc_clean": si_sdr(est, ref)})
        # E2
        if "E2_slot" in spec:
            k, newv = spec["E2_slot"], spec["E2_newv"]
            with torch.no_grad():
                _, ptn = e2model(torch.tensor(
                    np.stack(ests), dtype=torch.float32).to(dev))
            vh = float(theta_denorm_t(torch.tensor([0]),
                                      ptn[k][None].cpu().clamp(0, 1)))
            gt_v = float(theta_denorm_t(traj[k][None], thn[k][None]))
            ref = yn - gts[k] + render_node(
                dries[k].numpy().astype(np.float64), "passby", newv,
                alpha, env)
            d5 = pipeline_dry(ests[k], "passby", vh, alpha, env)
            node = render_node(d5, "passby", newv, alpha, env)
            est = yn - ests[k] + node
            R["X3"]["E2"].append({"scene": i, "sisdr": si_sdr(est, ref),
                                  "mrstft": mrstft_np(est, ref),
                                  "loc": locality(est, ref, yn - gts[k]),
                                      "loc_clean": si_sdr(yn - ests[k],
                                                         yn - gts[k])})
        # E3
        if "E3_slot" in spec:
            k, fm = spec["E3_slot"], spec["E3_factor"]
            ref = yn - gts[k] + gts[k] / fm
            est = yn - ests[k] + ests[k] / fm
            R["X3"]["E3"].append({"scene": i, "sisdr": si_sdr(est, ref),
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
                    "script": "src/p3_x3_soloaudio.py",
                    "split": args.split,
                    "locality": ("A12.2: loc = si_sdr((est - ref) + noedit_ref, noedit_ref), noedit_ref = y - gts[k]; for E1 ref == noedit_ref so loc == sisdr (recorded); measured, not granted"),
                    "loc_clean": ("A12.2 correction: loc_clean = si_sdr(y - s_hat_k, y - gts_k) = si_sdr(est - node_prime, noedit_ref) -- exact non-target damage of the identity edit, node-render error excluded (est - node_prime == y - s_hat exactly); for E1 node_prime = 0 so loc_clean == loc == sisdr"),
                    "system": "X3 SoloAudio (ICASSP'25)",
                    "sampler": ("DDIM " + str(DDIM_STEPS) + " steps, eta=" +
                                str(DDIM_ETA) + ", seed=" + str(DDIM_SEED) +
                                ", v-prediction, zero-SNR, trailing "
                                "(repo defaults)"),
                    "sr_adapter": ("scenes 16 kHz <-> SoloAudio VAE 24 kHz "
                                   "via torchaudio resample (in and out)"),
                    "notes": ("X3=SoloAudio(ICASSP25) language-TSE + our "
                              "est/inversion + edit identity; concessions "
                              "favoring the baseline (disclosed): GT "
                              "class-name text queries, LS scale fit to "
                              "the mixture, Wiener posterior projection, "
                              "our est heads for E2; same edits as " +
                              args.ledger)},
           "X3": {"stem_sisdr": agg(stem_sisdr),
                  "rows": R["X3"],
                  "edits": {e: {"sisdr": agg(R["X3"][e]),
                                "mrstft": agg(R["X3"][e], "mrstft"),
                                "loc": agg(R["X3"][e], "loc"),
                                "loc_clean": agg(R["X3"][e],
                                                 "loc_clean")}
                            for e in R["X3"] if R["X3"][e]}}}
    (PN / args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out["X3"]["edits"], indent=1))


if __name__ == "__main__":
    main()
