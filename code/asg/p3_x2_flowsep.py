"""External top-venue baseline X2 on the SAME edits as p3_edit_eval
(replayed from edit_specs in the main ledger; mirrors p3_baselines_ext.py):

X2 FlowSep (ICASSP'25, flow-matching text-queried source separation):
   FlowSep stems (GT class-name queries, least-squares scale fit to the
   mixture -- both favor the baseline, disclosed) -> our est heads ->
   semi-analytic inversion -> edit identity. E1-E3.

Adapter: FlowSep is natively 16 kHz (same as ours), but consumes a fixed
10.24 s / 163840-sample window (mel 1024x64, hop 160). Our SEG=64000
(4 s) mixture is zero-padded at the tail to 163840 (deterministic,
pad_wav_start_sample=0, no trim), separated, and the estimate is cropped
back to the first 64000 samples. FlowSep's loader also normalizes the
input (DC removal + peak 0.5); the LS scale fit absorbs that gain.

Run with the dedicated venv (audiocf venv lacks timm/ipdb):
  OMP_NUM_THREADS=8 nice -n 15 /workspace/venvs/flowsep/bin/python \
      src/p3_x2_flowsep.py --ledger ... --out p3_baselines_x2_v0.json
-> paper_numbers/p3_baselines_x2_v0.json
"""
import argparse
import datetime
import importlib.machinery
import json
import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio  # noqa: F401  (parity with template; adapter is length-only)
import yaml

# --- FlowSep import shims (no edits inside the external repo) ---------------
for _m in ("ipdb", "wandb"):
    try:
        __import__(_m)
    except ImportError:
        _stub = types.ModuleType(_m)
        _stub.set_trace = lambda *a, **k: None
        _stub.__spec__ = importlib.machinery.ModuleSpec(_m, None)
        sys.modules[_m] = _stub
# librosa >=0.10 made these keyword-only; FlowSep calls them positionally.
import librosa.util  # noqa: E402
import librosa.filters  # noqa: E402
_pad = librosa.util.pad_center
librosa.util.pad_center = (
    lambda data, size, axis=-1, **kw: _pad(data, size=size, axis=axis, **kw))
_mel = librosa.filters.mel
librosa.filters.mel = (
    lambda sr, n_fft, n_mels=128, fmin=0.0, fmax=None, **kw:
    _mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, **kw))

FS_ROOT = "/workspace/ext_baselines/FlowSep"
sys.path.insert(0, os.path.join(FS_ROOT, "src"))
_orig = torch.load
torch.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": False})
os.chdir(FS_ROOT)  # relative paths inside FlowSep (bigvgan cfg, metadata)
from utilities.data.dataset import AudioDataset  # noqa: E402
from latent_diffusion.util import instantiate_from_config  # noqa: E402
import latent_diffusion.modules.encoders.modules as ENC  # noqa: E402
# FlanT5HiddenState hardcodes cache_dir=/mnt/bn/... (nonexistent here);
# strip it and restore the HF cache env the encoders module clobbered.
os.environ["TRANSFORMERS_CACHE"] = "/workspace/.cache/huggingface/hub"
from transformers import AutoTokenizer, T5EncoderModel  # noqa: E402


class _NoCacheDir:
    def __init__(self, cls):
        self.cls = cls

    def from_pretrained(self, name, **kw):
        kw.pop("cache_dir", None)
        return self.cls.from_pretrained(name, **kw)


ENC.AutoTokenizer = _NoCacheDir(AutoTokenizer)
ENC.T5EncoderModel = _NoCacheDir(T5EncoderModel)

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
FS_SEG = 163840          # FlowSep fixed input window: 10.24 s @ 16 kHz
CKPT = os.path.join(FS_ROOT, "model_logs/pretrained/v2_100k.ckpt")
CFG = os.path.join(FS_ROOT, "lass_config/2channel_flow.yaml")


def q_of(cls):
    m = {"Vehicle_horn_and_car_horn_and_honking": "car horn",
         "Gunshot_and_gunfire": "gunshot", "Thump_and_thud": "thump"}
    return m.get(cls, cls.replace("_", " ").lower())


def build_flowsep(dev):
    """Load FlowSep latent flow-matching model + its feature pipeline."""
    cfg = yaml.load(open(CFG), Loader=yaml.FullLoader)
    # vae.ckpt is not shipped; v2_100k.ckpt holds the full state dict
    # (loaded strict=True below), so skip the VAE-only reload.
    cfg["model"]["params"]["first_stage_config"]["params"][
        "reload_from_ckpt"] = None
    ds_feat = AudioDataset(cfg, split="test", add_ons=[])
    model = instantiate_from_config(cfg["model"]).to(dev)
    sd = torch.load(CKPT, map_location=dev)["state_dict"]
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, ds_feat


def separate(model, ds_feat, mix_path, query, dev, steps, td):
    """(mixture wav path, text query) -> stem estimate, len FS_SEG.

    Mirrors lass_inference.py: real mixed_waveform/mixed_mel condition
    (encoded to x_T), random placeholders for the target-side keys
    (shape-only), caption = text query.
    """
    noise, _ = ds_feat.read_wav_file(mix_path)   # (1, FS_SEG) normalized
    noise = noise[0][:FS_SEG]
    mixed_mel, _ = ds_feat.wav_feature_extraction(noise.reshape(1, -1))
    batch = {
        "fname": [mix_path],
        "text": [query],
        "caption": [query],
        "waveform": torch.rand(1, 1, FS_SEG).to(dev),
        "log_mel_spec": torch.rand(1, 1024, 64).to(dev),
        "sampling_rate": torch.tensor([SR]).to(dev),
        "label_vector": torch.rand(1, 527).to(dev),
        "stft": torch.rand(1, 1024, 512).to(dev),
        "mixed_waveform": torch.from_numpy(
            noise.reshape(1, 1, FS_SEG).astype(np.float32)),
        "mixed_mel": mixed_mel.reshape(1, *mixed_mel.shape),
    }
    with torch.no_grad():
        wav = model.generate_sample(
            [batch], name=os.path.join(td, "flowsep_out"),
            unconditional_guidance_scale=1.0, ddim_steps=steps, n_gen=1,
            save=False, save_mixed=False)
    return np.asarray(wav).reshape(-1)[:FS_SEG]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default="p3_edit_eval_v0.json")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default="p3_baselines_x2_v0.json")
    ap.add_argument("--steps", type=int, default=20,
                    help="flow sampling steps (repo inference default)")
    ap.add_argument("--n_limit", type=int, default=0,
                    help="truncate specs for smoke tests (0 = all)")
    args = ap.parse_args()
    torch.set_num_threads(8)
    dev = "cuda"

    led = json.loads((PN / args.ledger).read_text())
    specs = led["edit_specs"]
    if args.n_limit > 0:
        specs = specs[:args.n_limit]
    model, ds_feat = build_flowsep(dev)
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

    R = {"X2": {e: [] for e in ("E1", "E2", "E3")}}
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
                w = separate(model, ds_feat, mp, q_of(m["cls"]), dev,
                             args.steps, td)
                w16 = w.astype(np.float64)[:SEG]   # crop 10.24 s -> 4 s
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
                R["X2"]["E1"].append({"scene": i, "sisdr": si_sdr(est, ref),
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
                R["X2"]["E2"].append({"scene": i, "sisdr": si_sdr(est, ref),
                                      "mrstft": mrstft_np(est, ref),
                                      "loc": locality(est, ref, yn - gts[k]),
                                      "loc_clean": si_sdr(yn - ests[k],
                                                         yn - gts[k])})
            # E3
            if "E3_slot" in spec:
                k, fm = spec["E3_slot"], spec["E3_factor"]
                ref = yn - gts[k] + gts[k] / fm
                est = yn - ests[k] + ests[k] / fm
                R["X2"]["E3"].append({"scene": i, "sisdr": si_sdr(est, ref),
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
                    "script": "src/p3_x2_flowsep.py",
                    "split": args.split,
                    "locality": ("A12.2: loc = si_sdr((est - ref) + noedit_ref, noedit_ref), noedit_ref = y - gts[k]; for E1 ref == noedit_ref so loc == sisdr (recorded); measured, not granted"),
                    "loc_clean": ("A12.2 correction: loc_clean = si_sdr(y - s_hat_k, y - gts_k) = si_sdr(est - node_prime, noedit_ref) -- exact non-target damage of the identity edit, node-render error excluded (est - node_prime == y - s_hat exactly); for E1 node_prime = 0 so loc_clean == loc == sisdr"),
                    "system": "X2 FlowSep (ICASSP'25)",
                    "ckpt": CKPT,
                    "steps": args.steps,
                    "sr_adapter": ("native 16 kHz == ours (no resample); "
                                   "4 s/64000-sample mixture zero-padded at "
                                   "tail to FlowSep's fixed 10.24 s/163840 "
                                   "window, estimate cropped back to 64000; "
                                   "FlowSep loader normalizes input (DC + "
                                   "peak 0.5), absorbed by LS scale fit"),
                    "notes": ("X2=FlowSep(ICASSP25) + our est/inversion + "
                              "edit identity; GT class-name queries and LS "
                              "scale fit favor the baseline (disclosed); "
                              "posterior (Wiener consistency) projection as "
                              "S1-S3; our est heads for E2; "
                              "same edits as " + args.ledger)},
           "X2": {"stem_sisdr": agg(stem_sisdr),
                  "rows": R["X2"],
                  "edits": {e: {"sisdr": agg(R["X2"][e]),
                                "mrstft": agg(R["X2"][e], "mrstft"),
                                "loc": agg(R["X2"][e], "loc"),
                                "loc_clean": agg(R["X2"][e],
                                                 "loc_clean")}
                            for e in R["X2"] if R["X2"][e]}}}
    (PN / args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out["X2"]["edits"], indent=1))


if __name__ == "__main__":
    main()
