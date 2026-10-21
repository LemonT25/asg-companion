"""External top-venue baseline X4 on the SAME edits as p3_edit_eval
(replayed from edit_specs in the main ledger):

X4 OmniSep (ICLR'25, omni-modal query-based sound separation): one stem
   per source via GT class-name text queries (ImageBind-huge text
   embedding -> U-Net mask on log-freq-warped mixture spectrogram,
   repo infer3.py path: binary mask @0.5, mixture-phase ISTFT) ->
   least-squares scale fit to the mixture -> our Wiener posterior
   projection -> our est heads -> semi-analytic inversion -> edit
   identity. E1-E3. All concessions favor the baseline (disclosed).

Sample-rate adapter: scenes are 16 kHz mono / SEG samples; OmniSep is
natively 16 kHz (train.py default audio_rate=16000) -- identity, no
resampling. Repo inference convention kept: the 4.0 s mixture is tiled
to 65535 samples (= 256 STFT frames, n_fft 1024 / hop 256), the
separated waveform is cropped back to SEG.

Config provenance: the HF checkpoint (Exgc/OmniSep best_model.pt) ships
without train-args.json; hyperparameters are train.py defaults (16 kHz,
n_fft 1024, hop 256, win 1024, layers 7, channels 32, binary/weighted/
log_freq True) + emb_dim 1024 from the README training command,
verified against checkpoint shapes (frame_net.weight = (32, 1024)).

Runs under /workspace/venvs/omnisep/bin/python (NOT the audiocf venv).
-> paper_numbers/p3_baselines_x4_v0.json
"""
import argparse
import datetime
import json
import sys
import types
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F

OMNI = "/workspace/ext_baselines/OmniSep"
IB = "/workspace/ext_baselines/ImageBind"

# ImageBind's package __init__ pulls in pytorchvideo (video/audio query
# path only); text queries need just models/ + the BPE tokenizer, so
# register the package without executing __init__.py.
_pkg = types.ModuleType("imagebind")
_pkg.__path__ = [f"{IB}/imagebind"]
sys.modules["imagebind"] = _pkg
from imagebind.models import imagebind_model  # noqa: E402
from imagebind.models.imagebind_model import ModalityType  # noqa: E402
from imagebind.models.multimodal_preprocessors import (  # noqa: E402
    SimpleTokenizer)

sys.path.insert(0, f"{OMNI}/omnisep")
import omnisep as omnisep_mod  # noqa: E402
import utils as omnisep_utils  # noqa: E402  (OmniSep repo utils: warpgrid)

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
OMNI_SR = 16000          # OmniSep native rate (train.py default) == SR
SEP_CKPT = f"{OMNI}/ckpt/best_model.pt"
IB_CKPT = f"{IB}/.checkpoints/imagebind_huge.pth"
BPE = f"{IB}/imagebind/bpe/bpe_simple_vocab_16e6.txt.gz"
N_FFT, HOP, WIN = 1024, 256, 1024    # train.py defaults
LAYERS, CHANNELS, EMB_DIM = 7, 32, 1024
AUD_LEN = 65535          # repo infer3 convention -> 256 STFT frames


def q_of(cls):
    m = {"Vehicle_horn_and_car_horn_and_honking": "car horn",
         "Gunshot_and_gunfire": "gunshot", "Thump_and_thud": "thump"}
    return m.get(cls, cls.replace("_", " ").lower())


def build_omnisep(dev):
    """Load the OmniSep mask U-Net (ckpt is DataParallel-prefixed) and
    the ImageBind-huge query encoder + BPE tokenizer -- mirrors the
    model setup in omnisep/infer3.py (text-query path)."""
    sep = omnisep_mod.OmniSep(2, LAYERS, CHANNELS, use_log_freq=True,
                              use_weighted_loss=True, use_binary_mask=True,
                              emb_dim=EMB_DIM)
    sd = torch.load(SEP_CKPT, map_location="cpu", weights_only=True)
    sep.load_state_dict({k.removeprefix("module."): v for k, v in sd.items()})
    sep.to(dev).eval()
    ib = imagebind_model.imagebind_huge(pretrained=False)
    ib.load_state_dict(torch.load(IB_CKPT, map_location="cpu",
                                  weights_only=True))
    ib.to(dev).eval()
    tok = SimpleTokenizer(bpe_path=BPE)
    return {"sep": sep, "ib": ib, "tok": tok, "emb_cache": {}}


@torch.no_grad()
def text_emb(M, text, dev):
    """GT class-name query -> normalized ImageBind text embedding
    (infer3.py text-only path; no prompt ensemble)."""
    if text not in M["emb_cache"]:
        tokens = M["tok"](text).unsqueeze(0).to(dev)
        emb = M["ib"]({ModalityType.TEXT: tokens})[ModalityType.TEXT]
        M["emb_cache"][text] = F.normalize(emb, dim=-1).float()
    return M["emb_cache"][text]


@torch.no_grad()
def omni_extract(M, y16, text, dev):
    """(16 kHz mixture np.float64, text query) -> 16 kHz stem np.float64.
    Mirrors infer3.py: tile to 65535 samples, STFT, predict mask on the
    log-freq-warped magnitude, unwarp, binarize @0.5, ISTFT with the
    mixture phase, clip to [-1, 1]; then crop back to SEG.
    (istft_reconstruction reimplemented: repo utils uses removed
    np.complex.)"""
    audio = np.clip(y16, -1, 1).astype(np.float32)
    if len(audio) < AUD_LEN:
        audio = np.tile(audio, AUD_LEN // len(audio) + 1)
    audio = audio[:AUD_LEN]
    spec = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP, win_length=WIN)
    mag = torch.tensor(np.abs(spec))[None, None].to(dev)
    phase = np.angle(spec)

    emb = text_emb(M, text, dev)
    mask = M["sep"].infer(mag, [emb])[0]      # sigmoid'ed (binary_mask)
    grid = torch.from_numpy(omnisep_utils.warpgrid(
        1, N_FFT // 2 + 1, mask.size(3), warp=False)).to(dev)
    mask_lin = F.grid_sample(mask, grid, align_corners=True)
    mask_lin = (mask_lin > 0.5).float()       # repo default binary mask
    pred_mag = mag[0, 0].cpu().numpy() * mask_lin[0, 0].cpu().numpy()
    wav = librosa.istft(pred_mag * np.exp(1j * phase), hop_length=HOP,
                        win_length=WIN, length=AUD_LEN)
    w16 = np.clip(wav, -1.0, 1.0).astype(np.float64)[:SEG]
    if len(w16) < SEG:
        w16 = np.pad(w16, (0, SEG - len(w16)))
    return w16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default="p3_edit_eval_v0.json")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default="p3_baselines_x4_v0.json")
    ap.add_argument("--n_limit", type=int, default=0,
                    help="truncate edit_specs for smoke tests (0 = all)")
    args = ap.parse_args()
    torch.set_num_threads(8)
    dev = "cuda"

    led = json.loads((PN / args.ledger).read_text())
    specs = led["edit_specs"]
    if args.n_limit > 0:
        specs = specs[:args.n_limit]
    M = build_omnisep(dev)
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

    R = {"X4": {e: [] for e in ("E1", "E2", "E3")}}
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
            w16 = omni_extract(M, yn, q_of(m["cls"]), dev)
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
            R["X4"]["E1"].append({"scene": i, "sisdr": si_sdr(est, ref),
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
            R["X4"]["E2"].append({"scene": i, "sisdr": si_sdr(est, ref),
                                  "mrstft": mrstft_np(est, ref),
                                  "loc": locality(est, ref, yn - gts[k]),
                                      "loc_clean": si_sdr(yn - ests[k],
                                                         yn - gts[k])})
        # E3
        if "E3_slot" in spec:
            k, fm = spec["E3_slot"], spec["E3_factor"]
            ref = yn - gts[k] + gts[k] / fm
            est = yn - ests[k] + ests[k] / fm
            R["X4"]["E3"].append({"scene": i, "sisdr": si_sdr(est, ref),
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
                    "script": "src/p3_x4_omnisep.py",
                    "split": args.split,
                    "locality": ("A12.2: loc = si_sdr((est - ref) + noedit_ref, noedit_ref), noedit_ref = y - gts[k]; for E1 ref == noedit_ref so loc == sisdr (recorded); measured, not granted"),
                    "loc_clean": ("A12.2 correction: loc_clean = si_sdr(y - s_hat_k, y - gts_k) = si_sdr(est - node_prime, noedit_ref) -- exact non-target damage of the identity edit, node-render error excluded (est - node_prime == y - s_hat exactly); for E1 node_prime = 0 so loc_clean == loc == sisdr"),
                    "system": "X4 OmniSep (ICLR'25)",
                    "query_encoder": ("ImageBind-huge text embedding "
                                      "(repo infer3.py path, no prompt "
                                      "ensemble), imagebind_huge.pth"),
                    "mask": ("binary mask @0.5 threshold on log-freq-"
                             "warped spectrogram, mixture-phase ISTFT "
                             "(repo defaults)"),
                    "sr_adapter": ("scenes 16 kHz == OmniSep native "
                                   "16 kHz (identity, no resampling); "
                                   "4.0 s mixture tiled to 65535 samples "
                                   "= 256 STFT frames per repo infer3 "
                                   "convention, output cropped to SEG"),
                    "notes": ("X4=OmniSep(ICLR25) text-queried separation "
                              "+ our est/inversion + edit identity; "
                              "concessions favoring the baseline "
                              "(disclosed): GT class-name text queries, "
                              "LS scale fit to the mixture, Wiener "
                              "posterior projection, our est heads for "
                              "E2; same edits as " + args.ledger)},
           "X4": {"stem_sisdr": agg(stem_sisdr),
                  "rows": R["X4"],
                  "edits": {e: {"sisdr": agg(R["X4"][e]),
                                "mrstft": agg(R["X4"][e], "mrstft"),
                                "loc": agg(R["X4"][e], "loc"),
                                "loc_clean": agg(R["X4"][e],
                                                 "loc_clean")}
                            for e in R["X4"] if R["X4"][e]}}}
    (PN / args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out["X4"]["edits"], indent=1))


if __name__ == "__main__":
    main()
