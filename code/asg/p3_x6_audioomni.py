"""External top-venue baseline X6 on the SAME edits as p3_edit_eval
(replayed from edit_specs in the main ledger; mirrors p3_baselines_ext.py):

X6 Audio-Omni (arXiv'26, unified understanding/generation/editing;
   frozen Qwen2.5-Omni-3B thinker + rectified-flow DiT): REGENERATION /
   instruction family -- the WHOLE mixture is re-generated from a
   natural-language edit instruction; the edited mixture is compared
   against the same reference signals as S1-S5/X2-X4. E1-E3.
   (Instruction edits are expected to struggle on parametric E2/E3;
   reported as-is.)

Instruction templates (exact strings logged per spec in the output JSON):
  E1: "remove the {class} sound"
  E2: "make the {class} pass by at {newv:.0f} meters per second"
  E3: fm>1 -> "make the {class} sound {fm:.0f} times farther away";
      fm=0.5 -> "make the {class} sound twice as close"

Adapter (disclosed in meta.sr_adapter): Audio-Omni is natively 44.1 kHz
stereo with a fixed 11.0 s / 485100-sample output window and a fixed
10 s / 441000-sample source-conditioning window (mel conditioner
hard-codes Linear(1723 -> 236) mel frames). Our 4 s / 64000-sample
16 kHz mixture is resampled to 44.1 kHz mono by the repo's own loader
(api._load_audio_mono) and tail-zero-padded to 441000 samples, exactly
as the repo's Editing UI does (interface/gradio.py
_load_audio_input_wav); the edited output is averaged to
mono, resampled 44.1 -> 16 kHz, trimmed to the first 64000 samples
(zero-padded if short), and least-squares scale-fitted to the input
mixture yn (the repo's _postprocess peak-normalizes to int16, so the
output gain is arbitrary; the LS fit absorbs it -- favors the baseline,
disclosed). Free-form instructions are injected by replicating
api.AudioOmni.edit()'s conditioning with the repo's own SYSTEM_EDITING
prompt wrapper (no edits inside the external repo). Deterministic seed
per (spec, edit).

Qwen2.5-Omni-3B weights live inside model.ckpt (verified: 2420/2420
state-dict keys match the SKIP_QWEN_WEIGHTS=1 skeleton); only its
config/tokenizer are read from a local snapshot (QWEN_OMNI_MODEL_PATH).

Run with the dedicated venv (audiocf venv untouched; audioomni chains
--system-site-packages torch 2.8.0+cu128 / transformers 5.14.1):
  OMP_NUM_THREADS=8 nice -n 15 /workspace/venvs/audioomni/bin/python \
      src/p3_x6_audioomni.py --ledger ... --out p3_baselines_x6_v0.json
-> paper_numbers/p3_baselines_x6_v0.json
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

AO_ROOT = "/workspace/ext_baselines/Audio-Omni"
QWEN_CFG = "/workspace/ext_baselines/qwen25omni3b_cfg"
# Qwen thinker weights come from model.ckpt; only config/tokenizer are
# read from the local snapshot (no HF network access at run time).
os.environ.setdefault("SKIP_QWEN_WEIGHTS", "1")
os.environ.setdefault("QWEN_OMNI_MODEL_PATH", QWEN_CFG)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, AO_ROOT)
os.chdir(AO_ROOT)  # relative vocab path inside Audio-Omni.json (tts-f5)
from audio_omni.api import AudioOmni  # noqa: E402
from audio_omni.prompts import SYSTEM_EDITING, _wrap  # noqa: E402
from audio_omni.inference.generation import generate_diffusion_cond  # noqa: E402

sys.path.insert(0, "/workspace/CF_ASG/src")
sys.path.insert(0, "/workspace/CF_PhysAudio/src")
sys.path.insert(0, "/workspace/audiocf_render")
from p3_train import SceneSet, SEG  # noqa: E402
from p3_edit_eval import (si_sdr, mrstft_np, render_node,
                          locality)  # noqa: E402

ROOT = Path("/workspace/CF_ASG")
PN = ROOT / "paper_numbers"
SR = 16000
AO_SR = 44100
CFG = os.path.join(AO_ROOT, "model", "Audio-Omni.json")
CKPT = os.path.join(AO_ROOT, "model", "model.ckpt")


def q_of(cls):
    m = {"Vehicle_horn_and_car_horn_and_honking": "car horn",
         "Gunshot_and_gunfire": "gunshot", "Thump_and_thud": "thump"}
    return m.get(cls, cls.replace("_", " ").lower())


def instr_e1(cls):
    return f"remove the {cls} sound"


def instr_e2(cls, newv):
    return f"make the {cls} pass by at {newv:.0f} meters per second"


def instr_e3(cls, fm):
    if fm > 1:
        return f"make the {cls} sound {fm:.0f} times farther away"
    inv = 1.0 / fm
    if abs(inv - 2.0) < 1e-6:
        return f"make the {cls} sound twice as close"
    return f"make the {cls} sound {inv:.0f} times closer"


def edit_instruction(model, wav_path, instruction, steps, cfg_scale, seed):
    """(mixture wav path, free-form instruction) -> edited mixture.

    Replicates audio_omni.api.AudioOmni.edit() (api.py:120) but with an
    arbitrary instruction instead of the fixed Add/Remove/Extract
    templates; same SYSTEM_EDITING system prompt, same conditioning
    layout, same sampler. The source wav is tail-zero-padded to the
    fixed 10 s / 441000-sample conditioning window (44.1 kHz), exactly
    as the repo's own Editing path does (interface/gradio.py:1746 ->
    _load_audio_input_wav, gradio.py:776): the mel conditioner
    hard-codes Linear(1723 -> 236) over mel frames (441000 / 256 hop ->
    1723), so shorter inputs crash; api._load_audio_mono truncates but
    never pads. Returns float64 mono at 16 kHz, length SEG.
    """
    qwen_prompt = _wrap(SYSTEM_EDITING, instruction)
    source_wav = model._load_audio_mono(wav_path)   # 44.1 kHz mono, <=10 s
    n_in = AO_SR * 10                               # fixed mel window
    if source_wav.shape[-1] < n_in:
        source_wav = torch.nn.functional.pad(
            source_wav, (0, n_in - source_wav.shape[-1]), "constant", 0.0)
    conditioning = [{
        "omni_prompt": {"text_prompt": qwen_prompt, "video_prompt": None,
                        "audio_prompt": None},
        "text_prompt": instruction,
        "audio_input_prompt": {"audio_input_wav": source_wav.to(model.device),
                               "TASK_TTS": False},
        "sync_feature": torch.zeros(1, 240, 768, device=model.device),
        "speech_prompt": None,
    }]
    out = generate_diffusion_cond(
        model.model, steps=steps, cfg_scale=cfg_scale,
        conditioning=conditioning, sample_size=model.sample_size,
        sample_rate=model.sample_rate, seed=seed, device=model.device)
    wav = model._postprocess(out)                    # (2, N) int16, 44.1 kHz
    mono = wav.to(torch.float32).mean(0) / 32767.0
    w16 = torchaudio.functional.resample(
        mono, AO_SR, SR).numpy().astype(np.float64)[:SEG]
    if len(w16) < SEG:
        w16 = np.pad(w16, (0, SEG - len(w16)))
    return w16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default="p3_edit_eval_v0.json")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default="p3_baselines_x6_v0.json")
    ap.add_argument("--n_limit", type=int, default=0,
                    help="truncate specs for smoke tests (0 = all)")
    ap.add_argument("--steps", type=int, default=100,
                    help="diffusion steps (repo edit default)")
    ap.add_argument("--cfg_scale", type=float, default=7.0,
                    help="classifier-free guidance (repo edit default)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    torch.set_num_threads(8)
    dev = args.device

    led = json.loads((PN / args.ledger).read_text())
    specs = led["edit_specs"]
    if args.n_limit > 0:
        specs = specs[:args.n_limit]
    model = AudioOmni(CFG, CKPT, device=dev)
    ds = SceneSet(args.split)

    R = {"X6": {e: [] for e in ("E1", "E2", "E3")}}
    instr_log = []
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
            rec = {"scene": i}

            def run(instr, e_idx):
                w16 = edit_instruction(model, mp, instr, args.steps,
                                       args.cfg_scale,
                                       seed=100000 + si * 10 + e_idx)
                a = float(np.dot(w16, yn) / (np.dot(w16, w16) + 1e-12))
                return a * w16   # LS scale fit to the mixture (disclosed)

            # E1 (A5.3: specs only contain E1 for K>=2 scenes)
            if "E1_slot" in spec:
                k = spec["E1_slot"]
                instr = instr_e1(q_of(meta[k]["cls"]))
                rec["E1"] = instr
                est = run(instr, 1)
                ref = yn - gts[k]
                R["X6"]["E1"].append({"scene": i, "sisdr": si_sdr(est, ref),
                                      "mrstft": mrstft_np(est, ref),
                                      "loc": locality(est, ref, ref)})
            # E2
            if "E2_slot" in spec:
                k, newv = spec["E2_slot"], spec["E2_newv"]
                instr = instr_e2(q_of(meta[k]["cls"]), newv)
                rec["E2"] = instr
                est = run(instr, 2)
                ref = yn - gts[k] + render_node(
                    dries[k].numpy().astype(np.float64), "passby", newv,
                    alpha, env)
                R["X6"]["E2"].append({"scene": i, "sisdr": si_sdr(est, ref),
                                      "mrstft": mrstft_np(est, ref),
                                      "loc": locality(est, ref, yn - gts[k])})
            # E3
            if "E3_slot" in spec:
                k, fm = spec["E3_slot"], spec["E3_factor"]
                instr = instr_e3(q_of(meta[k]["cls"]), fm)
                rec["E3"] = instr
                est = run(instr, 3)
                ref = yn - gts[k] + gts[k] / fm
                R["X6"]["E3"].append({"scene": i, "sisdr": si_sdr(est, ref),
                                      "mrstft": mrstft_np(est, ref),
                                      "loc": locality(est, ref, yn - gts[k])})
            instr_log.append(rec)
            if (si + 1) % 25 == 0:
                print(f"{si+1}/{len(specs)}", flush=True)

    rng = np.random.default_rng(0)

    def agg(rows, key="sisdr"):
        v = [r[key] for r in rows] if rows and isinstance(rows[0], dict) \
            else list(rows)
        if not v:
            return None
        m = np.median(rng.choice(v, (5000, len(v))), axis=1)
        return {"median": float(np.median(v)),
                "ci95": [float(np.percentile(m, 2.5)),
                         float(np.percentile(m, 97.5))], "n": len(v)}

    out = {"meta": {"date": str(datetime.date.today()),
                    "script": "src/p3_x6_audioomni.py",
                    "split": args.split,
                    "locality": ("A12.2: loc = si_sdr((est - ref) + noedit_ref, noedit_ref), noedit_ref = y - gts[k]; for E1 ref == noedit_ref so loc == sisdr (recorded); measured, not granted"),
                    "loc_clean": ("not defined for the regeneration family: the edited mixture has no node decomposition (est - node_prime does not exist); proxy loc only, disclosed"),
                    "system": "X6 Audio-Omni (arXiv'26)",
                    "family": "regeneration",
                    "ckpt": CKPT,
                    "qwen_cfg": QWEN_CFG,
                    "steps": args.steps,
                    "cfg_scale": args.cfg_scale,
                    "python": "/workspace/venvs/audioomni/bin/python",
                    "sr_adapter": ("native 44.1 kHz stereo, fixed 11.0 s/"
                                   "485100-sample output; 4 s/16 kHz "
                                   "mixture resampled to 44.1 kHz mono by "
                                   "the repo loader and tail-zero-padded "
                                   "to the fixed 10 s/441000-sample "
                                   "source-conditioning window (repo mel "
                                   "conditioner expects 1723 mel frames; "
                                   "same padding as the repo's Editing UI "
                                   "_load_audio_input_wav); edited output "
                                   "mono-averaged, resampled 44.1->16 kHz, "
                                   "trimmed to 64000 samples; repo output "
                                   "is peak-normalized int16, absorbed by "
                                   "LS scale fit to the mixture "
                                   "(disclosed)"),
                    "notes": ("X6=Audio-Omni(SIGGRAPH26) instruction-based "
                              "regeneration: whole mixture re-generated "
                              "from a natural-language edit instruction "
                              "(repo SYSTEM_EDITING prompt, free-form user "
                              "instruction; exact strings in "
                              "'instructions'); edited mixture compared "
                              "against the same references; GT class names "
                              "in instructions and LS scale fit favor the "
                              "baseline (disclosed); no stem/estimator "
                              "stage (regeneration family); deterministic "
                              "seed per (spec, edit); same edits as "
                              + args.ledger)},
           "X6": {"rows": R["X6"],
                  "edits": {e: {"sisdr": agg(R["X6"][e]),
                                "mrstft": agg(R["X6"][e], "mrstft"),
                                "loc": agg(R["X6"][e], "loc")}
                            for e in R["X6"] if R["X6"][e]},
                  "instructions": instr_log}}
    (PN / args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out["X6"]["edits"], indent=1))


if __name__ == "__main__":
    main()
