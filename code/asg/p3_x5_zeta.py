"""X5 ZETA (ICML'24) on the SAME edits as p3_edit_eval (replayed from
edit_specs in the main ledger).

Regeneration family: ZETA (DDPM-inversion text-based editing, AudioLDM2
backbone) edits the WHOLE mixture from a text instruction -- no stems, no
estimator, no edit identity. The edited mixture is compared to the exact
same reference signals as S1-S6 (E1 ref = y - stem_k; E2/E3 refs re-render
the GT node). The edited mixture is least-squares scale-fitted to the input
mixture y (same concession the stem baselines get, disclosed).

Inference is one subprocess per (scene, edit) through the official
AudioEditingCode/code/main_run.py with the P1 AMENDMENT_1-locked repo
defaults: mode=ours, num_diffusion_steps=200, tstart=100, cfg_src=3,
cfg_tar=12, seed=42+scene_index. Reuses the P1 env
(/workspace/venvs/audiocf/bin/python) -- run this script with that python.

-> paper_numbers/p3_baselines_x5_v0.json
"""
import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, "/workspace/CF_ASG/src")
sys.path.insert(0, "/workspace/CF_PhysAudio/src")
sys.path.insert(0, "/workspace/audiocf_render")
from p3_train import SceneSet, SEG  # noqa: E402
from p3_edit_eval import (si_sdr, mrstft_np, render_node,
                          locality)  # noqa: E402

ROOT = Path("/workspace/CF_ASG")
PN = ROOT / "paper_numbers"
SR = 16000
ZETA_CODE = Path("/workspace/CF_AQA/AudioEditingCode/code")
ZETA_PY = "/workspace/venvs/audiocf/bin/python"   # env P1 used for ZETA
MODEL_ID = "cvssp/audioldm2"                       # native SR 16 kHz
HF_SNAP = Path("/workspace/hf_cache/hub/models--cvssp--audioldm2/snapshots")
# P1 AMENDMENT_1 locked repo defaults (g0a_run_editor_A.py):
ZETA_CFG = {"mode": "ours", "num_diffusion_steps": 200, "tstart": 100,
            "cfg_src": 3, "cfg_tar": 12}
TMP_REL = "p3x5_tmp"          # results_path must be relative to ZETA_CODE


def q_of(cls):
    m = {"Vehicle_horn_and_car_horn_and_honking": "car horn",
         "Gunshot_and_gunfire": "gunshot", "Thump_and_thud": "thump"}
    return m.get(cls, cls.replace("_", " ").lower())


def join_names(names):
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def scene_prompt(classes):
    return "a recording of " + join_names(classes)


def prompts_for(edit, spec, classes):
    """Exact prompt templates per edit (recorded in the output JSON)."""
    src = scene_prompt(classes)
    if edit == "E1":
        k = spec["E1_slot"]
        rest = [c for j, c in enumerate(classes) if j != k]
        tar = (scene_prompt(rest) if rest
               else "a recording of background ambience")
        # ZETA-style remove edit: full scene vs scene without the class
        return src, tar
    if edit == "E2":
        k, newv = spec["E2_slot"], spec["E2_newv"]
        tar = (src + f", the {classes[k]} passes by at "
               f"{newv:.0f} meters per second")
        return src, tar
    k, fm = spec["E3_slot"], spec["E3_factor"]
    if fm > 1:
        tar = (src + f", the {classes[k]} is {fm:.0f} times "
               "farther away and quieter")
    else:  # fm = 0.5
        tar = src + f", the {classes[k]} is twice as close and louder"
    return src, tar


def run_zeta(mix_path, src_p, tar_p, seed, res_rel):
    """(input wav, source prompt, target prompt) -> (edited, roundtrip) paths
    via official main_run.py. Returns (None, None) on failure."""
    env = {"HF_HOME": "/workspace/hf_cache", "WANDB_MODE": "disabled",
           "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8",
           "PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": "/root"}
    cmd = ["nice", "-n", "15", ZETA_PY, "main_run.py",
           "--model_id", MODEL_ID,
           "--init_aud", str(mix_path),
           "--source_prompt", src_p, "--target_prompt", tar_p,
           "--mode", ZETA_CFG["mode"],
           "--num_diffusion_steps", str(ZETA_CFG["num_diffusion_steps"]),
           "--tstart", str(ZETA_CFG["tstart"]),
           "--cfg_src", str(ZETA_CFG["cfg_src"]),
           "--cfg_tar", str(ZETA_CFG["cfg_tar"]),
           "--seed", str(seed), "--wandb_disable",
           "--results_path", res_rel]
    r = subprocess.run(cmd, cwd=ZETA_CODE, env=env,
                       capture_output=True, text=True, timeout=1800)
    hits = [p for p in (ZETA_CODE / res_rel).rglob("*.wav")
            if p.name != "orig.wav"]
    if r.returncode != 0 or not hits:
        print(f"ZETA FAIL rc={r.returncode}\n{r.stderr[-800:]}", flush=True)
        return None, None
    rt = next((ZETA_CODE / res_rel).rglob("orig.wav"), None)
    return hits[0], rt


def load_fit(wav_path, yn):
    """Edited wav -> 16 kHz, SEG samples, LS scale-fitted to the mixture yn
    (same concession as the stem baselines, disclosed)."""
    w, sr = torchaudio.load(str(wav_path))
    w16 = torchaudio.functional.resample(
        w.mean(0), sr, SR).numpy().astype(np.float64)[:SEG]
    if len(w16) < SEG:
        w16 = np.pad(w16, (0, SEG - len(w16)))
    a = float(np.dot(w16, yn) / (np.dot(w16, w16) + 1e-12))
    return a * w16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default="p3_edit_eval_v0.json")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default="p3_baselines_x5_v0.json")
    ap.add_argument("--n_limit", type=int, default=0,
                    help="if >0, only the first n specs (smoke)")
    args = ap.parse_args()
    torch.set_num_threads(8)
    os.environ.setdefault("OMP_NUM_THREADS", "8")

    # model/code locating (fail fast, before any GPU work)
    assert (ZETA_CODE / "main_run.py").exists(), f"no ZETA at {ZETA_CODE}"
    assert Path(ZETA_PY).exists(), f"no python at {ZETA_PY}"
    snaps = sorted(HF_SNAP.glob("*/unet"))
    assert snaps, f"no audioldm2 unet under {HF_SNAP}"
    print(f"ZETA code: {ZETA_CODE}\naudioldm2: {snaps[0].parent}", flush=True)

    led = json.loads((PN / args.ledger).read_text())
    specs = led["edit_specs"]
    if args.n_limit > 0:
        specs = specs[:args.n_limit]
    ds = SceneSet(args.split)

    R = {e: [] for e in ("E1", "E2", "E3")}
    roundtrip = []          # AudioLDM2 vocoder/VAE codec floor vs y
    n_fail = 0
    try:
        with tempfile.TemporaryDirectory() as td:
            for si, spec in enumerate(specs):
                i = spec["scene"]
                y, stems, dries, exist, traj, thn, alpha, env, K = ds[int(i)]
                yn = y.numpy().astype(np.float64)
                gts = stems.numpy().astype(np.float64)
                alpha, env = float(alpha), float(env)
                meta = json.loads(str(np.load(ds.files[int(i)],
                                              allow_pickle=True)["meta"]))
                classes = [q_of(m["cls"]) for m in meta]
                mp = f"{td}/mix_{i}.wav"
                sf.write(mp, y.numpy().astype(np.float32), SR)

                rt_done = False
                for edit in ("E1", "E2", "E3"):
                    if f"{edit}_slot" not in spec:
                        continue
                    k = spec[f"{edit}_slot"]
                    src_p, tar_p = prompts_for(edit, spec, classes)
                    res_rel = f"{TMP_REL}/s{i}_{edit}"
                    ep, rt = run_zeta(mp, src_p, tar_p, 42 + si, res_rel)
                    if ep is None:
                        n_fail += 1
                        continue
                    est = load_fit(ep, yn)      # est = edited MIXTURE
                    if rt is not None and not rt_done:
                        roundtrip.append(si_sdr(load_fit(rt, yn), yn))
                        rt_done = True
                    # references: EXACTLY as in p3_baselines_ext.py
                    if edit == "E1":
                        ref = yn - gts[k]
                    elif edit == "E2":
                        ref = yn - gts[k] + render_node(
                            dries[k].numpy().astype(np.float64), "passby",
                            spec["E2_newv"], alpha, env)
                    else:
                        ref = yn - gts[k] + gts[k] / spec["E3_factor"]
                    ned = ref if edit == "E1" else yn - gts[k]
                    R[edit].append({"scene": i,
                                    "sisdr": si_sdr(est, ref),
                                    "mrstft": mrstft_np(est, ref),
                                    "loc": locality(est, ref, ned),
                                    "source_prompt": src_p,
                                    "target_prompt": tar_p})
                    shutil.rmtree(ZETA_CODE / res_rel, ignore_errors=True)
                print(f"{si + 1}/{len(specs)}", flush=True)
    finally:
        shutil.rmtree(ZETA_CODE / TMP_REL, ignore_errors=True)

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
                    "script": "src/p3_x5_zeta.py",
                    "split": args.split,
                    "locality": ("A12.2: loc = si_sdr((est - ref) + noedit_ref, noedit_ref), noedit_ref = y - gts[k]; for E1 ref == noedit_ref so loc == sisdr (recorded); measured, not granted"),
                    "loc_clean": ("not defined for the regeneration family: the edited mixture has no node decomposition (est - node_prime does not exist); proxy loc only, disclosed"),
                    "system": "X5 ZETA (ICML'24)",
                    "family": "regeneration",
                    "model_id": MODEL_ID, "model_sr": 16000,
                    "inversion": ZETA_CFG,
                    "seed_rule": "42 + spec_index",
                    "n_fail": n_fail,
                    "notes": ("ZETA edits the whole mixture from a text "
                              "instruction (prompts logged per row); the "
                              "edited mixture is LS scale-fitted to y "
                              "(disclosed), then scored against the same "
                              "references as the stem systems; same edits "
                              "as " + args.ledger)},
           "X5": {"roundtrip_sisdr": agg(roundtrip),
                  "edits": {e: {"sisdr": agg(R[e]),
                                "mrstft": agg(R[e], "mrstft"),
                                "loc": agg(R[e], "loc")}
                            for e in R if R[e]},
                  "rows": R}}
    (PN / args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out["X5"]["edits"], indent=1))


if __name__ == "__main__":
    main()
