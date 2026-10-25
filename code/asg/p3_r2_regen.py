"""A14 runner for the regeneration / instruction family: X5 ZETA and
X6 Audio-Omni, on the first 60 pre-registered specs (compute-bound, as
in the main ledger), with the chain edits C1..C4 in two forms:
  * combined: ONE instruction (or one target prompt) carrying both
    changes -> <scene>_<C>.wav
  * sequential: edit 1 applied, then edit 2 applied to the result
    -> <scene>_<C>seq.wav
Single edits E1..E3 are replayed as before (E4 has no instruction path
in the main ledger and is not added). Edited mixtures are LS-scale
fitted to the input mixture (main-ledger concession, disclosed) and
saved; metrics come from p3_r2_metrics.py.
  --system x5 : /workspace/venvs/audiocf   (ZETA via main_run.py)
  --system x6 : /workspace/venvs/audioomni (Audio-Omni)
-> paper_numbers/p3_r2_<system>.json
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

sys.path.insert(0, "/workspace/CF_ASG/src")
import p3_r2_common as C  # noqa: E402
from p3_r2_common import (SceneSet, SEG, SR, PN, R2_OUT, load_scene,  # noqa
                          chain_specs, references, score, save_wav,
                          save_refs, agg, q_of)

N_SUBSET = 60


# ------------------------------------------------------------ text
def join_names(names):
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def scene_prompt(classes):
    return "a recording of " + join_names(classes)


def e3_phrase(cls, fm):
    if fm > 1:
        return f"the {cls} is {fm:.0f} times farther away and quieter"
    return f"the {cls} is twice as close and louder"


def e2_phrase(cls, newv):
    return f"the {cls} passes by at {newv:.0f} meters per second"


def zeta_prompts(edit, spec, classes):
    """(source prompt, target prompt) per edit; chains combine the two
    phrases in one target prompt. Returns list of (src, tar) steps for
    the sequential form as well."""
    src = scene_prompt(classes)

    def single(e):
        if e == "E1":
            k = spec["E1_slot"]
            rest = [c for j, c in enumerate(classes) if j != k]
            return (scene_prompt(rest) if rest
                    else "a recording of background ambience")
        if e == "E2":
            return src + ", " + e2_phrase(classes[spec["E2_slot"]],
                                          spec["E2_newv"])
        if e == "E3":
            return src + ", " + e3_phrase(classes[spec["E3_slot"]],
                                          spec["E3_factor"])
        if e == "E4":
            room = ("in a less reverberant room"
                    if spec["E4_newa"] > spec.get("_alpha", 0)
                    else "in a more reverberant room")
            return src + ", " + room
    if edit in ("E1", "E2", "E3", "E4"):
        return single(edit), [(src, single(edit))]
    if edit == "C1":      # remove a + re-distance b
        a, b, fm = spec["C1_rm"], spec["C1_slot"], spec["C1_factor"]
        rest = [c for j, c in enumerate(classes) if j != a]
        t1 = scene_prompt(rest)
        t2 = t1 + ", " + e3_phrase(classes[b], fm)
        return t2, [(src, t1), (t1, t2)]
    if edit == "C2":      # re-velocity a + re-distance b
        a, nv = spec["C2_v_slot"], spec["C2_newv"]
        b, fm = spec["C2_r_slot"], spec["C2_factor"]
        t1 = src + ", " + e2_phrase(classes[a], nv)
        t2 = t1 + ", " + e3_phrase(classes[b], fm)
        return t2, [(src, t1), (t1, t2)]
    if edit == "C3":      # remove a + room absorption change
        a, na = spec["C3_rm"], spec["C3_newa"]
        rest = [c for j, c in enumerate(classes) if j != a]
        t1 = scene_prompt(rest)
        room = ("in a less reverberant room" if na > spec.get("_alpha", 0)
                else "in a more reverberant room")
        t2 = t1 + ", " + room
        return t2, [(src, t1), (t1, t2)]
    if edit == "C4":      # remove b + re-velocity a
        a, nv, b = spec["C4_v_slot"], spec["C4_newv"], spec["C4_rm"]
        rest = [c for j, c in enumerate(classes) if j != b]
        t1 = scene_prompt(rest)
        t2 = t1 + ", " + e2_phrase(classes[a], nv)
        return t2, [(src, t1), (t1, t2)]


def omni_instr(edit, spec, classes):
    """Audio-Omni free-form instructions: combined string and the
    sequential list."""
    def i1(k):
        return f"remove the {classes[k]} sound"

    def i2(k, nv):
        return f"make the {classes[k]} pass by at {nv:.0f} meters per second"

    def i3(k, fm):
        if fm > 1:
            return f"make the {classes[k]} sound {fm:.0f} times farther away"
        return f"make the {classes[k]} sound twice as close"

    def i4(na, old):
        return ("make the room less reverberant" if na > old
                else "make the room more reverberant")
    if edit == "E1":
        s = [i1(spec["E1_slot"])]
    elif edit == "E2":
        s = [i2(spec["E2_slot"], spec["E2_newv"])]
    elif edit == "E3":
        s = [i3(spec["E3_slot"], spec["E3_factor"])]
    elif edit == "E4":
        s = [i4(spec["E4_newa"], spec["_alpha"])]
    elif edit == "C1":
        s = [i1(spec["C1_rm"]), i3(spec["C1_slot"], spec["C1_factor"])]
    elif edit == "C2":
        s = [i2(spec["C2_v_slot"], spec["C2_newv"]),
             i3(spec["C2_r_slot"], spec["C2_factor"])]
    elif edit == "C3":
        s = [i1(spec["C3_rm"]), i4(spec["C3_newa"], spec["_alpha"])]
    elif edit == "C4":
        s = [i1(spec["C4_rm"]), i2(spec["C4_v_slot"], spec["C4_newv"])]
    return " and ".join(s), s


# ------------------------------------------------------------ systems
def fit(w16, yn):
    w16 = np.asarray(w16, np.float64)[:SEG]
    if len(w16) < SEG:
        w16 = np.pad(w16, (0, SEG - len(w16)))
    a = float(np.dot(w16, yn) / (np.dot(w16, w16) + 1e-12))
    return a * w16


def read16(path):
    w, sr = sf.read(str(path), dtype="float32", always_2d=True)
    w = torch.tensor(w.T).mean(0)
    if sr != SR:
        w = torchaudio.functional.resample(w, sr, SR)
    return w.numpy().astype(np.float64)


class Zeta:
    """In-process ZETA (p3_r2_zeta_inproc): same settings as the main
    ledger, pipeline loaded once. Run under /workspace/venvs/zeta."""

    def __init__(self):
        from p3_r2_zeta_inproc import ZetaInProc
        self.z = ZetaInProc()
        self.td = tempfile.mkdtemp()
        self.calls = 0

    def edit(self, wav_path, src_p, tar_p, seed, tag):
        out = f"{self.td}/{tag}.wav"
        ok = self.z.edit(wav_path, src_p, tar_p, seed, out)
        self.calls += 1
        return read16(out) if ok else None


class Omni:
    def __init__(self, steps=100, cfg_scale=7.0, dev="cuda"):
        from p3_x6_audioomni import (edit_instruction, AudioOmni, CFG,
                                     CKPT)
        self.model = AudioOmni(CFG, CKPT, device=dev)
        self.edit_instruction = edit_instruction
        self.steps, self.cfg = steps, cfg_scale

    def edit(self, wav_path, instr, seed):
        return self.edit_instruction(self.model, wav_path, instr,
                                     self.steps, self.cfg, seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=["x5", "x6"])
    ap.add_argument("--ledger", default="p3_edit_eval_v1.json")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n_limit", type=int, default=N_SUBSET)
    ap.add_argument("--out", default=None)
    ap.add_argument("--edits", default=None,
                    help="comma list; default = all but E4 (main ledger)")
    ap.add_argument("--merge", action="store_true",
                    help="merge rows/prompts into an existing --out JSON")
    args = ap.parse_args()
    edits = (args.edits.split(",") if args.edits
             else [e for e in C.EDITS if e != "E4"])
    outp = PN / (args.out or f"p3_r2_{args.system}.json")
    prev = json.loads(outp.read_text()) if (args.merge and outp.exists()) \
        else None
    torch.set_num_threads(4)
    led = json.loads((PN / args.ledger).read_text())
    ds = SceneSet(args.split)
    specs = chain_specs(ds, led["edit_specs"])[:args.n_limit]
    sysm = Zeta() if args.system == "x5" else Omni()
    odir = R2_OUT / args.system
    rows = dict(prev["rows"]) if prev else {}
    plog = list(prev["prompts"]) if prev else []
    n_fail = prev["meta"]["n_fail"] if prev else 0
    td = tempfile.mkdtemp()
    for si, spec in enumerate(specs):
        i = spec["scene"]
        sc = load_scene(ds, i)
        spec["_alpha"] = sc["alpha"]
        classes = [q_of(c) for c in sc["classes"]]
        refs = references(sc, spec)
        save_refs(sc, spec, refs)
        mp = f"{td}/mix_{i}.wav"
        sf.write(mp, sc["yn"].astype(np.float32), SR)
        rec = {"scene": i}
        for e in C.EDITS:
            if e not in refs or e not in edits:
                continue
            ref, ned = refs[e]
            seed = 42 + si if args.system == "x5" else 100000 + si * 10 \
                + C.EDITS.index(e)
            # combined form
            if args.system == "x5":
                tar, steps = zeta_prompts(e, spec, classes)
                src = scene_prompt(classes)
                rec[e] = {"source": src, "target": tar}
                w = sysm.edit(mp, src, tar, seed, f"s{i}_{e}")
            else:
                instr, steps = omni_instr(e, spec, classes)
                rec[e] = {"instruction": instr}
                w = sysm.edit(mp, instr, seed)
            if w is None:
                n_fail += 1
            else:
                est = fit(w, sc["yn"])
                rows.setdefault(e, []).append({"scene": i,
                                               **score(est, ref, ned)})
                save_wav(odir / f"{i:05d}_{e}.wav", est)
            # sequential form for chains
            if e.startswith("C") and len(steps) == 2:
                cur = mp
                ok = True
                for j, st in enumerate(steps):
                    if args.system == "x5":
                        w = sysm.edit(cur, st[0], st[1], seed + 1000 * (j + 1),
                                      f"s{i}_{e}seq{j}")
                    else:
                        w = sysm.edit(cur, st, seed + 1000 * (j + 1))
                    if w is None:
                        ok = False
                        break
                    cur = f"{td}/step_{i}_{e}_{j}.wav"
                    sf.write(cur, fit(w, sc["yn"]).astype(np.float32), SR)
                rec[e + "seq"] = steps
                if ok:
                    est = read16(cur)
                    rows.setdefault(e + "seq", []).append(
                        {"scene": i, **score(est, ref, ned)})
                    save_wav(odir / f"{i:05d}_{e}seq.wav", est)
                else:
                    n_fail += 1
        if prev:
            old = [r for r in plog if r["scene"] == i]
            if old:
                old[0].update({k: v for k, v in rec.items() if k != "scene"})
            else:
                plog.append(rec)
        else:
            plog.append(rec)
        print(f"{si+1}/{len(specs)} fails={n_fail}", flush=True)
        out = {"meta": {"date": str(datetime.date.today()),
                        "script": "src/p3_r2_regen.py",
                        "system": {"x5": "X5 ZETA (ICML'24)",
                                   "x6": "X6 Audio-Omni (SIGGRAPH'26)"}[
                                       args.system],
                        "split": args.split, "ledger": args.ledger,
                        "n_specs": len(specs), "n_done": si + 1,
                        "n_fail": n_fail, "chains": C.CHAIN_DEF,
                        "wav_dir": str(odir),
                        "notes": "A14.3: combined = one instruction with"
                                 " both changes; seq = two instructions"
                                 " applied one after the other; edited"
                                 " mixtures LS-fitted to y (main-ledger"
                                 " concession); E4 (room) added in the"
                                 " revision with the prompt 'in a less/"
                                 "more reverberant room' (ZETA) or 'make"
                                 " the room less/more reverberant'"
                                 " (Audio-Omni)",
                        "edits_run": sorted(set(edits) | set(
                            prev["meta"].get("edits_run", []) if prev
                            else []))},
               "specs": specs, "prompts": plog, "rows": rows,
               "edits": {e: {k: agg(rows[e], k) for k in
                             ("sisdr", "mrstft", "loc")}
                         for e in rows if rows[e]}}
        outp.write_text(json.dumps(out, indent=1))
    print(json.dumps({e: (round(out["edits"][e]["sisdr"]["median"], 2),
                          out["edits"][e]["sisdr"]["n"])
                      for e in out["edits"]}, indent=1))


if __name__ == "__main__":
    main()
