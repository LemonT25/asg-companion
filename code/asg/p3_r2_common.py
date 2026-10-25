"""A14 shared harness: chained edits, references, waveform saving.

Used by p3_r2_run.py (ours + X1..X4 identity systems) and by the
regeneration-family runners (X5/X6). All systems write their edited
waveforms to p3/r2_out/<system>/<scene>_<edit>.wav so that metrics
(SI-SDR, mrSTFT, locality, CLAP, RT60) are computed afterwards by
p3_r2_metrics.py from identical files.

Edits: E1..E4 as in p3_edit_eval (same pre-registered specs) and the
chains C1..C4 (A14.3). Chain specs are derived deterministically from
the single-edit specs; extra slot draws use seed 14.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, "/workspace/CF_ASG/src")
sys.path.insert(0, "/workspace/CF_PhysAudio/src")
sys.path.insert(0, "/workspace/audiocf_render")
from p3_train import (SceneSet, Estimator, theta_denorm_t, SEG,  # noqa: E402
                      ALPHA_NODES, MAXK)
from p3_edit_eval import (si_sdr, mrstft_np, render_node, pipeline_dry,  # noqa
                          posterior_stems, posterior_stems_own, locality,
                          wiener_dereverb, pit_by_sisdr)
from p2_common import phi_reverb  # noqa: E402

ROOT = Path("/workspace/CF_ASG")
PN = ROOT / "paper_numbers"
CK = ROOT / "p3" / "ckpt"
R2_OUT = ROOT / "p3" / "r2_out"
R2_REF = ROOT / "p3" / "r2_ref"
SR = 16000
R_FACTORS = [0.5, 2.0, 4.0]
EDITS = ("E1", "E2", "E3", "E4", "C1", "C2", "C3", "C4")
CHAIN_DEF = {"C1": "E1+E3", "C2": "E2+E3", "C3": "E1+E4", "C4": "E1+E2"}


def q_of(cls):
    m = {"Vehicle_horn_and_car_horn_and_honking": "car horn",
         "Gunshot_and_gunfire": "gunshot", "Thump_and_thud": "thump"}
    return m.get(cls, cls.replace("_", " ").lower())


def load_scene(ds, i):
    y, stems, dries, exist, traj, thn, alpha, env, K = ds[int(i)]
    meta = json.loads(str(np.load(ds.files[int(i)],
                                  allow_pickle=True)["meta"]))
    return {"y": y, "yn": y.numpy().astype(np.float64),
            "gts": stems.numpy().astype(np.float64),
            "gtd": dries.numpy().astype(np.float64),
            "traj": traj, "thn": thn, "alpha": float(alpha),
            "env": float(env), "K": int(K), "meta": meta, "scene_idx": int(i),
            "classes": [m["cls"] for m in meta]}


def theta_of(sc, k):
    return float(theta_denorm_t(sc["traj"][k][None], sc["thn"][k][None]))


def traj_name(sc, k):
    return "passby" if int(sc["traj"][k]) == 0 else "static"


# ---------------------------------------------------------------- specs
def chain_specs(ds, specs, seed=14):
    """Derive C1..C4 from the pre-registered single-edit specs.
    Returns a new list of specs (copies) with C*_ fields added."""
    rng = np.random.default_rng(seed)
    out = []
    for spec in specs:
        s = dict(spec)
        sc = load_scene(ds, spec["scene"])
        K = sc["K"]
        static = [k for k in range(K) if int(sc["traj"][k]) == 1]
        passby = [k for k in range(K) if int(sc["traj"][k]) == 0]
        # C1 = E1 + E3: remove a (= E1_slot), re-distance static b != a
        if "E1_slot" in s:
            a = s["E1_slot"]
            cand = [b for b in static if b != a]
            if cand:
                if "E3_slot" in s and s["E3_slot"] != a:
                    b, fm = s["E3_slot"], s["E3_factor"]
                else:
                    b = int(cand[int(rng.integers(len(cand)))])
                    fm = float(R_FACTORS[int(rng.integers(len(R_FACTORS)))])
                s["C1_rm"], s["C1_slot"], s["C1_factor"] = a, b, fm
        # C2 = E2 + E3: both single specs exist (open scene, both types)
        if "E2_slot" in s and "E3_slot" in s:
            s["C2_v_slot"], s["C2_newv"] = s["E2_slot"], s["E2_newv"]
            s["C2_r_slot"], s["C2_factor"] = s["E3_slot"], s["E3_factor"]
        # C3 = E1 + E4: room scene with K >= 2
        if "E1_slot" in s and "E4_newa" in s:
            s["C3_rm"], s["C3_newa"] = s["E1_slot"], s["E4_newa"]
        # C4 = E1 + E2: open scene K >= 2, remove b != pass-by a
        if "E2_slot" in s and K >= 2:
            a = s["E2_slot"]
            if "E1_slot" in s and s["E1_slot"] != a:
                b = s["E1_slot"]
            else:
                cand = [b for b in range(K) if b != a]
                b = int(cand[int(rng.integers(len(cand)))])
            s["C4_v_slot"], s["C4_newv"], s["C4_rm"] = a, s["E2_newv"], b
        out.append(s)
    return out


# ----------------------------------------------------------- references
def references(sc, spec):
    """Closed-form GT references for every edit present in spec.
    Returns {edit: (ref, noedit_ref)}; noedit_ref is the non-target bed
    used by the locality metric (for E4/C3: the mixture itself)."""
    yn, gts, gtd, K = sc["yn"], sc["gts"], sc["gtd"], sc["K"]
    al, env = sc["alpha"], sc["env"]
    R = {}
    if "E1_slot" in spec:
        k = spec["E1_slot"]
        R["E1"] = (yn - gts[k], yn - gts[k])
    if "E2_slot" in spec:
        k, nv = spec["E2_slot"], spec["E2_newv"]
        ned = yn - gts[k]
        R["E2"] = (ned + render_node(gtd[k], "passby", nv, al, env), ned)
    if "E3_slot" in spec:
        k, fm = spec["E3_slot"], spec["E3_factor"]
        ned = yn - gts[k]
        R["E3"] = (ned + gts[k] / fm, ned)
    if "E4_newa" in spec:
        na = spec["E4_newa"]
        ref = sum(render_node(gtd[j], traj_name(sc, j), theta_of(sc, j),
                              na, 1.0) for j in range(K))
        R["E4"] = (ref, yn)
    if "C1_rm" in spec:
        a, b, fm = spec["C1_rm"], spec["C1_slot"], spec["C1_factor"]
        ned = yn - gts[a] - gts[b]
        R["C1"] = (ned + gts[b] / fm, ned)
    if "C2_v_slot" in spec:
        a, nv = spec["C2_v_slot"], spec["C2_newv"]
        b, fm = spec["C2_r_slot"], spec["C2_factor"]
        ned = yn - gts[a] - gts[b]
        R["C2"] = (ned + render_node(gtd[a], "passby", nv, al, env)
                   + gts[b] / fm, ned)
    if "C3_rm" in spec:
        a, na = spec["C3_rm"], spec["C3_newa"]
        ref = sum(render_node(gtd[j], traj_name(sc, j), theta_of(sc, j),
                              na, 1.0) for j in range(K) if j != a)
        R["C3"] = (ref, yn - gts[a])
    if "C4_v_slot" in spec:
        a, nv, b = spec["C4_v_slot"], spec["C4_newv"], spec["C4_rm"]
        ned = yn - gts[a] - gts[b]
        R["C4"] = (ned + render_node(gtd[a], "passby", nv, al, env), ned)
    return R


# ------------------------------------------------ identity-system edits
class Heads:
    """Our estimation heads (shared by ours and X1..X4)."""

    def __init__(self, dev):
        from p3_train import Est2
        self.dev = dev
        import os
        self.e2 = Est2().to(dev)
        if os.environ.get("VEL3_CKPT"):      # R1: vel3 speed head (src/p3_vel3.py)
            from p3_vel3 import Vel3AsEst2
            self.e2 = Vel3AsEst2(CK / os.environ["VEL3_CKPT"], dev)
            print(f"vel3 speed head {os.environ['VEL3_CKPT']} step {self.e2.step}", flush=True)
        else:
            self.e2.load_state_dict(torch.load(CK / "est2_latest.pt",
                                               map_location=dev)["model"])
            self.e2.eval()
        self.est = Estimator().to(dev)
        self.est.load_state_dict(torch.load(CK / "est_latest.pt",
                                            map_location=dev)["model"])
        self.est.eval()

    @torch.no_grad()
    def velocities(self, stems):
        _, pv2 = self.e2(torch.tensor(np.stack(stems),
                                      dtype=torch.float32).to(self.dev))
        return [float(theta_denorm_t(torch.tensor([0]),
                                     pv2[k][None].cpu().clamp(0, 1)))
                for k in range(len(stems))]

    @torch.no_grad()
    def alpha(self, yn):
        import p3_edit_eval as _ee
        _ee.SCENE_LAM = None
        if os.environ.get("ALPHA_SEL") == "kurt":
            return _ee.select_alpha(np.asarray(yn, dtype=np.float64)[:SEG])
        _, _, pal = self.est(torch.tensor(yn, dtype=torch.float32)[None]
                             .to(self.dev))
        if os.environ.get("ALPHA_SEL") == "kurtj":
            a, l = _ee.select_alpha_joint(np.asarray(yn, dtype=np.float64)[:SEG], float(pal[0]))
            _ee.SCENE_LAM = l
            return a
        return float(pal[0])


_DEX = {}


def dex_pair(yn, stem, v, nv, K, dev="cuda"):
    """R1 speed-conditioned extraction for the re-velocity edit (ours only):
    returns (stem to subtract, node at the new speed) or None when the
    gate keeps the separator stem. Configured by env DEX_CKPT (checkpoint
    in p3/ckpt) and DEX_GATE (default K>1&agree<10, as in p3_edit_eval)."""
    name = os.environ.get("DEX_CKPT")
    if not name:
        return None
    if "model" not in _DEX:
        from p3_dex import load_dex
        _DEX["model"], step = load_dex(name, dev)
        _DEX["gate"] = os.environ.get("DEX_GATE", "K>1&agree<10")
        print(f"dex extractor {name} step {step} gate {_DEX['gate']}", flush=True)
    from p3_dex import dewarp_ng, rewarp_ng
    from p3_edit_eval import si_sdr
    with torch.no_grad():
        ty = torch.tensor(yn[:SEG], dtype=torch.float32, device=dev)[None]
        ts = torch.tensor(stem[:SEG], dtype=torch.float32, device=dev)[None]
        tv = torch.tensor([float(v)], device=dev)
        dd = _DEX["model"](dewarp_ng(ty, tv), dewarp_ng(ts, tv))
        stem_hat = rewarp_ng(dd, tv, tv)[0].cpu().numpy().astype(np.float64)
        node = rewarp_ng(dd, tv, torch.tensor([float(nv)], device=dev))[0].cpu().numpy().astype(np.float64)
    feats = {"K": K, "agree": si_sdr(stem_hat, stem[:SEG])}
    import re as _re
    for clause in _DEX["gate"].split("&"):
        m_ = _re.match(r"([a-zA-Z0-9_]+)([<>])(.*)", clause)
        fv, op, thr = feats[m_.group(1)], m_.group(2), float(m_.group(3))
        if not ((fv < thr) if op == "<" else (fv > thr)):
            return None
    return stem_hat, node


def identity_edits(sc, spec, ests, heads, alpha_hat=None, env_hat=None,
                   order_test=False, vel_stems=None):
    """Apply every edit in spec with the node identities on estimated
    stems `ests` (list of K arrays, mixture-consistent).
    alpha_hat/env_hat: the system's own structure estimates; for
    external stem systems the main-ledger convention (GT alpha/env
    granted) is kept when None is passed.
    Returns {edit: est} and, if order_test, {edit: max|diff| of the
    reverse order} for chains."""
    yn, K = sc["yn"], sc["K"]
    external = alpha_hat is None
    al = sc["alpha"] if external else alpha_hat
    env = sc["env"] if external else env_hat
    # room operator: every system uses OUR absorption estimate (the
    # paper's E4 path); external stems never enter the mixture-level
    # E4, so E4 is ours-only and C3 is the stem-dependent room chain.
    al_room = heads.alpha(yn) if external else al
    # velocity head input: main-ledger convention -- ours reads the
    # separator's raw slot outputs (p3_edit_eval S3: e2model(dsep)),
    # external stem systems read their LS-fitted posterior stems
    # (p3_baselines_ext: e2model(ests)).
    vh = heads.velocities(ests if vel_stems is None else vel_stems)
    E, D = {}, {}

    def revel(k, nv):
        d = pipeline_dry(ests[k], "passby", vh[k], al, env)
        return render_node(d, "passby", nv, al, env)

    def revel_pair(k, nv):
        """(stem to subtract, node at nv): the extractor path when it is
        configured and its gate opens (ours only), else the separator stem."""
        if not external and env < 0.5:
            r = dex_pair(yn, ests[k], vh[k], nv, K)
            if r is not None:
                return r
        return ests[k], revel(k, nv)

    def reroom(x, na):
        nn_ = min(ALPHA_NODES, key=lambda t: abs(t - na))
        return phi_reverb(wiener_dereverb(x, al_room)[:SEG], nn_)[:SEG]

    if "E1_slot" in spec:
        E["E1"] = yn - ests[spec["E1_slot"]]
    if "E2_slot" in spec:
        k = spec["E2_slot"]
        s_, n_ = revel_pair(k, spec["E2_newv"])
        E["E2"] = yn - s_ + n_
    if "E3_slot" in spec:
        k, fm = spec["E3_slot"], spec["E3_factor"]
        E["E3"] = yn - ests[k] + ests[k] / fm
    if "E4_newa" in spec and not external:
        E["E4"] = reroom(yn, spec["E4_newa"])
    if "C1_rm" in spec:
        a, b, fm = spec["C1_rm"], spec["C1_slot"], spec["C1_factor"]
        E["C1"] = yn - ests[a] - ests[b] + ests[b] / fm
        if order_test:
            alt = (yn - ests[b] + ests[b] / fm) - ests[a]
            D["C1"] = float(np.max(np.abs(alt - E["C1"])))
    if "C2_v_slot" in spec:
        a, nv = spec["C2_v_slot"], spec["C2_newv"]
        b, fm = spec["C2_r_slot"], spec["C2_factor"]
        s_a, node = revel_pair(a, nv)
        E["C2"] = yn - s_a - ests[b] + node + ests[b] / fm
        if order_test:
            alt = (yn - ests[b] + ests[b] / fm) - s_a + node
            D["C2"] = float(np.max(np.abs(alt - E["C2"])))
    if "C3_rm" in spec:
        a, na = spec["C3_rm"], spec["C3_newa"]
        E["C3"] = reroom(yn - ests[a], na)
        if order_test:
            alt = reroom(yn, na) - reroom(ests[a], na)
            D["C3"] = float(np.max(np.abs(alt - E["C3"])))
    if "C4_v_slot" in spec:
        a, nv, b = spec["C4_v_slot"], spec["C4_newv"], spec["C4_rm"]
        s_a, node = revel_pair(a, nv)
        E["C4"] = yn - s_a - ests[b] + node
        if order_test:
            alt = (yn - ests[b]) - s_a + node
            D["C4"] = float(np.max(np.abs(alt - E["C4"])))
    return E, D


# ------------------------------------------------------------- scoring
def score(est, ref, ned):
    return {"sisdr": si_sdr(est, ref), "mrstft": mrstft_np(est, ref),
            "loc": locality(est, ref, ned)}


def save_wav(path, x):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(x, np.float32), SR)


def save_refs(sc, spec, refs):
    """Mixture and every reference, once per scene."""
    i = spec["scene"]
    p = R2_REF / f"{i:05d}_mix.wav"
    if not p.exists():
        save_wav(p, sc["yn"])
    for e, (ref, _) in refs.items():
        q = R2_REF / f"{i:05d}_{e}_ref.wav"
        if not q.exists():
            save_wav(q, ref)


def agg(rows, key="sisdr", rng=None):
    rng = rng or np.random.default_rng(0)
    if rows and not isinstance(rows[0], dict):
        v = [float(x) for x in rows]
    else:
        v = [r[key] for r in rows if key in r and r[key] is not None]
    if not v:
        return None
    m = np.median(rng.choice(v, (5000, len(v))), axis=1)
    return {"median": float(np.median(v)),
            "ci95": [float(np.percentile(m, 2.5)),
                     float(np.percentile(m, 97.5))], "n": len(v)}
