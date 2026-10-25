"""A11 protocol R: R-E1 / R-E3 on the real mixture set (p3/rset/).

Edits (full-reference; mix = stem_vehicle + stem_distractor exactly):
  R-E1a keep-vehicle:      est = extracted vehicle stem,
                           ref = stem_vehicle.
  R-E1b remove-distractor: est = mix - extracted distractor stem,
                           ref = mix - stem_distractor (= stem_vehicle;
                           the two directions differ only in the
                           system's extraction path).
  R-E3  distractor 2x farther (gain 1/2):
                           est = mix - s_hat_dist + s_hat_dist/2,
                           ref = mix - stem_dist + stem_dist/2.
Metrics: SI-SDR + MRSTFT vs the reference.

Systems / extraction rules (documented):
  ours (sep2 ownership front-end, p3/ckpt/sep2_latest.pt):
    slots on the mix -> ownership posterior stems (A8,
    posterior_stems_own) over active slots (exist > 0.5; if none, all
    slots treated active).  Vehicle slot rule (fixed pre-measurement on
    the 8-mixture smoke probe, logged in logs/r_e1e3_ours_smoke.log):
    argmax over active slots of p_passby * RMS, where p_passby is the
    passby softmax of the REAL-DOMAIN est2 (p3/ckpt/est2_vs13.pt
    trajectory head) evaluated on the posterior stem, and RMS is the
    stem's energy.  (sep2's own synthetic-trained trajectory head is
    unreliable on real audio -- logged per slot as diagnostic.)
    Distractor
    estimate = mix - vehicle stem (the ownership decomposition is
    mixture-consistent, so this is the complement); consequently R-E1b
    coincides with R-E1a for ours by construction (noted in ledger).
  x1 AudioSep / x2 FlowSep / x3 SoloAudio / x4 OmniSep:
    query-based extraction with the SAME wiring as the synthetic
    harnesses; vehicle query = "car passing by", distractor query = the
    clip's FSD class name (GT class query, a disclosed concession);
    each stem least-squares scale-fitted to the mixture (disclosed).
    DEVIATION from the synthetic protocol (documented): the joint
    Wiener consistency projection is NOT applied here, because it
    forces the two extractions to partition the mixture, collapsing
    R-E1a and R-E1b into the same estimate.  Instead each external
    stem is re-synthesized PER-STEM with the mixture phase (STFT
    1024/256 hann, magnitude from the stem estimate, phase from the
    mixture) -- a relaxation of A5.1 that repairs the phase
    misalignment of generative decoders (FlowSep/SoloAudio) without
    forcing a partition; applied uniformly to X1-X4, then LS
    scale-fitted; disclosed.  Ours is untouched (its posterior stems
    are mixture-phase consistent by construction).

Run per system in ITS venv (each writes a partial ledger), then
aggregate:
  audiocf  : src/p3_r_eval_ours.py --system ours
  audiocf  : src/p3_r_eval_ours.py --system x1
  flowsep  : src/p3_r_eval_ours.py --system x2
  soloaudio: src/p3_r_eval_ours.py --system x3
  omnisep  : src/p3_r_eval_ours.py --system x4
  any      : src/p3_r_eval_ours.py --aggregate
-> p3/rset_out/r_e1e3_<sys>.json, paper_numbers/p3_r_e1e3_v0.json
"""
import argparse
import datetime
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, "/workspace/CF_ASG/src")
from p3_r_common import (ROUT, PN, CK, SR, SEG, VEH_QUERY,  # noqa: E402
                         si_sdr, agg, list_rset, load_mix)

SYSTEMS = ("ours", "x1", "x2", "x3", "x4")
SYS_LABEL = {"ours": "ours (sep3 mask front end, R1)", "ours2": "ours (sep2 ownership)",
             "x1": "X1 AudioSep (ICLR'24)",
             "x2": "X2 FlowSep (ICASSP'25)",
             "x3": "X3 SoloAudio (ICASSP'25)",
             "x4": "X4 OmniSep (ICLR'25)"}


def ls_fit(w, yn):
    a = float(np.dot(w, yn) / (np.dot(w, w) + 1e-12))
    return a * w


def mixphase(est, mx):
    """Per-stem mixture-phase re-synthesis (A5.1 relaxation, see module
    docstring): |STFT(est)| with angle(STFT(mix)), same STFT params as
    posterior_stems (1024/256 hann)."""
    import torch
    w = torch.hann_window(1024)
    t = torch.tensor(np.stack([est[:SEG], mx[:SEG]]),
                     dtype=torch.float32)
    Z = torch.stft(t, 1024, 256, window=w, return_complex=True)
    Y = Z[0].abs() * torch.exp(1j * Z[1].angle())
    return torch.istft(Y[None], 1024, 256, window=w,
                       length=SEG)[0].numpy().astype(np.float64)


def ext_stem(w, mx):
    return ls_fit(mixphase(w, mx), mx)


# ---------------- per-system extractor factories ----------------
def make_ours(dev):
    import torch
    from p3_train import SceneModel, Est2, MAXK
    from p3_edit_eval import posterior_stems_own
    m = SceneModel("sep2").to(dev)
    m.load_state_dict(torch.load(CK / "sep2_latest.pt",
                                 map_location=dev,
                                 weights_only=False)["model"])
    m.eval()
    e2 = Est2().to(dev)
    e2.load_state_dict(torch.load(CK / "est2_vs13.pt",
                                  map_location=dev,
                                  weights_only=False)["model"])
    e2.eval()

    def extract(mx, query_dist):
        y = torch.tensor(mx, dtype=torch.float32)
        with torch.no_grad():
            d, e, trj, tn, al, ev = m.slots(y[None].to(dev))
        d = d[0, ..., :SEG].cpu()
        esig = torch.sigmoid(e[0]).cpu()
        if not bool((esig > 0.5).any()):
            esig = torch.ones_like(esig)     # fallback: all active
        post = posterior_stems_own(m, mx, [d[k].numpy()
                                           for k in range(MAXK)],
                                   esig, dev)
        with torch.no_grad():
            tr2, _ = e2(torch.tensor(np.stack(post),
                                     dtype=torch.float32).to(dev))
        p_pb = torch.softmax(tr2, -1)[:, 0].cpu().numpy()
        p_sep = torch.softmax(trj[0].cpu(), -1)[:, 0].numpy()
        diag, best, bs = [], None, -1.0
        for k in range(MAXK):
            act = bool(esig[k] > 0.5)
            r = float(np.sqrt((post[k] ** 2).mean()))
            s = float(p_pb[k]) * r
            diag.append({"slot": k, "active": act,
                         "p_passby_est2vs13": float(p_pb[k]),
                         "p_passby_sep2head": float(p_sep[k]),
                         "rms": r, "score": s})
            if act and s > bs:
                bs, best = s, k
        veh = post[best]
        dist = mx - veh                      # mixture-consistent complement
        return veh, dist, {"slots": diag, "vehicle_slot": best,
                           "e1b_note": "== e1a by mixture consistency"}
    return extract


def make_x1(dev):
    import torch
    import torchaudio
    sys.path.insert(0, "/workspace/CF_AQA/AudioSep")
    _orig = torch.load
    torch.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": False})
    os.chdir("/workspace/CF_AQA/AudioSep")
    from pipeline import build_audiosep, separate_audio
    ckpt = ("/workspace/hf_cache/hub/models--audo--AudioSep/snapshots/"
            "891454f4cb60363dd67fdef052508d4866ad42f3/"
            "audiosep_base_4M_steps.ckpt")
    model = build_audiosep("config/audiosep_base.yaml", ckpt, dev)
    td = tempfile.mkdtemp(prefix="p3rx1_")

    def one(mp, query, mx):
        import soundfile as sf  # noqa: F401
        op = f"{td}/est.wav"
        separate_audio(model, mp, query, op, dev)
        w, sr = torchaudio.load(op)
        w16 = torchaudio.functional.resample(
            w.mean(0), sr, SR).numpy().astype(np.float64)[:SEG]
        if len(w16) < SEG:
            w16 = np.pad(w16, (0, SEG - len(w16)))
        return ext_stem(w16, mx)

    def extract(mx, query_dist):
        import soundfile as sf
        mp = f"{td}/mix.wav"
        sf.write(mp, mx.astype(np.float32), SR)
        veh = one(mp, VEH_QUERY, mx)
        dist = one(mp, query_dist, mx)
        return veh, dist, {}
    return extract


def make_x2(dev):
    from p3_x2_flowsep import build_flowsep, separate
    model, ds_feat = build_flowsep(dev)
    td = tempfile.mkdtemp(prefix="p3rx2_")

    def one(mp, query, mx):
        w = separate(model, ds_feat, mp, query, dev, 20, td)
        w16 = w.astype(np.float64)[:SEG]
        if len(w16) < SEG:
            w16 = np.pad(w16, (0, SEG - len(w16)))
        return ext_stem(w16, mx)

    def extract(mx, query_dist):
        import soundfile as sf
        mp = f"{td}/mix.wav"
        sf.write(mp, mx.astype(np.float32), SR)
        return one(mp, VEH_QUERY, mx), one(mp, query_dist, mx), {}
    return extract


def make_x3(dev):
    from p3_x3_soloaudio import build_soloaudio, solo_extract
    M = build_soloaudio(dev)

    def extract(mx, query_dist):
        veh = ext_stem(solo_extract(M, mx, VEH_QUERY, dev), mx)
        dist = ext_stem(solo_extract(M, mx, query_dist, dev), mx)
        return veh, dist, {}
    return extract


def make_x4(dev):
    from p3_x4_omnisep import build_omnisep, omni_extract
    M = build_omnisep(dev)

    def extract(mx, query_dist):
        veh = ext_stem(omni_extract(M, mx, VEH_QUERY, dev), mx)
        dist = ext_stem(omni_extract(M, mx, query_dist, dev), mx)
        return veh, dist, {}
    return extract


def make_ours_sep3(dev):
    """R1 adopted front end: sep3 masked slots (env SEP3_CKPT) through the
    Wiener consistency projection; the vehicle slot is chosen by the same
    rule as before (argmax p_passby(est2_vs13) * RMS over the slots); the
    distractor is the mixture-consistent complement."""
    import os
    import torch
    from p3_train import Est2, MAXK
    from p3_sep3 import Sep3
    from p3_edit_eval import posterior_stems
    name = os.environ.get("SEP3_CKPT", "sep3_d3_best.pt")
    st = torch.load(CK / name, map_location=dev, weights_only=False)
    a = st["args"]
    m = Sep3(a["dim"], a["hidden"], a["blocks"], hop=a.get("hop", 128), mask_bound=a.get("mask_bound", 0.0),
             k_cond=a.get("k_cond", False)).to(dev)
    m.load_state_dict(st["model"])
    m.eval()
    print(f"sep3 front end {name} step {st.get('step')}", flush=True)
    e2 = Est2().to(dev)
    e2.load_state_dict(torch.load(CK / "est2_vs13.pt", map_location=dev, weights_only=False)["model"])
    e2.eval()

    def extract(mx, query_dist):
        y = torch.tensor(mx, dtype=torch.float32)
        with torch.no_grad():
            d = m(y[None].to(dev))[0, ..., :SEG].cpu().numpy().astype(np.float64)
        post = posterior_stems(mx, [d[k] for k in range(MAXK)])
        with torch.no_grad():
            tr2, _ = e2(torch.tensor(np.stack(post), dtype=torch.float32).to(dev))
        p_pb = torch.softmax(tr2, -1)[:, 0].cpu().numpy()
        diag, best, bs = [], None, -1.0
        for k in range(MAXK):
            r = float(np.sqrt((post[k] ** 2).mean()))
            s = float(p_pb[k]) * r
            diag.append({"slot": k, "active": True, "p_passby_est2vs13": float(p_pb[k]), "rms": r, "score": s})
            if s > bs:
                bs, best = s, k
        veh = post[best]
        dist = mx - veh
        return veh, dist, {"slots": diag, "vehicle_slot": best, "front_end": name,
                           "e1b_note": "== e1a by mixture consistency"}
    return extract


MAKERS = {"ours": make_ours_sep3, "ours2": make_ours, "x1": make_x1, "x2": make_x2,
          "x3": make_x3, "x4": make_x4}


def run_system(name, n_limit):
    import torch
    torch.set_num_threads(8)
    dev = "cuda"
    sys.path.insert(0, "/workspace/CF_PhysAudio/src")
    from p2_train import mrstft_loss

    def mrstft_np(a, b):
        ta = torch.tensor(a, dtype=torch.float32)[None]
        tb = torch.tensor(b, dtype=torch.float32)[None]
        L = min(ta.shape[-1], tb.shape[-1])
        return float(mrstft_loss(ta[:, :L], tb[:, :L]))

    files = list_rset()
    if n_limit > 0:
        files = files[:n_limit]
    extract = MAKERS[name](dev)
    rows, fails = [], []
    for fi, f in enumerate(files):
        m = load_mix(f)
        mx, sv, sd = m["mix"], m["stem_vehicle"], m["stem_distractor"]
        q = m["meta"]["fsd_query"]
        try:
            veh, dist, diag = extract(mx, q)
        except Exception as ex:
            fails.append({"idx": m["meta"]["idx"],
                          "error": f"{type(ex).__name__}: {ex}",
                          "trace": traceback.format_exc()[-600:]})
            print(f"FAIL {name} idx={m['meta']['idx']}: {ex}", flush=True)
            continue
        ref_a = sv                       # == mix - sd (additive)
        est_b = mx - dist
        e3_est = mx - dist + dist / 2.0
        e3_ref = mx - sd + sd / 2.0
        rows.append({
            "idx": m["meta"]["idx"], "snr_db": m["snr"],
            "veh_source": m["meta"]["veh_source"], "fsd_query": q,
            "e1a_sisdr": si_sdr(veh, ref_a),
            "e1a_mrstft": mrstft_np(veh, ref_a),
            "e1b_sisdr": si_sdr(est_b, ref_a),
            "e1b_mrstft": mrstft_np(est_b, ref_a),
            "e3_sisdr": si_sdr(e3_est, e3_ref),
            "e3_mrstft": mrstft_np(e3_est, e3_ref),
            "dist_stem_sisdr": si_sdr(dist, sd), **diag})
        if (fi + 1) % 20 == 0:
            print(f"{name} {fi + 1}/{len(files)}", flush=True)
    ROUT.mkdir(parents=True, exist_ok=True)
    out = {"meta": {"system": SYS_LABEL[name],
                    "date": str(datetime.date.today()),
                    "n": len(rows), "n_fail": len(fails)},
           "rows": rows, "fails": fails}
    (ROUT / f"r_e1e3_{name}.json").write_text(json.dumps(out, indent=1))
    for e in ("e1a", "e1b", "e3"):
        print(name, e, json.dumps(agg(rows, f"{e}_sisdr")))


def aggregate(out_name):
    led = {"meta": {
        "date": str(datetime.date.today()),
        "script": "src/p3_r_eval_ours.py",
        "protocol": "PROTOCOL_P3.md amendment A11, R-E1/R-E3",
        "rset": "p3/rset (120 additive real mixtures, seed 0)",
        "edits": {"e1a": "keep-vehicle: est = extracted vehicle stem",
                  "e1b": ("remove-distractor: est = mix - extracted "
                          "distractor; ref identical to e1a (additive "
                          "2-source mixture)"),
                  "e3": ("distractor 2x farther (gain 1/2): est = mix "
                         "- s_hat_d/2, ref = mix - stem_d/2")},
        "concessions": ("externals: GT FSD class-name distractor query, "
                        "'car passing by' vehicle query, per-stem "
                        "mixture-phase re-synthesis (A5.1 relaxation; "
                        "the joint Wiener projection would collapse "
                        "e1a/e1b) + LS scale fit to the mixture "
                        "(favor baselines, disclosed)"),
        "ours_rule": ("vehicle slot = argmax p_passby(est2_vs13 traj "
                      "head on the posterior stem) * RMS over active "
                      "posterior stems (rule frozen pre-measurement "
                      "on the smoke probe); distractor = mixture "
                      "complement; e1b == e1a by construction")},
        "systems": {}, "rows": {}}
    for name in SYSTEMS:
        p = ROUT / f"r_e1e3_{name}.json"
        if not p.exists():
            led["systems"][name] = {"status": "MISSING"}
            continue
        part = json.loads(p.read_text())
        rows = part["rows"]
        led["rows"][name] = rows
        led["systems"][name] = {
            "label": part["meta"]["system"],
            "n": len(rows), "n_fail": part["meta"]["n_fail"],
            "fails": [{k: f[k] for k in ("idx", "error")}
                      for f in part["fails"]],
            "edits": {e: {"sisdr": agg(rows, f"{e}_sisdr"),
                          "mrstft": agg(rows, f"{e}_mrstft")}
                      for e in ("e1a", "e1b", "e3")},
            "edits_by_source": {
                src: {e: agg([r for r in rows
                              if r["veh_source"] == src],
                             f"{e}_sisdr")
                      for e in ("e1a", "e1b", "e3")}
                for src in ("vs13", "idmt")},
            "dist_stem_sisdr": agg(rows, "dist_stem_sisdr")}
    (PN / out_name).write_text(json.dumps(led, indent=1))
    for name, s in led["systems"].items():
        if "edits" in s:
            print(name, {e: (None if s["edits"][e]["sisdr"] is None else
                             round(s["edits"][e]["sisdr"]["median"], 2))
                         for e in s["edits"]})
        else:
            print(name, s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=SYSTEMS)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--n_limit", type=int, default=0)
    ap.add_argument("--out", default="p3_r_e1e3_v0.json")
    args = ap.parse_args()
    if args.aggregate:
        aggregate(args.out)
    elif args.system:
        run_system(args.system, args.n_limit)
    else:
        ap.error("need --system or --aggregate")


if __name__ == "__main__":
    main()
