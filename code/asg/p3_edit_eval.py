"""P3 edit evaluation (E1-E3, full-reference) per PROTOCOL_P3.

Systems: S1 OURS-A (amortized), S2 OURS-AR (+exact-likelihood refinement),
S3 PIPE (sep -> est -> semi-analytic inversion -> Phi re-render),
S4 ORACLE-STEMS (GT stems + est heads; isolates estimation from separation).
All systems use the edit identity y' = y - stem_k + rerender(node_k').
References are exact renders of the GT scene graph with the edit applied.
-> paper_numbers/p3_edit_eval_v0.json
"""
import argparse
import datetime
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import mannwhitneyu, wilcoxon

import p3_pwg as pwg

sys.path.insert(0, "/workspace/CF_ASG/src")
sys.path.insert(0, "/workspace/CF_PhysAudio/src")
sys.path.insert(0, "/workspace/audiocf_render")
from p3_train import (SceneModel, Estimator, SceneSet, theta_denorm_t,  # noqa
                      theta_norm, ALPHA_NODES, SEG, MAXK)
from p2_train import mrstft_loss  # noqa: E402
from p2_common import SR, phi_velocity, phi_distance, phi_reverb  # noqa
from core import passby_positions, C_SOUND  # noqa: E402
from room import image_source_rir  # noqa: E402

ROOT = Path("/workspace/CF_ASG")
PN = ROOT / "paper_numbers"
CK = ROOT / "p3" / "ckpt"
V_GRID = [5.0, 10.0, 20.0, 30.0, 40.0]
R_FACTORS = [0.5, 2.0, 4.0]   # E3: relative re-distancing (far-field
                              # gain component; delay/absorption need
                              # absolute r, outside the identifiable set)
rng = np.random.default_rng(0)


def si_sdr(est, ref):
    ref = ref - ref.mean()
    est = est - est.mean()
    a = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    e = est - a * ref
    return float(10 * np.log10((a * ref @ (a * ref) + 1e-12)
                               / (e @ e + 1e-12)))


def locality(est, ref, noedit_ref):
    """A12.2 locality: how faithfully the non-edited content survives.
    The edit residual (est - ref) is treated as contamination of the
    non-target bed: loc = si_sdr((est - ref) + noedit_ref, noedit_ref).
    noedit_ref = y - gts[k] for E1/E2/E3 (the non-target content);
    for E4 the pre-edit mixture y is its own reference frame (every
    node is re-rendered, so no untouched bed exists).  Conservative:
    node-render error is charged to locality as well."""
    return si_sdr((est - ref) + noedit_ref, noedit_ref)


def mrstft_np(a, b):
    ta = torch.tensor(a, dtype=torch.float32)[None]
    tb = torch.tensor(b, dtype=torch.float32)[None]
    L = min(ta.shape[-1], tb.shape[-1])
    return float(mrstft_loss(ta[:, :L], tb[:, :L]))


# ---------- GT rendering helpers (reference edits) ----------
def render_node(d, traj, theta, alpha, env=1.0):
    w = phi_velocity(d, theta) if traj == "passby" else phi_distance(d, theta)
    if env > 0.5:
        w = phi_reverb(w[:SEG], alpha)[:SEG]
    return w[:SEG]


# A13: reference renders go through this hook so a physics-mismatch
# wrapper can point it at a perturbed renderer while the system keeps
# its own operators.
REF_RENDER = render_node

# ---------- pipeline semi-analytic inversion ----------
_RIRS = {a: image_source_rir((30.0, 20.0, 8.0), (18.0, 12.0, 1.5),
                             (15.0, 10.0, 1.6), SR, alpha=a, max_order=22,
                             rir_len_s=2.5) for a in ALPHA_NODES}


def wiener_dereverb(x, alpha):
    a = min(ALPHA_NODES, key=lambda t: abs(t - alpha))
    h = _RIRS[a]
    n = len(x) + len(h)
    X, H = np.fft.rfft(x, n), np.fft.rfft(h, n)
    lam = 1e-3 * np.max(np.abs(H) ** 2)
    d = np.fft.irfft(X * np.conj(H) / (np.abs(H) ** 2 + lam), n)
    return d[:len(x)]


def inverse_passby(x, v):
    """Undo warp + spreading of the pass-by (air absorption neglected in
    the inverse: bounded ~<1 dB in-band at these ranges)."""
    n = len(x)
    pos = passby_positions(n, SR, v=v, d0=2.0, t_close=n / SR / 2)
    r = np.linalg.norm(pos, axis=1)
    t = np.arange(n, dtype=np.float64)
    t_read = t - (r / C_SOUND) * SR          # emission sample for output t
    xs = x * r                                # undo spreading
    d = np.interp(t, t_read, xs, left=0.0, right=0.0)
    return d


def inverse_static(x, r):
    """Undo spreading gain and the constant propagation delay r/c."""
    shift = int(round((r / C_SOUND) * SR))
    d = np.zeros_like(x)
    if shift < len(x):
        d[:len(x) - shift] = x[shift:]
    return d * r


def pipeline_dry(stem, traj, theta, alpha, env):
    d = wiener_dereverb(stem, alpha) if env > 0.5 else stem
    if traj == "passby":
        return inverse_passby(d, theta)
    return stem if env > 0.5 else d   # static: dry only needed for E2


# ---------- model-side helpers ----------
@torch.no_grad()
def graph_infer(model, y, dev):
    d, e, trj, tn, al, ev = model.slots(y[None].to(dev))
    d = d[..., :SEG]
    env = (torch.sigmoid(ev) > 0.5).float()
    stems, _ = model.compose(d, e, trj, tn, al, env)
    return (d[0].cpu(), torch.sigmoid(e[0]).cpu(), trj[0].argmax(-1).cpu(),
            tn[0].cpu(), float(al[0]), float(env[0]),
            stems[0, ..., :SEG].cpu())


def posterior_stems_own(model, yn, stems, exist, dev):
    """A8 posterior stems: masks from the supervised ownership head
    (softmax over active slots), phase from the mixture."""
    w = torch.hann_window(1024, device=dev)
    t = torch.tensor(np.stack([yn[:SEG]] + [s[:SEG] for s in stems]),
                     dtype=torch.float32, device=dev)
    Z = torch.stft(t, 1024, 256, window=w, return_complex=True)
    Pm, Pp = Z[0:1].abs().pow(2), Z[1:].abs().pow(2)
    feats = torch.cat([Pp, Pm], 0).clamp_min(1e-10).log()[None]
    with torch.no_grad():
        logits = model.own(feats)[0]
    act = (exist > 0.5).to(dev)
    logits = logits.masked_fill(~act[:, None, None], -1e4)
    M = torch.softmax(logits, 0)
    post = torch.istft(M * Z[0][None], 1024, 256, window=w.cpu().to(dev),
                       length=SEG)
    return [p.cpu().numpy().astype(np.float64) for p in post]


def posterior_stems(yn, stems):
    """Wiener consistency projection (A5.1): posterior stem means given
    the observed mixture under a Gaussian residual model -- masks from
    the model's stem spectra, phase from the mixture. Sums to y."""
    t = torch.tensor(np.stack([yn[:SEG]] + [s[:SEG] for s in stems]),
                     dtype=torch.float32)
    w = torch.hann_window(1024)
    Z = torch.stft(t, 1024, 256, window=w, return_complex=True)
    P = Z[1:].abs() ** 2
    M = P / (P.sum(0, keepdim=True) + 1e-10)
    post = torch.istft(M * Z[0][None], 1024, 256, window=w,
                       length=SEG)
    return [p.numpy().astype(np.float64) for p in post]


def _residual(model, yd, d, e, tj, tn, al, env):
    stems = compose_free(model, d, e, tj, tn, al, env)
    return float(mrstft_loss(stems.sum(0)[None, :SEG], yd[None, :SEG]))


def refine(model, y, d0, trj_id, exist, tn0, al0, env0, dev, T=80,
           lr=5e-3):
    """Exact-likelihood refinement (A5.2). Discrete structure by
    likelihood model selection (env in {0,1}); theta by global 1-D
    grid line-search per pass-by slot (likelihood oscillatory in v);
    then joint Adam polish of (d, theta_n, alpha). Existence and
    trajectory types held fixed. Returns refined env."""
    yd = y.to(dev)
    e_fix = exist.to(dev)
    tj = trj_id.to(dev)
    best = None
    for env in (0.0, 1.0):
        d = d0.clone().to(dev)
        tn = tn0.clone().to(dev)
        al = torch.tensor([al0], device=dev)
        with torch.no_grad():
            for k in range(MAXK):
                if int(tj[k]) != 0 or float(e_fix[k]) <= 0.5:
                    continue
                cands = torch.linspace(0, 1, 29, device=dev)
                rs = []
                for c in cands:
                    tt = tn.clone()
                    tt[k] = c
                    rs.append(_residual(model, yd, d, e_fix, tj, tt,
                                        al, env))
                tn[k] = cands[int(np.argmin(rs))]
        d.requires_grad_(True)
        tn.requires_grad_(True)
        al.requires_grad_(True)
        opt = torch.optim.Adam([{"params": [d], "lr": lr},
                                {"params": [tn, al], "lr": 2e-3}])
        for _ in range(T):
            stems = compose_free(model, d, e_fix, tj, tn, al, env)
            loss = mrstft_loss(stems.sum(0)[None, :SEG], yd[None, :SEG])
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                tn.clamp_(0.0, 1.0)
                al.clamp_(ALPHA_NODES[0] + 1e-3, ALPHA_NODES[-1] - 1e-3)
        with torch.no_grad():
            stems = compose_free(model, d, e_fix, tj, tn, al, env)
            resid = float(mrstft_loss(stems.sum(0)[None, :SEG],
                                      yd[None, :SEG]))
        if best is None or resid < best[0]:
            best = (resid, d.detach().cpu(), tn.detach().cpu(),
                    float(al), float(env),
                    stems.detach().cpu()[..., :SEG])
    return best[1], best[2], best[3], best[4], best[5]


def compose_free(model, d, e, trj_id, tn, al, env):
    """Per-slot render with hard existence/trajectory (K,N tensors)."""
    from phi_torch import phi_velocity_t, phi_distance_t
    stems = []
    for k in range(MAXK):
        dk = d[k][None]
        th = theta_denorm_t(trj_id[k][None], tn[k][None].clamp(0, 1))
        if int(trj_id[k]) == 0:
            w = phi_velocity_t(dk, th.clamp(5, 40))
        else:
            w = phi_distance_t(dk, th.clamp(1, 32))
        if env > 0.5:
            w = model.rir(w[:, :SEG], al.expand(1))[:, :SEG]
        stems.append(w[0, :SEG] * e[k])
    return torch.stack(stems)


V_SEARCH = np.linspace(5.0, 40.0, 36)


def theta_linesearch(po_k, dprior):
    """A6: physics-inversion velocity search against a learned dry
    prior -- v_hat = argmin_v mrstft(Phi^-1_v(stem), d_prior), with
    parabolic refinement around the grid minimum.  Diagnostic
    p3_diag_theta_v0: rel 0.013 (GT prior) / 0.11-0.13 (5-10 dB
    prior) vs 0.33 (feedforward est head)."""
    costs = np.array([mrstft_np(inverse_passby(po_k, float(v))[:SEG],
                                dprior[:SEG]) for v in V_SEARCH])
    i = int(np.argmin(costs))
    v = float(V_SEARCH[i])
    if 0 < i < len(V_SEARCH) - 1:
        c0, c1, c2 = costs[i - 1], costs[i], costs[i + 1]
        den = c0 - 2 * c1 + c2
        if den > 1e-12:
            v += 0.5 * (c0 - c2) / den * (V_SEARCH[1] - V_SEARCH[0])
    return float(np.clip(v, V_SEARCH[0], V_SEARCH[-1])), costs


def assign_dry_slots(po, ddry, K, is_pb):
    """Match posterior stems to dry-model slots: pair cost is the
    line-search minimum for pass-by slots (best-v inversion residual
    against that dry prior), plain mrstft for static slots.  Returns
    per-po-slot best dry index and v_hat (0 for statics)."""
    C = np.full((K, len(ddry)), np.inf)
    V = np.zeros((K, len(ddry)))
    for k in range(K):
        for j in range(len(ddry)):
            if is_pb[k]:
                v, costs = theta_linesearch(po[k], ddry[j])
                C[k, j] = float(costs.min())
                V[k, j] = v
            else:
                C[k, j] = mrstft_np(po[k][:SEG], ddry[j][:SEG])
    best, bp = np.inf, None
    for p in itertools.permutations(range(len(ddry)), K):
        c = sum(C[k, p[k]] for k in range(K))
        if c < best:
            best, bp = c, p
    return [int(j) for j in bp], [float(V[k, bp[k]]) for k in range(K)]


def rerender_node_t(model, d_k, traj, theta_phys, alpha, env, dev):
    """Re-render one node at new physical theta (torch, via model.rir)."""
    from phi_torch import phi_velocity_t, phi_distance_t
    dk = d_k[None].to(dev)
    th = torch.tensor([theta_phys], device=dev)
    if traj == "passby":
        w = phi_velocity_t(dk, th)
    else:
        w = phi_distance_t(dk, th)
    if env > 0.5:
        al = torch.tensor([alpha], device=dev)
        w = model.rir(w[:, :SEG], al)[:, :SEG]
    return w[0, :SEG].cpu().numpy().astype(np.float64)


def pit_by_sisdr(pred_stems, gt_stems, K):
    perms = list(itertools.permutations(range(MAXK)))
    best, bi = -1e9, perms[0]
    for p in perms:
        v = sum(si_sdr(pred_stems[p[k]], gt_stems[k]) for k in range(K))
        if v > best:
            best, bi = v, p
    return bi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_scenes", type=int, default=250)
    ap.add_argument("--refine_T", type=int, default=80)
    ap.add_argument("--split", default="val",
                    help="SceneSet split (val | conf1 confirmation cohort)")
    ap.add_argument("--out", default="p3_edit_eval_v0.json")
    args = ap.parse_args()
    dev = "cuda"
    torch.set_num_threads(8)

    gmodel = SceneModel("graph").to(dev)
    gmodel.load_state_dict(torch.load(CK / "graph_latest.pt",
                                      map_location=dev)["model"])
    gmodel.eval()
    if (CK / "sep2_latest.pt").exists():   # A8
        smodel = SceneModel("sep2").to(dev)
        smodel.load_state_dict(torch.load(CK / "sep2_latest.pt",
                                          map_location=dev)["model"])
    else:
        smodel = SceneModel("sep").to(dev)
        smodel.load_state_dict(torch.load(CK / "sep_latest.pt",
                                          map_location=dev)["model"])
    smodel.eval()
    emodel = Estimator().to(dev)
    emodel.load_state_dict(torch.load(CK / "est_latest.pt",
                                      map_location=dev)["model"])
    emodel.eval()
    from p3_train import Est2
    e2model = Est2().to(dev)
    e2model.load_state_dict(torch.load(CK / "est2_latest.pt",
                                       map_location=dev)["model"])
    e2model.eval()

    ds = SceneSet(args.split)
    idxs = rng.choice(len(ds), size=min(args.n_scenes, len(ds)),
                      replace=False)
    R = {sys_: {e: [] for e in ("E1", "E2", "E3", "E4")}
         for sys_ in ("S1", "S2", "S3", "S4", "S2P", "ORE4A")}
    specs = []
    inf = {"stem_sisdr": {"S1": [], "S2": [], "S3": []},
           "theta_abserr": {"S1": [], "S2": [], "S3": []},
           "traj_acc": {"S1": [], "S3": []},
           "alpha_abserr": {"S1": [], "S2": [], "S3": []},
           "exist": {"S1": {"tp": 0, "fp": 0, "fn": 0},
                     "S3": {"tp": 0, "fp": 0, "fn": 0}}}

    for si, i in enumerate(idxs):
        y, stems, dries, exist, traj, thn, alpha, env, K = ds[int(i)]
        yn = y.numpy().astype(np.float64)
        gts = stems.numpy().astype(np.float64)
        gtd = dries.numpy().astype(np.float64)
        alpha, env = float(alpha), float(env)
        K = int(K)

        # ---- S1 amortized (env predicted)
        d1, e1, trj1, tn1, al1, env1, st1 = graph_infer(gmodel, y, dev)
        # ---- S2 refined (env selected by exact likelihood, A5.2)
        d2, tn2, al2, env2, st2 = refine(gmodel, y, d1, trj1, e1, tn1,
                                         al1, env1, dev,
                                         T=args.refine_T)
        # ---- S3 pipeline (env from its own separator's head)
        with torch.no_grad():
            dsep, esep, trjsep, _, _, evsep = smodel.slots(y[None].to(dev))
            dsep = dsep[0, ..., :SEG].cpu()
            esep = torch.sigmoid(esep[0]).cpu()
            env3 = float(torch.sigmoid(evsep[0]) > 0.5)
            sepst = dsep.numpy().astype(np.float64)
            ptr, ptn, _ = emodel(dsep.to(dev))
            _, pv2 = e2model(dsep.to(dev))     # A10 T-F velocity
            _, _, pal = emodel(y[None].to(dev))
        trj3 = ptr.argmax(-1).cpu()
        tn3 = pv2.cpu().clamp(0, 1)
        al3 = float(pal[0])
        # ---- S4 oracle stems + est heads
        with torch.no_grad():
            _, pv4 = e2model(stems.to(dev))    # A10
        tn4 = pv4.cpu().clamp(0, 1)

        # ---- posterior stems (A5.1); PIT and identities consume these
        po1 = posterior_stems(yn, [st1[k].numpy() for k in range(MAXK)])
        po2 = posterior_stems(yn, [st2[k].numpy() for k in range(MAXK)])
        if hasattr(smodel, "own"):     # A8 ownership masks
            po3 = posterior_stems_own(smodel, yn,
                                      [sepst[k] for k in range(MAXK)],
                                      esep, dev)
        else:
            po3 = posterior_stems(yn, [sepst[k] for k in range(MAXK)])
        pg = pit_by_sisdr(po1, gts, K)
        pp = pit_by_sisdr(po3, gts, K)

        # ---- S2P (A7): PWG inference -- theta identified by the
        # Whittle marginal likelihood on the posterior-isolated stem
        # (moment-initialised W, grid + parabolic refinement); other
        # structure amortized (env: sep head; alpha: est).
        # Truncated-prior MAP: the amortized graph head is the prior
        # (trust region +-40%), the Whittle likelihood picks within it;
        # flat likelihoods then stay near the amortized estimate.
        s2p_v = [0.0] * K
        for k in range(K):
            if int(traj[k]) == 0:
                vg = float(theta_denorm_t(
                    traj[k][None], tn1[pg[k]][None].clamp(0, 1)))
                grid = np.linspace(max(5.0, 0.6 * vg),
                                   min(40.0, 1.4 * vg), 15)
                s2p_v[k] = pwg.pwg_theta(po3[pp[k]], dev, grid=grid)

        # ---- inference metrics
        inf.setdefault("stem_sisdr_prior", {"S1": [], "S2": [], "S3": []})
        for k in range(K):
            inf["stem_sisdr"]["S1"].append(si_sdr(po1[pg[k]], gts[k]))
            inf["stem_sisdr"]["S2"].append(si_sdr(po2[pg[k]], gts[k]))
            inf["stem_sisdr"]["S3"].append(si_sdr(po3[pp[k]], gts[k]))
            inf["stem_sisdr_prior"]["S1"].append(
                si_sdr(st1[pg[k]].numpy(), gts[k]))
            inf["stem_sisdr_prior"]["S2"].append(
                si_sdr(st2[pg[k]].numpy(), gts[k]))
            inf["stem_sisdr_prior"]["S3"].append(
                si_sdr(sepst[pp[k]], gts[k]))
            if int(traj[k]) == 0:      # velocity: pass-by slots only
                gt_v = float(theta_denorm_t(traj[k][None], thn[k][None]))
                for tag, tn_, perm in (("S1", tn1, pg), ("S2", tn2, pg),
                                       ("S3", tn3, pp)):
                    vh = float(theta_denorm_t(
                        traj[k][None], tn_[perm[k]][None].clamp(0, 1)))
                    inf["theta_abserr"][tag].append(
                        {"traj": "passby", "err": abs(vh - gt_v),
                         "rel": abs(vh - gt_v) / gt_v})
                if s2p_v is not None:
                    inf["theta_abserr"].setdefault("S2P", []).append(
                        {"traj": "passby", "err": abs(s2p_v[k] - gt_v),
                         "rel": abs(s2p_v[k] - gt_v) / gt_v})
            inf["traj_acc"]["S1"].append(int(trj1[pg[k]]) == int(traj[k]))
            inf["traj_acc"]["S3"].append(int(trj3[pp[k]]) == int(traj[k]))
        if env > 0.5:
            for tag, al_ in (("S1", al1), ("S2", al2), ("S3", al3)):
                inf["alpha_abserr"][tag].append(abs(al_ - alpha))
        inf.setdefault("env_acc", {"S1": [], "S3": []})
        inf["env_acc"]["S1"].append(float(env1) == env)
        inf["env_acc"]["S3"].append(env3 == env)
        for tag, e_ in (("S1", e1), ("S3", esep)):
            pred = (e_ > 0.5).numpy()
            for k in range(MAXK):
                gt_on = k < K
                if pred[k] and gt_on:
                    inf["exist"][tag]["tp"] += 1
                elif pred[k] and not gt_on:
                    inf["exist"][tag]["fp"] += 1
                elif not pred[k] and gt_on:
                    inf["exist"][tag]["fn"] += 1

        # ---- E1 remove (K>=2 only, A5.3: zero reference is meaningless)
        spec = {"scene": int(i)}
        if K >= 2:
            k = int(rng.integers(K))
            spec["E1_slot"] = k
            ref = yn - gts[k]
            for tag, po_, perm in (("S1", po1, pg), ("S2", po2, pg),
                                   ("S3", po3, pp)):
                est = yn - po_[perm[k]]
                l1 = locality(est, ref, ref)
                R[tag]["E1"].append({"scene": int(i),
                                     "sisdr": si_sdr(est, ref),
                                     "mrstft": mrstft_np(est, ref),
                                     "loc": l1, "loc_clean": l1})
            l1 = locality(yn - gts[k], ref, ref)
            R["S4"]["E1"].append({"scene": int(i),
                                  "sisdr": si_sdr(yn - gts[k], ref),
                                  "mrstft": 0.0,
                                  "loc": l1, "loc_clean": l1})
            if s2p_v is not None:   # shared front-end: same as S3
                est = yn - po3[pp[k]]
                l1 = locality(est, ref, ref)
                R["S2P"]["E1"].append({"scene": int(i),
                                       "sisdr": si_sdr(est, ref),
                                       "mrstft": mrstft_np(est, ref),
                                       "loc": l1, "loc_clean": l1})

        # ---- E2 re-velocity (open scenes, pass-by slot)
        pb = [k for k in range(K) if int(traj[k]) == 0]
        if pb and env < 0.5:
            k = pb[int(rng.integers(len(pb)))]
            gt_v = float(theta_denorm_t(traj[k][None], thn[k][None]))
            tgt = [g for g in V_GRID if abs(g - gt_v) > 1e-6]
            newv = float(tgt[int(rng.integers(len(tgt)))])
            spec["E2_slot"], spec["E2_newv"] = k, newv
            ned = yn - gts[k]
            ref = ned + REF_RENDER(gtd[k], "passby", newv,
                                    alpha, env)
            for tag, d_, po_, al_, ev_, perm in (
                    ("S1", d1, po1, al1, env1, pg),
                    ("S2", d2, po2, al2, env2, pg)):
                node = rerender_node_t(gmodel, d_[perm[k]], "passby", newv,
                                       al_, ev_, dev)
                est = yn - po_[perm[k]] + node
                R[tag]["E2"].append({"scene": int(i),
                                     "sisdr": si_sdr(est, ref),
                                     "mrstft": mrstft_np(est, ref),
                                     "loc": locality(est, ref, ned),
                                     "loc_clean": si_sdr(
                                         yn - po_[perm[k]], ned)})
            v3 = float(theta_denorm_t(traj[k][None], tn3[pp[k]][None]))
            d3 = pipeline_dry(po3[pp[k]], "passby", v3, al3, env3)
            node = render_node(d3, "passby", newv, al3, env3)
            est = yn - po3[pp[k]] + node
            R["S3"]["E2"].append({"scene": int(i), "sisdr": si_sdr(est, ref),
                                  "mrstft": mrstft_np(est, ref),
                                  "loc": locality(est, ref, ned),
                                  "loc_clean": si_sdr(yn - po3[pp[k]],
                                                      ned)})
            if s2p_v is not None:   # theta from physics line-search
                dp = inverse_passby(po3[pp[k]], s2p_v[k])
                node = render_node(dp, "passby", newv, al3, env3)
                est = yn - po3[pp[k]] + node
                R["S2P"]["E2"].append({"scene": int(i),
                                       "sisdr": si_sdr(est, ref),
                                       "mrstft": mrstft_np(est, ref),
                                       "loc": locality(est, ref, ned),
                                       "loc_clean": si_sdr(
                                           yn - po3[pp[k]], ned)})
            v4 = float(theta_denorm_t(traj[k][None], tn4[k][None]))
            d4 = pipeline_dry(gts[k], "passby", v4, alpha, env)
            node = render_node(d4, "passby", newv, alpha, env)
            est = yn - gts[k] + node
            R["S4"]["E2"].append({"scene": int(i), "sisdr": si_sdr(est, ref),
                                  "mrstft": mrstft_np(est, ref),
                                  "loc": locality(est, ref, ned),
                                  "loc_clean": si_sdr(yn - gts[k], ned)})

        # ---- E3 relative re-distancing (static slot; gain component)
        st_slots = [k for k in range(K) if int(traj[k]) == 1]
        if st_slots:
            k = st_slots[int(rng.integers(len(st_slots)))]
            fm = float(R_FACTORS[int(rng.integers(len(R_FACTORS)))])
            spec["E3_slot"], spec["E3_factor"] = k, fm
            ned = yn - gts[k]
            ref = ned + gts[k] / fm
            for tag, po_, perm in (("S1", po1, pg), ("S2", po2, pg),
                                   ("S3", po3, pp)):
                sk = po_[perm[k]]
                est = yn - sk + sk / fm
                R[tag]["E3"].append({"scene": int(i),
                                     "sisdr": si_sdr(est, ref),
                                     "mrstft": mrstft_np(est, ref),
                                     "loc": locality(est, ref, ned),
                                     "loc_clean": si_sdr(yn - sk, ned)})
            R["S4"]["E3"].append({"scene": int(i), "sisdr": si_sdr(ref, ref),
                                  "mrstft": 0.0,
                                  "loc": locality(ref, ref, ned),
                                  "loc_clean": si_sdr(ned, ned)})
            if s2p_v is not None:   # shared front-end: same as S3
                sk = po3[pp[k]]
                est = yn - sk + sk / fm
                R["S2P"]["E3"].append({"scene": int(i),
                                       "sisdr": si_sdr(est, ref),
                                       "mrstft": mrstft_np(est, ref),
                                       "loc": locality(est, ref, ned),
                                       "loc_clean": si_sdr(yn - sk,
                                                           ned)})

        # ---- E4 room re-absorption (room scenes; secondary)
        if env > 0.5:
            tgt = [a for a in ALPHA_NODES if abs(a - alpha) > 1e-6]
            newa = float(tgt[int(rng.integers(len(tgt)))])
            spec["E4_newa"] = newa
            ref = sum(REF_RENDER(gtd[j],
                                  "passby" if int(traj[j]) == 0 else "static",
                                  float(theta_denorm_t(traj[j][None],
                                                       thn[j][None])),
                                  newa, 1.0) for j in range(K))
            for tag, d_, tn_, al_ in (("S1", d1, tn1, al1),
                                      ("S2", d2, tn2, al2)):
                est = np.zeros(SEG)
                for j in range(K):
                    tj = "passby" if int(trj1[pg[j]]) == 0 else "static"
                    th = float(theta_denorm_t(
                        torch.tensor([0 if tj == "passby" else 1]),
                        tn_[pg[j]][None].clamp(0, 1)))
                    est = est + rerender_node_t(gmodel, d_[pg[j]], tj, th,
                                                newa, 1.0, dev)
                R.setdefault(tag, {}).setdefault("E4", []).append(
                    {"scene": int(i), "sisdr": si_sdr(est, ref),
                     "mrstft": mrstft_np(est, ref),
                     "loc": locality(est, ref, yn)})
            na = min(ALPHA_NODES, key=lambda t: abs(t - newa))
            dmix = wiener_dereverb(yn, al3)
            est = phi_reverb(dmix[:SEG], na)[:SEG]
            R.setdefault("S3", {}).setdefault("E4", []).append(
                {"scene": int(i), "sisdr": si_sdr(est, ref),
                 "mrstft": mrstft_np(est, ref),
                 "loc": locality(est, ref, yn)})
            if s2p_v is not None:
                # per-slot Wiener + re-reverb (source-specific gains;
                # mixture-level op is the S3 row above)
                est = np.zeros(SEG)
                for j in range(K):
                    est = est + phi_reverb(
                        wiener_dereverb(po3[pp[j]], al3)[:SEG], na)[:SEG]
                R["S2P"]["E4"].append({"scene": int(i),
                                       "sisdr": si_sdr(est, ref),
                                       "mrstft": mrstft_np(est, ref),
                                       "loc": locality(est, ref, yn)})
            # A12.5 oracle-E4 rung -> rows.S4.E4 (S4 = oracle stems).
            # GT stems + ESTIMATED alpha, mixture-level re-render.
            # NOTE: GT stems sum to y exactly and the alpha-conditioned
            # Wiener dereverb is a fixed linear filter, so this
            # coincides numerically with the S3 mixture-level path --
            # recorded anyway (the rung documents that S3's E4 is
            # already at the perfect-front-end ceiling).
            gmix = np.sum(gts[:K], axis=0)
            est_o = phi_reverb(wiener_dereverb(gmix[:SEG], al3)[:SEG],
                               na)[:SEG]
            R["S4"]["E4"].append({"scene": int(i),
                                  "sisdr": si_sdr(est_o, ref),
                                  "mrstft": mrstft_np(est_o, ref),
                                  "loc": locality(est_o, ref, yn)})
            # ORE4A: GT stems + GT alpha (oracle-alpha): isolates the
            # absorption-estimate cost; residual vs the reference is the
            # semi-analytic Wiener-dereverb approximation gap (L2/L8).
            est_oa = phi_reverb(wiener_dereverb(gmix[:SEG], alpha)[:SEG],
                                na)[:SEG]
            R["ORE4A"]["E4"].append({"scene": int(i),
                                     "sisdr": si_sdr(est_oa, ref),
                                     "mrstft": mrstft_np(est_oa, ref),
                                     "loc": locality(est_oa, ref, yn)})

        specs.append(spec)
        if (si + 1) % 25 == 0:
            print(f"{si+1}/{len(idxs)}", flush=True)

    # ---- aggregate
    def agg(rows, key="sisdr"):
        v = [r[key] for r in rows if key in r]
        if not v:
            return None
        m = np.median(rng.choice(v, (5000, len(v))), axis=1)
        return {"median": float(np.median(v)),
                "ci95": [float(np.percentile(m, 2.5)),
                         float(np.percentile(m, 97.5))], "n": len(v)}

    out = {"meta": {"date": str(datetime.date.today()),
                    "script": "src/p3_edit_eval.py",
                    "split": args.split,
                    "n_scenes": len(idxs), "refine_T": args.refine_T,
                    "protocol": ("PROTOCOL_P3.md criteria C1-C5 with "
                                 "amendment A5: posterior stems (Wiener "
                                 "consistency projection) in all "
                                 "subtraction identities; refinement w/ "
                                 "env likelihood selection + theta grid "
                                 "line-search; E1 on K>=2 only; primary "
                                 "test scene-paired Wilcoxon"),
                    "locality": ("A12.2 proxy: loc = si_sdr((est - ref)"
                                 " + noedit_ref, noedit_ref); "
                                 "noedit_ref = y - gts[k] for "
                                 "E1/E2/E3; for E4 noedit_ref = y "
                                 "(pre-edit mixture as its own "
                                 "reference frame: all nodes "
                                 "re-rendered, no untouched bed). For "
                                 "E1 ref == noedit_ref, so loc == "
                                 "sisdr by construction (recorded). "
                                 "The proxy charges node-render error "
                                 "to locality (conservative) but is "
                                 "computable for EVERY system incl. "
                                 "the regeneration family."),
                    "loc_clean": ("clean non-target damage for "
                                  "identity-based edits: est = y - "
                                  "s_hat_k + node', so est - node' = "
                                  "y - s_hat_k EXACTLY and loc_clean "
                                  "= si_sdr(y - s_hat_k, y - gts_k); "
                                  "node-render error excluded. E1: "
                                  "node' = 0 -> loc_clean == loc == "
                                  "sisdr. E4: undefined (whole scene "
                                  "re-rendered, no bed survives; "
                                  "proxy only). Regeneration family "
                                  "(X5/X6): undefined, proxy only "
                                  "(no node decomposition exists)."),
                    "oracle_E4": ("A12.5 rung -> rows.S4.E4: GT stems "
                                  "+ estimated alpha, mixture-level "
                                  "dereverb + re-render (coincides "
                                  "with S3's mixture path since GT "
                                  "stems sum to y and the dereverb is "
                                  "linear -- documented); ORE4A = GT "
                                  "stems + GT alpha (oracle-alpha), "
                                  "isolating the absorption-estimate "
                                  "cost; its residual = semi-analytic "
                                  "inverse approximation gap (L2)")},
           "edits": {}, "rows": {sys_: R[sys_] for sys_ in R},
           "stats": {}, "inference": {}, "edit_specs": specs}
    for sys_ in R:
        out["edits"][sys_] = {e: {"sisdr": agg(R[sys_][e]),
                                  "mrstft": agg(R[sys_][e], "mrstft"),
                                  "loc": agg(R[sys_][e], "loc"),
                                  "loc_clean": agg(R[sys_][e],
                                                   "loc_clean")}
                              for e in R[sys_] if R[sys_][e]}
    def paired_p(ra, rb):
        """Scene-paired Wilcoxon signed-rank (A5.4, primary,
        cluster-aware: one paired observation per scene)."""
        da = {r["scene"]: r["sisdr"] for r in ra}
        db = {r["scene"]: r["sisdr"] for r in rb}
        d = [da[s] - db[s] for s in sorted(set(da) & set(db))]
        if len(d) < 8 or all(abs(x) < 1e-12 for x in d):
            return None
        return float(wilcoxon(d).pvalue)

    for e in ("E1", "E2", "E3", "E4"):
        a = [r["sisdr"] for r in R["S2"][e]]
        b = [r["sisdr"] for r in R["S3"][e]]
        if a and b:
            out["stats"][f"{e}_S2_vs_S3_wilcoxon_p"] = paired_p(
                R["S2"][e], R["S3"][e])
            out["stats"][f"{e}_S2_vs_S3_mw_p"] = float(
                mannwhitneyu(a, b, alternative="two-sided").pvalue)
        a1 = [r["sisdr"] for r in R["S1"][e]]
        if a and a1:
            out["stats"][f"{e}_S2_vs_S1_wilcoxon_p"] = paired_p(
                R["S2"][e], R["S1"][e])
            out["stats"][f"{e}_S2_vs_S1_mw_p"] = float(
                mannwhitneyu(a, a1, alternative="two-sided").pvalue)
        if R["S2P"][e] and b:
            out["stats"][f"{e}_S2P_vs_S3_wilcoxon_p"] = paired_p(
                R["S2P"][e], R["S3"][e])
    for m in ("stem_sisdr", "stem_sisdr_prior", "alpha_abserr"):
        out["inference"][m] = {t: agg([{"sisdr": v} for v in inf[m][t]])
                               for t in inf[m]}
    out["inference"]["theta_relerr"] = {
        t: agg([{"sisdr": r["rel"]} for r in inf["theta_abserr"][t]])
        for t in inf["theta_abserr"]}
    out["inference"]["traj_acc"] = {
        t: float(np.mean(inf["traj_acc"][t])) for t in inf["traj_acc"]}
    out["inference"]["env_acc"] = {
        t: float(np.mean(inf["env_acc"][t])) for t in inf.get("env_acc", {})}
    f1 = {}
    for t, c in inf["exist"].items():
        p = c["tp"] / max(1, c["tp"] + c["fp"])
        r = c["tp"] / max(1, c["tp"] + c["fn"])
        f1[t] = {"precision": p, "recall": r,
                 "f1": 2 * p * r / max(1e-9, p + r)}
    out["inference"]["exist"] = f1
    (PN / args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out["edits"], indent=1)[:1500])
    print(json.dumps(out["stats"], indent=1))


if __name__ == "__main__":
    main()
