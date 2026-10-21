"""P3 scene-graph model + training (S1 ours / S3 pipeline components).

OURS (kind=graph): mixture -> trunk -> 3 slot queries (cross-attention)
-> per-slot {existence, trajectory type, theta, attention-masked latent}
-> shared dry decoder (slot-FiLM) -> analytic compositor
   sum_k exist_k * Room_alpha( Traj_k(d_k; theta_k) ).
Supervision (all GT from the rendered corpus): PIT-matched stem mrSTFT,
dry mrSTFT, theta L1, trajectory CE, existence BCE, alpha L1, mixture
mrSTFT through the compositor.

PIPELINE components at the same budget:
kind=sep: same trunk/slots/decoder but predicts WET stems directly.
kind=est: per-stem estimator (traj, theta) + scene alpha from mix.
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, "/workspace/CF_PhysAudio/src")
sys.path.insert(0, "/workspace/audiocf_render")
from p2_train import Encoder, Decoder, ConvBlock, mrstft_loss  # noqa: E402
from phi_torch import phi_velocity_t, phi_distance_t, RIRBank  # noqa: E402

ROOT = Path("/workspace/CF_ASG")
SCENES = ROOT / "p3" / "scenes3"
SR = 16000
SEG = 4 * SR
MAXK = 3
ALPHA_NODES = tuple(round(0.30 + 0.05 * i, 2) for i in range(5))


def tpool(x):
    """Order-aware pooling: [mean, std, first temporal moment].
    The temporal moment E[x * tau], tau linear in [-1,1], is the
    projection onto the linear time mode -- the statistic a monotone
    trajectory (Doppler) lives in, which mean/std cannot express."""
    T = x.shape[-1]
    tau = torch.linspace(-1.0, 1.0, T, device=x.device)
    return torch.cat([x.mean(-1), x.std(-1), (x * tau).mean(-1)], -1)


def theta_norm(traj, theta):
    """passby v in [5,40] -> [0,1]; static r in [1,32] log -> [0,1]."""
    if traj == "passby":
        return (theta - 5.0) / 35.0
    return float(np.log(theta) / np.log(32.0))


def theta_denorm_t(traj_id, n):
    """traj_id: (B,) 0=passby 1=static; n: (B,) in [0,1] -> physical."""
    v = 5.0 + 35.0 * n
    r = torch.exp(n * float(np.log(32.0)))
    return torch.where(traj_id == 0, v, r)


class SceneSet(torch.utils.data.Dataset):
    def __init__(self, split):
        self.files = sorted((SCENES / split).glob("*.npz"))
        assert self.files, f"empty {split}"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        z = np.load(self.files[i], allow_pickle=True)
        K = int(z["K"])
        meta = json.loads(str(z["meta"]))
        stems = np.zeros((MAXK, SEG), np.float32)
        dries = np.zeros((MAXK, SEG), np.float32)
        stems[:K] = z["stems"][:, :SEG]
        dries[:K] = z["dries"][:, :SEG]
        traj = np.zeros(MAXK, np.int64)
        thn = np.zeros(MAXK, np.float32)
        for k, m in enumerate(meta):
            traj[k] = 0 if m["traj"] == "passby" else 1
            thn[k] = theta_norm(m["traj"], m["theta"])
        exist = np.zeros(MAXK, np.float32)
        exist[:K] = 1.0
        env = int(z["env"]) if "env" in z else 1
        return (torch.tensor(z["y"][:SEG]), torch.tensor(stems),
                torch.tensor(dries), torch.tensor(exist),
                torch.tensor(traj), torch.tensor(thn),
                torch.tensor(float(z["alpha"])),
                torch.tensor(float(env)), K)


class SlotHeads(nn.Module):
    """Feature-space competitive masking: sources in a scene are
    SIMULTANEOUS, so binding cannot happen along time alone; slots
    compete per (channel, frame) cell of the trunk representation
    (TasNet-style masks over a learned filterbank), and every per-slot
    quantity is read from the masked, source-specific features."""

    def __init__(self, cdim=1024, sdim=192):
        super().__init__()
        self.vproj = nn.Conv1d(cdim, sdim, 1)
        self.mask = nn.Sequential(
            nn.Conv1d(cdim, 2 * sdim, 5, 1, 2), nn.GELU(),
            nn.Conv1d(2 * sdim, MAXK * sdim, 1))
        # trajectory-sensitive per-slot readout: velocity lives in the
        # TEMPORAL structure of a source's features; mean-pooling alone
        # is order-invariant, so the readout keeps a temporal conv and
        # concatenates first and second moments.
        self.readout = nn.Sequential(
            nn.Conv1d(sdim, sdim, 5, 1, 2), nn.GELU(),
            nn.Conv1d(sdim, sdim, 5, 1, 2), nn.GELU())
        self.norm = nn.LayerNorm(3 * sdim)
        self.exist = nn.Linear(3 * sdim, 1)
        self.traj = nn.Linear(3 * sdim, 2)
        self.theta = nn.Linear(3 * sdim, 1)   # UNBOUNDED; clamp at use
        self.alpha = nn.Linear(3 * sdim, 1)
        self.env = nn.Linear(3 * sdim, 1)

    def forward(self, h):
        # h: (B, C, T)
        B, _, T = h.shape
        vv = self.vproj(h)                       # (B,S,T)
        S = vv.shape[1]
        M = self.mask(h).reshape(B, MAXK, S, T).softmax(1)   # over slots
        zk = vv.unsqueeze(1) * M                 # (B,K,S,T) source features
        r = self.readout(zk.reshape(B * MAXK, S, T))
        feat = self.norm(tpool(r).reshape(B, MAXK, -1))   # (B,K,3S)
        s = zk.mean(-1)                          # decoder FiLM vector
        pooled = self.norm(tpool(vv))
        alpha = 0.30 + 0.20 * torch.sigmoid(self.alpha(pooled))[:, 0]
        env = self.env(pooled)[:, 0]
        return (zk, s, self.exist(feat)[..., 0], self.traj(feat),
                self.theta(feat)[..., 0], alpha, env)


class OwnershipHead(nn.Module):
    """A8: supervised per-cell slot-ownership refiner.  Input: log
    power spectra of the K predicted stems + the mixture (K+1,F,T);
    output: per-slot ownership logits (K,F,T)."""
    def __init__(self, K=3, ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(K + 1, ch, 3, padding=1), nn.GroupNorm(4, ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=2, dilation=2),
            nn.GroupNorm(4, ch), nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.GroupNorm(4, ch),
            nn.SiLU(),
            nn.Conv2d(ch, K, 1))

    def forward(self, feats):
        return self.net(feats)


class SceneModel(nn.Module):
    def __init__(self, kind="graph", zdim=192):
        super().__init__()
        self.kind = kind
        self.enc = Encoder(zdim=zdim)          # outputs mu,lv (2*zdim heads)
        self.heads = SlotHeads(cdim=2 * zdim, sdim=zdim)
        self.dec = Decoder(zdim=zdim, film_dim=zdim)
        if kind == "graph":
            self.rir = RIRBank(nodes=ALPHA_NODES)
        if kind == "sep2":
            self.own = OwnershipHead(K=MAXK)

    def trunk(self, y):
        mu, lv = self.enc(y)
        return torch.cat([mu, lv], 1)          # (B, 2*zdim, T)

    def slots(self, y):
        h = self.trunk(y)
        zk, s, e, traj, thn, alpha, env = self.heads(h)
        outs = []
        for k in range(MAXK):
            outs.append(self.dec(zk[:, k], s[:, k, :]))
        d = torch.stack(outs, 1)               # (B,K,N) dry (graph) / wet (sep)
        return d, e, traj, thn, alpha, env

    def compose(self, d, e, traj, thn, alpha, env):
        """Render per-slot stems through Phi and sum. d: (B,K,N).
        env: (B,) in {0,1}; the room is applied only to room scenes."""
        B, K, N = d.shape
        traj_id = traj.argmax(-1)              # (B,K)
        stems = []
        for k in range(K):
            dk = d[:, k, :]
            th = theta_denorm_t(traj_id[:, k], thn[:, k].clamp(0, 1))
            wp = phi_velocity_t(dk, th.clamp(5, 40))
            ws = phi_distance_t(dk, th.clamp(1, 32))
            w = torch.where((traj_id[:, k] == 0)[:, None],
                            wp[:, :N], ws[:, :N])
            wr = self.rir(w, alpha)[:, :N]
            w = torch.where((env > 0.5)[:, None], wr, w)
            stems.append(w * torch.sigmoid(e[:, k])[:, None])
        stems = torch.stack(stems, 1)
        return stems, stems.sum(1)


class Est2(nn.Module):
    """A10: T-F velocity regressor.  The Doppler signature is a
    spectral-line glide -- a 2-D conv stack on the log-spectrogram has
    the right inductive bias, unlike waveform convs.  Trained with
    bleed augmentation (stem + lambda*(y-stem)) and random T-F erasure
    (mimicking ownership-mask holes), closing the clean-train /
    posterior-deploy domain gap that capped the est head."""
    def __init__(self, ch=48):
        super().__init__()

        def blk(ci, co):
            return nn.Sequential(nn.Conv2d(ci, co, 3, stride=2, padding=1),
                                 nn.GroupNorm(4, co), nn.SiLU())
        self.net = nn.Sequential(blk(1, ch), blk(ch, 2 * ch),
                                 blk(2 * ch, 4 * ch), blk(4 * ch, 4 * ch),
                                 blk(4 * ch, 4 * ch))
        self.head = nn.Linear(4 * ch, 3)   # traj logits (2) + v_norm

    def forward(self, s):
        w = torch.hann_window(1024, device=s.device)
        Z = torch.stft(s, 1024, 256, window=w, return_complex=True)
        X = Z.abs().pow(2).clamp_min(1e-10).log()[:, None]
        X = X - X.mean(dim=(2, 3), keepdim=True)
        h = self.net(X).mean(dim=(2, 3))
        o = self.head(h)
        return o[:, :2], o[:, 2]


class Dryer(nn.Module):
    """Amortized physics inverse (A6.1): wet stem -> dry source.
    Single-stem task -- no separation, no PIT, no slot assignment.
    Bleed augmentation in the training loop simulates imperfect
    separation: input = stem + lambda*(y - stem), lambda~U[0,0.5].
    Shape/scale split (A6.2/A6.3): dries are pre-gain sources with
    |x| up to ~100.  A joint waveform-times-gain loss collapses to
    silence (mrstft's silence attractor), so shape and scale are
    fully decoupled: the shape branch is trained in a space where
    BOTH sides are RMS-normalised (silence unreachable), the gain
    head regresses log-RMS separately."""
    def __init__(self, zdim=192):
        super().__init__()
        self.enc = Encoder(zdim=zdim)
        self.dec = Decoder(zdim=zdim)
        self.gain = nn.Linear(zdim, 1)

    def parts(self, s):
        mu, lv = self.enc(s)
        shape = self.dec(mu)
        shape = shape / shape.pow(2).mean(-1, keepdim=True) \
            .sqrt().clamp(min=1e-4)                # unit RMS
        g = self.gain(mu.mean(-1)).squeeze(-1)     # log-RMS
        return shape, g

    def forward(self, s):
        shape, g = self.parts(s)
        return shape * torch.exp(g.clamp(-6, 5))[:, None]


def pit_match(pred, gt, exist):
    """Cheap PIT on per-stem L1-STFT; returns per-batch permutation."""
    B, K, N = pred.shape
    with torch.no_grad():
        w = torch.hann_window(512, device=pred.device)
        P = torch.stft(pred.reshape(B * K, N), 512, 256, window=w,
                       return_complex=True).abs().reshape(B, K, -1)
        G = torch.stft(gt.reshape(B * K, N), 512, 256, window=w,
                       return_complex=True).abs().reshape(B, K, -1)
        cost = torch.cdist(P, G, p=1)          # (B,K,K)
        perms = list(itertools.permutations(range(K)))
        pc = torch.stack([cost[:, list(p), range(K)].sum(1) for p in perms], 1)
        best = pc.argmin(1)
    idx = torch.tensor(perms, device=pred.device)[best]   # (B,K)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["graph", "sep", "sep2", "dry",
                                       "dryer", "est", "est2"],
                    required=True)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=0,
                    help="manual seed (A12.3: scratch verifier retrain "
                         "uses a fresh seed; default 0 = historical "
                         "behavior)")
    args = ap.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda"
    name = args.kind + args.tag
    ck = ROOT / "p3" / "ckpt"
    ck.mkdir(parents=True, exist_ok=True)

    if args.kind == "est":
        model = Estimator().to(dev)
    elif args.kind == "est2":
        model = Est2().to(dev)
    elif args.kind == "dryer":
        model = Dryer().to(dev)
    else:
        model = SceneModel(args.kind).to(dev)
    nparam = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"kind={args.kind} params={nparam/1e6:.1f}M", flush=True)

    if args.kind == "sep2" and not (ck / "sep2_latest.pt").exists() \
            and (ck / "sep_latest.pt").exists():
        st0 = torch.load(ck / "sep_latest.pt", map_location=dev)
        r = model.load_state_dict(st0["model"], strict=False)
        print(f"warm-start from sep: {len(r.missing_keys)} new keys",
              flush=True)

    ds = SceneSet("train")
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True,
                                     num_workers=6, drop_last=True,
                                     persistent_workers=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    step, t0, log = 0, time.time(), []
    ckp = ck / f"{name}_latest.pt"
    if args.resume and ckp.exists():
        st = torch.load(ckp, map_location=dev)
        model.load_state_dict(st["model"])
        step = st["step"]
        for _ in range(step):
            sched.step()
        lp = ck / f"{name}_log.json"
        if lp.exists():
            log = [e for e in json.loads(lp.read_text()) if e["step"] <= step]
        print(f"resumed {name} from {step}", flush=True)
    while step < args.steps:
        for y, stems, dries, exist, traj, thn, alpha, env, K in dl:
            y, stems, dries = y.to(dev), stems.to(dev), dries.to(dev)
            exist, traj, thn = exist.to(dev), traj.to(dev), thn.to(dev)
            alpha, env = alpha.to(dev), env.to(dev)
            if args.kind == "est":
                loss, parts = model.loss(stems, dries, exist, traj, thn,
                                         alpha, env, y)
            elif args.kind == "est2":
                sel = exist.flatten() > 0
                s = stems.flatten(0, 1)[sel]
                tr = traj.flatten()[sel]
                tn_t = thn.flatten()[sel]
                bleed = (y[:, None, :] - stems).flatten(0, 1)[sel]
                lam = torch.rand(s.shape[0], 1, device=dev) * 0.6
                inp = s + lam * bleed
                # random T-F erasure: zero a few time/freq bands in the
                # waveform domain via stft round-trip is costly; approx
                # with random short time-segment dropout
                if True:
                    B2, N2 = inp.shape
                    for _ in range(2):
                        wlen = int(0.1 * N2)
                        st = torch.randint(0, N2 - wlen, (B2,), device=dev)
                        ar = torch.arange(N2, device=dev)[None, :]
                        mask = ((ar >= st[:, None])
                                & (ar < st[:, None] + wlen))
                        inp = inp * (~mask | (torch.rand(B2, 1, device=dev)
                                              > 0.5)).float()
                ptr, pvn = model(inp)
                l_tr = F.cross_entropy(ptr, tr)
                mv = (tr == 0).float()
                l_v = (torch.abs(pvn - tn_t) * mv).sum() \
                    / mv.sum().clamp(min=1)
                loss = 0.3 * l_tr + l_v
                parts = {"tr": float(l_tr), "v": float(l_v)}
            elif args.kind == "dryer":
                sel = exist.flatten() > 0
                s = stems.flatten(0, 1)[sel]
                dtg = dries.flatten(0, 1)[sel]
                bleed = (y[:, None, :] - stems).flatten(0, 1)[sel]
                lam = torch.rand(s.shape[0], 1, device=dev) * 0.5
                shape, g = model.parts(s + lam * bleed)
                L = min(shape.shape[-1], dtg.shape[-1])
                # A6.3: both sides unit-RMS -- silence unreachable
                r = dtg[..., :L].pow(2).mean(-1, keepdim=True) \
                    .sqrt().clamp(min=1e-3)
                sh = shape[..., :L]
                sh = sh / sh.pow(2).mean(-1, keepdim=True) \
                    .sqrt().clamp(min=1e-4)
                l_shape = mrstft_loss(sh, dtg[..., :L] / r)
                l_gain = F.mse_loss(g, r.log().squeeze(-1))
                loss = l_shape + 0.1 * l_gain
                parts = {"dry": float(l_shape), "gain": float(l_gain)}
            else:
                d, e, trj, tn, al, ev = model.slots(y)
                d = d[..., :SEG]
                if args.kind == "graph":
                    pstems, ymix = model.compose(d, e, trj, tn, al, env)
                else:
                    pstems, ymix = d, d.sum(1)
                pstems, ymix = pstems[..., :SEG], ymix[..., :SEG]
                # kind "dry": slots predict DRY sources (amortized
                # physics inverse); PIT and stem loss on dries, and no
                # mixture loss (dries do not sum to y).
                tgt = dries if args.kind == "dry" else stems
                idx = pit_match(pstems, tgt, exist)
                bi = torch.arange(y.shape[0], device=dev)[:, None]
                ps, pd = pstems[bi, idx], d[bi, idx]
                pe, ptrj = e[bi, idx], trj[bi, idx]
                ptn = tn[bi, idx]
                m = exist  # (B,K) 1 for real stems
                L = min(ps.shape[-1], tgt.shape[-1])
                sel = m.flatten() > 0            # real stems only (A2b)
                l_stem = mrstft_loss(
                    ps[..., :L].flatten(0, 1)[sel],
                    tgt[..., :L].flatten(0, 1)[sel])
                l_mix = mrstft_loss(ymix[:, :L], y[:, :L]) \
                    if args.kind != "dry" else l_stem.detach() * 0
                l_ex = F.binary_cross_entropy_with_logits(pe, m)
                l_tr = (F.cross_entropy(ptrj.flatten(0, 1), traj.flatten(),
                                        reduction="none")
                        * m.flatten()).mean()
                mv = m * (traj == 0).float()        # theta: pass-by only
                l_th = (torch.abs(ptn - thn) * mv).sum() \
                    / mv.sum().clamp(min=1)
                mr = (env > 0.5).float()            # alpha: room scenes only
                l_al = (torch.abs(al - alpha) * mr).sum() \
                    / mr.sum().clamp(min=1)
                l_env = F.binary_cross_entropy_with_logits(ev, env)
                loss = l_stem + l_mix + 0.2 * l_ex + 0.2 * l_tr \
                    + 0.5 * l_th + 0.5 * l_al + 0.2 * l_env
                parts = {"stem": float(l_stem), "mix": float(l_mix),
                         "th": float(l_th)}
                if args.kind == "graph":
                    Ld = min(pd.shape[-1], dries.shape[-1])
                    l_dry = mrstft_loss(
                        pd[..., :Ld].flatten(0, 1)[sel],
                        dries[..., :Ld].flatten(0, 1)[sel])
                    loss = loss + 0.5 * l_dry
                    parts["dry"] = float(l_dry)
                if args.kind == "sep2":       # A8 ownership supervision
                    B = y.shape[0]
                    win = torch.hann_window(1024, device=dev)

                    def pow_spec(x):
                        Z = torch.stft(x, 1024, 256, window=win,
                                       return_complex=True)
                        return Z.abs().pow(2)

                    Pp = pow_spec(ps[..., :L].reshape(B * MAXK, L)) \
                        .reshape(B, MAXK, 513, -1)
                    with torch.no_grad():
                        Pg = pow_spec(stems[..., :L].reshape(B * MAXK, L)) \
                            .reshape(B, MAXK, 513, -1)
                        Pm = pow_spec(y[:, :L])[:, None]
                        share = Pg * m[:, :, None, None]
                        share = share / share.sum(1, keepdim=True) \
                            .clamp_min(1e-12)
                        wcell = Pm.pow(0.3)
                        wcell = wcell / wcell.mean().clamp_min(1e-12)
                    feats = torch.cat([Pp, Pm], 1).clamp_min(1e-10).log()
                    logits = model.own(feats)
                    logits = logits.masked_fill(
                        (m < 0.5)[:, :, None, None], -1e4)
                    logp = torch.log_softmax(logits, 1)
                    l_own = (-(share * logp).sum(1, keepdim=True)
                             * wcell).mean()
                    loss = loss + 0.7 * l_own
                    parts["own"] = float(l_own)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            step += 1
            if step % 200 == 0:
                log.append({"step": step, "loss": float(loss), **parts})
                print(f"{step} loss={float(loss):.4f} "
                      f"{json.dumps(parts)} "
                      f"({(time.time()-t0)/200:.2f}s/it)", flush=True)
                t0 = time.time()
            if step % 1000 == 0:
                (ck / f"{name}_log.json").write_text(json.dumps(log))
            if step % 2000 == 0 or step >= args.steps:
                torch.save({"model": model.state_dict(), "step": step,
                            "args": vars(args)}, ckp)
            if step >= args.steps:
                break
    print("TRAINING DONE", name)


class Estimator(nn.Module):
    """Pipeline stage 2: per-stem (traj, theta) + scene alpha from mix."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(ConvBlock(1, 64, 4), ConvBlock(64, 128, 4),
                                 ConvBlock(128, 256, 4), ConvBlock(256, 256, 4))
        self.head = nn.Linear(3 * 256, 4)  # traj(2), theta(1), alpha(1)

    def forward(self, x):
        h = tpool(self.net(x[:, None, :]))
        o = self.head(h)
        return (o[:, :2], o[:, 2],
                0.30 + 0.20 * torch.sigmoid(o[:, 3]))

    def loss(self, stems, dries, exist, traj, thn, alpha, env, y):
        B, K, N = stems.shape
        m = exist.flatten() > 0
        s = stems.flatten(0, 1)[m]
        tr, tn = traj.flatten()[m], thn.flatten()[m]
        ptr, ptn, _ = self.forward(s)
        _, _, pal = self.forward(y)
        l_tr = F.cross_entropy(ptr, tr)
        mv = (tr == 0).float()
        l_th = (torch.abs(ptn - tn) * mv).sum() / mv.sum().clamp(min=1)
        mr = (env > 0.5).float()
        l_al = (torch.abs(pal - alpha) * mr).sum() / mr.sum().clamp(min=1)
        loss = l_tr + l_th + l_al
        return loss, {"tr": float(l_tr), "th": float(l_th),
                      "al": float(l_al)}


if __name__ == "__main__":
    main()
