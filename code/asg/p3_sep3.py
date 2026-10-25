"""sep3: a mask-based time-frequency separation front end for the ASG
pipeline (revision R1, 2026-10-22).

Motivation. The deployed separator (sep2) synthesises each slot's
waveform from a latent through a convolutional decoder and is trained
with a spectral-magnitude loss only. Its stems reach a median SI-SDR of
5.1 dB on the main cohort, below AudioSep's 6.65 dB, and the slots are
not nameable by CLAP in two- and three-source scenes (41% and 35%
agreement with the oracle correspondence). sep3 keeps every later stage
of the pipeline (ownership-free, mixture-consistent stems; existence
and environment from the sep2 heads; velocity and absorption from the
est heads; the same inverse and edit rules) and replaces only the stem
estimator:

  STFT(y) -> dual-path (frequency / time) recurrent blocks -> MAXK
  complex masks -> masked mixture -> iSTFT.

Training: PIT over the real stems with negative SI-SDR plus the
multi-resolution STFT loss used for sep2, a quiet-slot penalty on the
unused slots and a mixture-consistency term. Same 24k rendered training
scenes, same split discipline (val never touched for training).

  python src/p3_sep3.py train --steps 40000 --batch 8
  python src/p3_sep3.py eval  --split val --n 200     # stem SI-SDR

Checkpoint p3/ckpt/sep3_latest.pt, log p3/ckpt/sep3_log.json.
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

sys.path.insert(0, "/workspace/CF_ASG/src")
sys.path.insert(0, "/workspace/CF_PhysAudio/src")
from p3_train import SceneSet, SEG, MAXK, ROOT, SR  # noqa: E402
from p2_train import mrstft_loss  # noqa: E402

N_FFT, HOP = 512, 128
N_BIN = N_FFT // 2 + 1


def si_sdr_t(est, ref, eps=1e-8):
    """(B, N) -> (B,) SI-SDR in dB."""
    ref = ref - ref.mean(-1, keepdim=True)
    est = est - est.mean(-1, keepdim=True)
    a = (est * ref).sum(-1, keepdim=True) / (ref.pow(2).sum(-1, keepdim=True) + eps)
    t = a * ref
    return 10 * torch.log10(t.pow(2).sum(-1) / ((est - t).pow(2).sum(-1) + eps) + eps)


class DualPathBlock(nn.Module):
    """One intra-frequency and one inter-time bidirectional GRU, each with
    a residual connection and layer norm (dual-path RNN on a T-F grid)."""

    def __init__(self, dim, hidden):
        super().__init__()
        self.f_rnn = nn.GRU(dim, hidden, batch_first=True, bidirectional=True)
        self.f_proj = nn.Linear(2 * hidden, dim)
        self.f_norm = nn.LayerNorm(dim)
        self.t_rnn = nn.GRU(dim, hidden, batch_first=True, bidirectional=True)
        self.t_proj = nn.Linear(2 * hidden, dim)
        self.t_norm = nn.LayerNorm(dim)

    def forward(self, x):                      # x: (B, F, T, D)
        B, Fn, T, D = x.shape
        h = x.permute(0, 2, 1, 3).reshape(B * T, Fn, D)      # along frequency
        h = self.f_proj(self.f_rnn(h)[0]).reshape(B, T, Fn, D).permute(0, 2, 1, 3)
        x = self.f_norm(x + h)
        h = x.reshape(B * Fn, T, D)                           # along time
        h = self.t_proj(self.t_rnn(h)[0]).reshape(B, Fn, T, D)
        return self.t_norm(x + h)


class Sep3(nn.Module):
    def __init__(self, dim=96, hidden=96, blocks=5, use_ckpt=True, amp=True, hop=HOP, mask_bound=0.0,
                 k_cond=False):
        super().__init__()
        self.use_ckpt, self.amp, self.hop, self.mask_bound = use_ckpt, amp, hop, mask_bound
        # source-count conditioning: the edit specification states K, so the
        # separator is told K (learned embedding added to the input features)
        # and only its first K slots are live
        self.k_cond = k_cond
        if k_cond:
            self.kemb = nn.Embedding(MAXK, dim)
        self.inp = nn.Sequential(nn.Conv2d(3, dim, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(dim, dim, 3, padding=1))
        self.blocks = nn.ModuleList([DualPathBlock(dim, hidden) for _ in range(blocks)])
        self.out = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.GELU(),
                                 nn.Conv2d(dim, 2 * MAXK, 1))
        self.register_buffer("win", torch.hann_window(N_FFT))

    def stft(self, x):
        return torch.stft(x, N_FFT, self.hop, window=self.win, return_complex=True)

    def forward(self, y, K=None):              # y: (B, N), K: (B,) ints -> stems (B, MAXK, N)
        N = y.shape[-1]
        Z = self.stft(y)                       # (B, F, T)
        scale = Z.abs().pow(2).mean((1, 2), keepdim=True).sqrt().clamp_min(1e-5)
        Zn = Z / scale
        feat = torch.stack([Zn.real, Zn.imag, Zn.abs().clamp_min(1e-5).log()], 1)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
            h = self.inp(feat).permute(0, 2, 3, 1)  # (B, F, T, D)
            if self.k_cond:
                assert K is not None, "k_cond model needs K"
                h = h + self.kemb(K.long().clamp(1, MAXK) - 1)[:, None, None, :]
            for b in self.blocks:
                if self.use_ckpt and self.training:
                    h = torch.utils.checkpoint.checkpoint(b, h, use_reentrant=False)
                else:
                    h = b(h)
            m = self.out(h.permute(0, 3, 1, 2))     # (B, 2K, F, T)
        m = m.float()
        if self.mask_bound > 0:
            m = self.mask_bound * torch.tanh(m / self.mask_bound)
        M = torch.complex(m[:, :MAXK], m[:, MAXK:])
        if self.k_cond:
            live = (torch.arange(MAXK, device=y.device)[None] < K.long()[:, None])
            M = M * live[:, :, None, None]
        S = M * Z[:, None]
        B, K, Fn, T = S.shape
        out = torch.istft(S.reshape(B * K, Fn, T), N_FFT, self.hop, window=self.win, length=N)
        return out.reshape(B, K, N)


def pit_loss(pred, gt, exist):
    """pred, gt: (B, K, N); exist: (B, K). Returns loss and the matched
    permutation indices (B, K)."""
    B, K, N = pred.shape
    perms = list(itertools.permutations(range(K)))
    costs = []
    for p in perms:
        idx = torch.tensor(p, device=pred.device)
        s = si_sdr_t(pred[:, idx].reshape(B * K, N), gt.reshape(B * K, N)).reshape(B, K)
        s = s.clamp(max=30.0)
        costs.append(-(s * exist).sum(1) / exist.sum(1).clamp(min=1))
    costs = torch.stack(costs, 1)              # (B, P)
    best = costs.argmin(1)
    idx = torch.tensor(perms, device=pred.device)[best]   # (B, K)
    return costs.gather(1, best[:, None]).mean(), idx


def train(args):
    dev = "cuda"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = Sep3(args.dim, args.hidden, args.blocks, use_ckpt=not args.no_ckpt, amp=not args.no_amp, hop=args.hop,
                 mask_bound=args.mask_bound, k_cond=args.k_cond).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"sep3 params={nparam/1e6:.2f}M", flush=True)
    ds = SceneSet("train")
    if args.multi_only:
        cache = ROOT / "p3" / "scenes3" / "train_multi_idx.json"
        keep = json.loads(cache.read_text())
        print(f"multi_only: {len(keep)} of {len(ds)} training scenes", flush=True)
        ds = torch.utils.data.Subset(ds, keep)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True,
                                     num_workers=6, drop_last=True,
                                     persistent_workers=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05,
        anneal_strategy="cos", div_factor=20, final_div_factor=50)
    ck = ROOT / "p3" / "ckpt"
    ckp = ck / f"sep3{args.tag}_latest.pt"
    lp = ck / f"sep3{args.tag}_log.json"
    step, log = 0, []
    if args.resume and ckp.exists():
        st = torch.load(ckp, map_location=dev)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        step = st["step"]
        for _ in range(step):
            sched.step()
        if lp.exists():
            log = [e for e in json.loads(lp.read_text()) if e["step"] <= step]
        print("resumed at", step, flush=True)
    elif args.init:
        st = torch.load(ck / args.init, map_location=dev)
        model.load_state_dict(st["model"])
        print(f"init weights from {args.init} (step {st.get('step')})", flush=True)
    vds = SceneSet("val")
    vidx = multi_idx(vds, "val", args.n_val)
    best_v = -1e9
    t0 = time.time()
    model.train()
    while step < args.steps:
        for y, stems, dries, exist, traj, thn, alpha, env, K in dl:
            y, stems, exist = y.to(dev), stems.to(dev), exist.to(dev)
            if args.dm > 0 and float(torch.rand(())) < args.dm:
                # dynamic mixing: slots 1.. are drawn from other scenes of
                # the batch (stems are rendered independently and the
                # mixture is their sum), each stem gets a gain in
                # [-dm_db, dm_db]; the number of sources per scene is kept
                # at two or more because every scene in the multi-source
                # subset fills slots 0 and 1
                B_ = stems.shape[0]
                for k in range(1, MAXK):
                    sh = int(torch.randint(1, B_, (1,)))
                    stems[:, k] = torch.roll(stems[:, k], sh, 0)
                    exist[:, k] = torch.roll(exist[:, k], sh, 0)
                g = 10 ** ((2 * torch.rand(B_, MAXK, 1, device=dev) - 1) * args.dm_db / 20)
                stems = stems * g * exist[:, :, None]
                y = stems.sum(1)
                K = exist.sum(1).long().cpu()
            if args.aug_db > 0 or args.aug_tilt > 0:
                # level and spectral-tilt augmentation: each source gets an
                # independent gain in [-aug_db, 0] dB and an air-absorption
                # like tilt 10^(-A (f/4 kHz)^gamma / 20) with A in
                # [0, aug_tilt] dB (attenuation at 4 kHz) and gamma in
                # [1, 1.7]; the mixture is rebuilt as the sum of the
                # augmented stems (the Phi_B cohort sits about 16 dB darker
                # above 4 kHz than the training scenes)
                B_, _, T_ = stems.shape
                g = 10 ** (-args.aug_db * torch.rand(B_, MAXK, 1, device=dev) / 20)
                X = torch.fft.rfft(stems.float() * g, dim=-1)
                fk = torch.fft.rfftfreq(T_, 1 / SR).to(dev) / 4000.0
                beta = args.aug_tilt * torch.rand(B_, MAXK, 1, device=dev)
                gam = 1 + 0.7 * torch.rand(B_, MAXK, 1, device=dev)
                H = 10 ** (-beta * fk[None, None, :] ** gam / 20)
                stems = torch.fft.irfft(X * H, n=T_, dim=-1) * exist[:, :, None]
                y = stems.sum(1)
            pred = model(y, K.to(dev)) if args.k_cond else model(y)
            l_pit, idx = pit_loss(pred, stems, exist)
            bi = torch.arange(y.shape[0], device=dev)[:, None]
            ps = pred[bi, idx]                             # matched
            sel = exist.flatten() > 0
            L = min(ps.shape[-1], stems.shape[-1])
            if args.w_spec > 0:
                l_spec = mrstft_loss(ps[..., :L].flatten(0, 1)[sel],
                                     stems[..., :L].flatten(0, 1)[sel])
            else:
                l_spec = l_pit.detach() * 0
            # unused slots must stay quiet (relative to the mixture)
            quiet = (1 - exist).flatten() > 0
            if quiet.any():
                e_q = ps.flatten(0, 1)[quiet].pow(2).mean(-1)
                e_y = y.pow(2).mean(-1)[:, None].expand(-1, MAXK).flatten()[quiet]
                l_quiet = (e_q / (e_y + 1e-8)).clamp(max=10).mean()
            else:
                l_quiet = l_pit.detach() * 0
            l_mix = -si_sdr_t(pred.sum(1), y).clamp(max=40).mean() / 10
            # scale-aware mixture consistency: the stems must add up to y
            r = (pred.sum(1) - y).pow(2).sum(-1) / (y.pow(2).sum(-1) + 1e-8)
            l_sum = (10 * torch.log10(r + 1e-6)).clamp(min=-20).mean() / 10   # below -20 dB the projection takes over
            loss = (l_pit / 10 + args.w_spec * l_spec + args.w_quiet * l_quiet + args.w_mix * l_mix
                    + args.w_sum * l_sum)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            step += 1
            if step % 200 == 0:
                with torch.no_grad():
                    s_all = si_sdr_t(ps.flatten(0, 1), stems.flatten(0, 1)).reshape(y.shape[0], MAXK)
                    multi = ((exist.sum(1) >= 2)[:, None] * exist) > 0
                    multi_sisdr = float(s_all[multi].mean()) if multi.any() else float("nan")
                parts = {"pit_sisdr": float(-l_pit), "multi_sisdr": multi_sisdr, "spec": float(l_spec),
                         "quiet": float(l_quiet), "mix": float(-l_mix * 10), "sum_db": float(l_sum * 10)}
                log.append({"step": step, "loss": float(loss), **parts})
                print(f"{step} loss={float(loss):.3f} {json.dumps(parts)} "
                      f"({(time.time()-t0)/200:.2f}s/it)", flush=True)
                t0 = time.time()
            if step % args.val_every == 0 or step >= args.steps:
                v, per = evaluate(model, vds, vidx, dev)
                log.append({"step": step, "val_stem_sisdr_median": v, "val_per_k": per})
                print(f"  val stem SI-SDR median {v:.2f} dB on {len(vidx)} multi-source scenes, per K {per}", flush=True)
                lp.write_text(json.dumps(log))
                state = {"model": model.state_dict(), "opt": opt.state_dict(),
                         "step": step, "args": vars(args), "val": v}
                torch.save(state, ckp)
                if v > best_v:
                    best_v = v
                    torch.save(state, ck / f"sep3{args.tag}_best.pt")
                    print(f"  new best {v:.2f} dB at step {step}", flush=True)
                model.train()
            if step >= args.steps:
                break
    print("TRAINING DONE sep3", flush=True)


@torch.no_grad()
def evaluate(model, ds, idxs, dev):
    """Median stem SI-SDR over the scenes with two or more sources (the
    runner bypasses the separator on single-source scenes), plus the
    per-K medians."""
    model.eval()
    vals = {}
    for i in idxs:
        y, stems, dries, exist, traj, thn, alpha, env, K = ds[int(i)]
        K = int(K)
        if getattr(model, "k_cond", False):
            pred = model(y[None].to(dev), torch.tensor([K], device=dev))[0].cpu()
        else:
            pred = model(y[None].to(dev))[0].cpu()
        best = max(itertools.permutations(range(MAXK)),
                   key=lambda p: sum(float(si_sdr_t(pred[p[k]][None], stems[k][None])) for k in range(K)))
        vals.setdefault(K, []).extend(float(si_sdr_t(pred[best[k]][None], stems[k][None])) for k in range(K))
    multi = [v for k, vs in vals.items() if k >= 2 for v in vs]
    per = {int(k): float(np.median(v)) for k, v in sorted(vals.items())}
    return (float(np.median(multi)) if multi else float("nan")), per


def multi_idx(ds, split, n, seed=0):
    """n scene indices of `split` with two or more sources (cached)."""
    cache = ROOT / "p3" / "scenes3" / f"{split}_multi_idx.json"
    if cache.exists():
        keep = json.loads(cache.read_text())
    else:
        keep = [i for i, f in enumerate(ds.files)
                if int(np.load(f, allow_pickle=True)["K"]) >= 2]
        cache.write_text(json.dumps(keep))
    rs = np.random.RandomState(seed)
    return [int(i) for i in rs.choice(keep, min(n, len(keep)), replace=False)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval", "smoke"])
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dim", type=int, default=96)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--w_spec", type=float, default=0.5)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dm", type=float, default=0.0, help="probability of dynamic mixing per batch")
    ap.add_argument("--dm_db", type=float, default=6.0, help="per-stem gain range of dynamic mixing in dB")
    ap.add_argument("--init", default="", help="checkpoint in p3/ckpt whose weights start the run (fresh optimizer and schedule)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_ckpt", action="store_true", help="no gradient checkpointing (faster, more memory)")
    ap.add_argument("--hop", type=int, default=HOP)
    ap.add_argument("--no_amp", action="store_true", help="fp32 (no bf16 autocast)")
    ap.add_argument("--val_every", type=int, default=2000)
    ap.add_argument("--mask_bound", type=float, default=0.0, help="tanh bound on mask real/imag parts (0 = unbounded)")
    ap.add_argument("--w_quiet", type=float, default=0.5)
    ap.add_argument("--w_mix", type=float, default=0.2)
    ap.add_argument("--multi_only", action="store_true", help="train on scenes with two or more sources only")
    ap.add_argument("--k_cond", action="store_true", help="condition the separator on the source count K")
    ap.add_argument("--w_sum", type=float, default=0.0, help="weight of the scale-aware sum-to-mixture term")
    ap.add_argument("--aug_db", type=float, default=0.0, help="per-source random attenuation range in dB (0 = off)")
    ap.add_argument("--aug_tilt", type=float, default=0.0, help="per-source random spectral tilt: attenuation at 4 kHz in dB (0 = off)")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--n_val", type=int, default=96, help="multi-source validation scenes scored every val_every steps")
    args = ap.parse_args()
    torch.set_num_threads(4)
    if args.cmd == "train":
        train(args)
    elif args.cmd == "smoke":
        dev = "cuda"
        model = Sep3(args.dim, args.hidden, args.blocks, use_ckpt=not args.no_ckpt, amp=not args.no_amp, hop=args.hop,
                     mask_bound=args.mask_bound, k_cond=args.k_cond).to(dev)
        print("params", sum(p.numel() for p in model.parameters()) / 1e6, "M")
        y = torch.randn(args.batch, SEG, device=dev)
        gt = torch.randn(args.batch, MAXK, SEG, device=dev)
        ex = torch.ones(args.batch, MAXK, device=dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        torch.cuda.synchronize(); t0 = time.time()
        Ks = torch.full((args.batch,), 2, device=dev)
        for _ in range(5):
            pred = model(y, Ks) if args.k_cond else model(y)
            l, _ = pit_loss(pred, gt, ex)
            opt.zero_grad(); l.backward(); opt.step()
        torch.cuda.synchronize()
        print(f"{(time.time()-t0)/5:.2f} s/it, peak mem {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
    else:
        dev = "cuda"
        st = torch.load(ROOT / "p3" / "ckpt" / f"sep3{args.tag}_latest.pt", map_location=dev)
        a = st["args"]
        model = Sep3(a["dim"], a["hidden"], a["blocks"], hop=a.get("hop", HOP), mask_bound=a.get("mask_bound", 0.0),
                     k_cond=a.get("k_cond", False)).to(dev)
        model.load_state_dict(st["model"])
        ds = SceneSet(args.split)
        idxs = multi_idx(ds, args.split, args.n)
        print("stem SI-SDR median (multi-source scenes), per K:", evaluate(model, ds, idxs, dev))


if __name__ == "__main__":
    main()
