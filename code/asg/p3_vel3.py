"""Dedicated pass-by speed estimator (vel3) for the R1 revision.

The deployed speed head (Est2, src/p3_train.py) is a five-block conv
stack with global average pooling trained on ground-truth stems with
bleed augmentation. Its relative error on separated stems is 0.13 to
0.19 and the speed edit E2 needs a few percent (see the tolerance
derivation). vel3 keeps the time axis (the pass-by geometry is fixed:
closest approach 2 m at the clip center, so the Doppler glide and the
1/r envelope are functions of v and absolute time only), reads the
spectrogram with frequency-strided convolutions, a bidirectional GRU
over time with a learned time embedding and attention pooling, and
predicts a distribution over a speed grid. It is trained on both the
ground-truth stems and the sep3 front end's stems (cached once by the
`cache` command), so the deployment domain is in the training set.

  p3_vel3.py cache --split train|val|conf1 [--ckpt sep3_f1_best.pt]
  p3_vel3.py train --tag _v1 --steps 20000
  p3_vel3.py eval  --tag _v1 --split val
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

sys.path.insert(0, "/workspace/CF_ASG/src")
from p3_train import SceneSet, SEG, MAXK, ROOT, Est2  # noqa: E402

CK = ROOT / "p3" / "ckpt"
CACHE = ROOT / "p3" / "vel3_cache"
V_LO, V_HI, NBIN = 5.0, 40.0, 71
V_GRID = torch.linspace(V_LO, V_HI, NBIN)


def make_grid(nbins):
    return torch.linspace(V_LO, V_HI, nbins)


def si_sdr_t(est, ref):
    ref = ref - ref.mean(-1, keepdim=True)
    est = est - est.mean(-1, keepdim=True)
    a = (est * ref).sum(-1, keepdim=True) / (ref.pow(2).sum(-1, keepdim=True) + 1e-8)
    e = est - a * ref
    return 10 * torch.log10((a * ref).pow(2).sum(-1) / (e.pow(2).sum(-1) + 1e-8) + 1e-8)


# ----------------------------------------------------------------- cache
def cache(args):
    from p3_sep3 import Sep3
    dev = "cuda"
    st = torch.load(CK / args.ckpt, map_location=dev)
    a = st["args"]
    m = Sep3(a["dim"], a["hidden"], a["blocks"], hop=a.get("hop", 128), mask_bound=a.get("mask_bound", 0.0),
             k_cond=a.get("k_cond", False)).to(dev)
    m.load_state_dict(st["model"], strict=False)
    m.eval()
    ds = SceneSet(args.split)
    dl = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False, num_workers=6)   # small batch: the GPU is shared with two training runs
    seps, idx = [], []
    perms = list(itertools.permutations(range(MAXK)))
    with torch.no_grad():
        base = 0
        for y, stems, dries, exist, traj, thn, alpha, env, K in dl:
            B = y.shape[0]
            pred = m(y.to(dev))[..., :SEG]
            gts = stems.to(dev)
            for b in range(B):
                Kb = int(K[b])
                if Kb == 1:
                    match = {0: y[b].to(dev)}            # runner convention: the mixture is the stem
                else:
                    best, bp = -1e9, perms[0]
                    for p in perms:
                        v = sum(float(si_sdr_t(pred[b, p[k]][None], gts[b, k][None])) for k in range(Kb))
                        if v > best:
                            best, bp = v, p
                    match = {k: pred[b, bp[k]] for k in range(Kb)}
                for k in range(Kb):
                    if int(traj[b, k]) != 0:
                        continue                         # pass-by sources only
                    seps.append(match[k].cpu().half().numpy())
                    idx.append({"scene": base + b, "k": k, "K": Kb, "v": 5.0 + 35.0 * float(thn[b, k]),
                                "sisdr": float(si_sdr_t(match[k][None], gts[b, k][None]))})
            base += B
            if base % 1600 == 0 or base == 400:
                print(f"{base}/{len(ds)} ({len(idx)} pass-by stems)", flush=True)
    CACHE.mkdir(exist_ok=True)
    np.save(CACHE / f"{args.split}_sep.npy", np.stack(seps))
    (CACHE / f"{args.split}_idx.json").write_text(json.dumps({"ckpt": args.ckpt, "items": idx}))
    s = np.array([d["sisdr"] for d in idx])
    print(f"cached {len(idx)} pass-by stems for {args.split}; sep3 stem SI-SDR median {np.median(s):.2f} dB", flush=True)


# ----------------------------------------------------------------- data
class PassbySet(torch.utils.data.Dataset):
    """One item per pass-by source: ground-truth stem, mixture, cached
    sep3 stem and the speed."""
    def __init__(self, split):
        self.ds = SceneSet(split)
        self.sep = np.load(CACHE / f"{split}_sep.npy", mmap_mode="r")
        self.items = json.loads((CACHE / f"{split}_idx.json").read_text())["items"]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        y, stems, dries, exist, traj, thn, alpha, env, K = self.ds[it["scene"]]
        return (stems[it["k"]], y, torch.tensor(np.asarray(self.sep[i], dtype=np.float32)),
                torch.tensor(it["v"], dtype=torch.float32))


# ----------------------------------------------------------------- model
class Vel3(nn.Module):
    def __init__(self, ch=(32, 48, 64, 96, 128), hid=192, nfft=1024, hop=256, mix_in=False, nbins=NBIN, phase=False):
        super().__init__()
        self.register_buffer("grid", make_grid(nbins))
        self.phase = phase                   # second channel: instantaneous frequency deviation
        self.nfft, self.hop = nfft, hop
        self.mix_in = mix_in                 # second input channel: the mixture spectrogram
        blocks, ci = [], (2 if mix_in else 1) + (1 if phase else 0)
        for co in ch:
            blocks.append(nn.Sequential(nn.Conv2d(ci, co, 3, stride=(2, 1), padding=1),
                                        nn.GroupNorm(4, co), nn.SiLU()))
            ci = co
        self.conv = nn.Sequential(*blocks)
        nf = nfft // 2 + 1
        for _ in ch:
            nf = (nf + 1) // 2
        self.proj = nn.Linear(ch[-1] * nf, hid)
        self.T = SEG // hop + 1
        self.pos = nn.Parameter(torch.zeros(1, self.T, hid))
        nn.init.normal_(self.pos, std=0.02)
        self.gru = nn.GRU(hid, hid, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.att = nn.Linear(2 * hid, 1)
        self.head = nn.Sequential(nn.Linear(2 * hid, hid), nn.SiLU(), nn.Linear(hid, nbins))

    def feats(self, s):
        w = torch.hann_window(self.nfft, device=s.device)
        Z = torch.stft(s.float(), self.nfft, self.hop, window=w, return_complex=True)
        X = Z.abs().pow(2).clamp_min(1e-10).log()[:, None]       # (B, 1, F, T)
        X = X - X.mean(dim=(2, 3), keepdim=True)
        X = X / 4.0
        if self.phase:
            # phase advance between frames minus the bin's nominal advance,
            # wrapped: a sub-bin frequency offset shows as a constant, a
            # glide as a drift (log power alone cannot resolve sub-bin shifts)
            k = torch.arange(Z.shape[1], device=s.device, dtype=torch.float32)
            nominal = 2 * torch.pi * k * self.hop / self.nfft
            dphi = torch.angle(Z[:, :, 1:] * Z[:, :, :-1].conj()) - nominal[None, :, None]
            dphi = torch.remainder(dphi + torch.pi, 2 * torch.pi) - torch.pi
            dphi = F.pad(dphi, (1, 0)) / torch.pi
            X = torch.cat([X, dphi[:, None]], 1)
        return X

    def forward(self, s, y=None):
        X = self.feats(s)
        if self.mix_in:
            X = torch.cat([X, self.feats(y)], 1)
        h = self.conv(X)                                          # (B, C, F', T)
        B, C, Fp, T = h.shape
        h = h.permute(0, 3, 1, 2).reshape(B, T, C * Fp)
        h = self.proj(h) + self.pos[:, :T]
        h, _ = self.gru(h)
        a = torch.softmax(self.att(h).squeeze(-1), dim=1)       # attention pooling over time
        z = (a[..., None] * h).sum(1)
        return self.head(z)                                       # (B, NBIN) logits

    def speed(self, logits):
        p = torch.softmax(logits, -1)
        return (p * self.grid.to(logits.device)).sum(-1)


class Vel3AsEst2:
    """Drop-in replacement for the Est2 head: called on a batch of stems
    it returns (traj logits placeholder, normalized speed in [0, 1])."""
    def __init__(self, path, dev):
        st = torch.load(path, map_location=dev)
        self.m = Vel3(mix_in=st["args"].get("mix_in", False), nbins=st["args"].get("nbins", NBIN), phase=st["args"].get("phase", False)).to(dev)
        self.m.load_state_dict(st["model"], strict=False)
        self.m.eval()
        self.step = st.get("step")
        assert not self.m.mix_in, "mixture-channel models need the mixture"

    def eval(self):
        return self

    @torch.no_grad()
    def __call__(self, x):
        v = self.m.speed(self.m(x))
        return torch.zeros(x.shape[0], 2, device=x.device), (v - 5.0) / 35.0


def soft_targets(v, grid, sigma=1.0):
    g = grid.to(v.device)
    d = (v[:, None] - g[None]) / (sigma * (g[1] - g[0]))
    t = torch.exp(-0.5 * d ** 2)
    return t / t.sum(-1, keepdim=True)


def augment(gt, y, sep, args, dev):
    """Per-sample input choice and corruption: with probability p_sep the
    sep3 stem, else the ground-truth stem plus a random share of the rest
    of the mixture (bleed); then a random gain and two random time
    dropouts as in Est2."""
    B, N = gt.shape
    use_sep = (torch.rand(B, 1, device=dev) < args.p_sep).float()
    lam = torch.rand(B, 1, device=dev) * args.bleed
    inp = use_sep * sep + (1 - use_sep) * (gt + lam * (y - gt))
    g = 10 ** ((torch.rand(B, 1, device=dev) * 20 - 10) / 20)
    inp = inp * g
    for _ in range(2):
        wlen = int(0.1 * N)
        st = torch.randint(0, N - wlen, (B,), device=dev)
        ar = torch.arange(N, device=dev)[None, :]
        mask = (ar >= st[:, None]) & (ar < st[:, None] + wlen)
        inp = inp * (~mask | (torch.rand(B, 1, device=dev) > 0.5)).float()
    if getattr(args, "mix_in", False):
        return inp, y * g
    return inp


@torch.no_grad()
def evaluate(model, vds, dev, e2=None, n=None):
    model.eval()
    dl = torch.utils.data.DataLoader(vds, batch_size=32, shuffle=False, num_workers=4)
    err = {"gt": [], "sep": []}
    err2 = {"gt": [], "sep": []}
    seen = 0
    for gt, y, sep, v in dl:
        gt, y, sep, v = gt.to(dev), y.to(dev), sep.to(dev), v.to(dev)
        for name, x in (("gt", gt), ("sep", sep)):
            vh = model.speed(model(x, y) if model.mix_in else model(x))
            err[name] += ((vh - v).abs() / v).cpu().tolist()
            if e2 is not None:
                _, pv = e2(x)
                v2 = 5.0 + 35.0 * pv.clamp(0, 1)
                err2[name] += ((v2 - v).abs() / v).cpu().tolist()
        seen += gt.shape[0]
        if n and seen >= n:
            break
    model.train()
    out = {k: float(np.median(e)) for k, e in err.items()}
    if e2 is not None:
        out.update({k + "_est2": float(np.median(e)) for k, e in err2.items()})
    out["n"] = len(err["gt"])
    return out


# ----------------------------------------------------------------- train
def train(args):
    torch.manual_seed(args.seed)
    dev = "cuda"
    model = Vel3(mix_in=args.mix_in, nbins=args.nbins, phase=args.phase).to(dev)
    print(f"vel3 params={sum(p.numel() for p in model.parameters())/1e6:.2f}M mix_in={args.mix_in}", flush=True)
    ds = PassbySet("train")
    vds = PassbySet("val")
    print(f"train pass-by stems {len(ds)}, val {len(vds)}", flush=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=6,
                                     drop_last=True, persistent_workers=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # constant learning rate for the first half, cosine decay to 2 percent
    # after (a one-cycle warm-up from 2.5e-5 left the network at the uniform
    # prediction for over 1000 steps)
    def lr_at(st):
        if st < args.clean_steps:
            return 1.0
        if st < 0.5 * args.steps:
            return args.lr_corrupt / args.lr
        q = (st - 0.5 * args.steps) / (0.5 * args.steps)
        return (args.lr_corrupt / args.lr) * (0.02 + 0.98 * 0.5 * (1 + np.cos(np.pi * q)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    if args.init:
        st0 = torch.load(CK / args.init, map_location=dev)
        model.load_state_dict(st0["model"], strict=False)
        print(f"init weights from {args.init} (step {st0.get('step')})", flush=True)
    e2 = Est2().to(dev)
    e2.load_state_dict(torch.load(CK / "est2_latest.pt", map_location=dev)["model"])
    e2.eval()
    step, t0, log, best = 0, time.time(), [], 1e9
    model.train()
    while step < args.steps:
        for gt, y, sep, v in dl:
            gt, y, sep, v = gt.to(dev), y.to(dev), sep.to(dev), v.to(dev)
            # curriculum: clean ground-truth stems first, then the bleed
            # share and the sep3 input probability ramp up over the first
            # args.ramp fraction of training (with the full corruption from
            # the start the network found no signal in 1000 steps)
            ramp = min(1.0, max(0.0, step - args.clean_steps) / max(1.0, args.ramp * args.steps))
            cur = argparse.Namespace(p_sep=args.p_sep * ramp, bleed=args.bleed * ramp, mix_in=args.mix_in)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if args.mix_in:
                    inp, ym = augment(gt, y, sep, cur, dev)
                    logits = model(inp, ym)
                else:
                    inp = augment(gt, y, sep, cur, dev)
                    logits = model(inp)
            loss = -(soft_targets(v, model.grid, args.sigma) * F.log_softmax(logits.float(), -1)).sum(-1).mean()
            if args.w_l1 > 0:   # relative L1 on the decoded speed: sharper than the grid pitch
                loss = loss + args.w_l1 * ((model.speed(logits.float()) - v).abs() / v).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            step += 1
            if step % 200 == 0:
                with torch.no_grad():
                    rel = float(((model.speed(logits.float()) - v).abs() / v).median())
                print(f"{step} loss={float(loss):.3f} batch rel err {rel:.3f} ({(time.time()-t0)/200:.2f}s/it)", flush=True)
                t0 = time.time()
            if step % args.val_every == 0 or step >= args.steps:
                r = evaluate(model, vds, dev, e2=e2 if step == args.val_every else None)
                log.append({"step": step, **r})
                print(f"  val median rel err: gt {r['gt']:.3f} sep3 {r['sep']:.3f}"
                      + (f" (Est2: gt {r['gt_est2']:.3f} sep3 {r['sep_est2']:.3f})" if "gt_est2" in r else "")
                      + f" n {r['n']}", flush=True)
                state = {"model": model.state_dict(), "step": step, "args": vars(args), "val": r}
                torch.save(state, CK / f"vel3{args.tag}_latest.pt")
                if r["sep"] < best:
                    best = r["sep"]
                    torch.save(state, CK / f"vel3{args.tag}_best.pt")
                    print(f"  new best sep3 rel err {best:.3f} at step {step}", flush=True)
                (CK / f"vel3{args.tag}_log.json").write_text(json.dumps(log))
            if step >= args.steps:
                break
    print("TRAINING DONE vel3", flush=True)


def eval_cmd(args):
    dev = "cuda"
    st = torch.load(CK / f"vel3{args.tag}_best.pt", map_location=dev)
    model = Vel3(mix_in=st["args"].get("mix_in", False), nbins=st["args"].get("nbins", NBIN), phase=st["args"].get("phase", False)).to(dev)
    model.load_state_dict(st["model"], strict=False)
    e2 = Est2().to(dev)
    e2.load_state_dict(torch.load(CK / "est2_latest.pt", map_location=dev)["model"])
    e2.eval()
    r = evaluate(model, PassbySet(args.split), dev, e2=e2)
    print(json.dumps({"split": args.split, "step": st["step"], **r}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cache", "train", "eval"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--ckpt", default="sep3_f1_best.pt")
    ap.add_argument("--tag", default="")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_every", type=int, default=1000)
    ap.add_argument("--p_sep", type=float, default=0.5, help="probability of feeding the sep3 stem")
    ap.add_argument("--bleed", type=float, default=0.6, help="max bleed share added to a ground-truth stem")
    ap.add_argument("--ramp", type=float, default=0.3, help="fraction of training over which bleed and p_sep ramp up")
    ap.add_argument("--clean_steps", type=int, default=1500, help="clean ground-truth stems only before the ramp starts")
    ap.add_argument("--lr_corrupt", type=float, default=1e-3, help="learning rate after the clean phase")
    ap.add_argument("--mix_in", action="store_true", help="feed the mixture spectrogram as a second channel")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nbins", type=int, default=NBIN, help="speed grid bins over 5 to 40 m/s")
    ap.add_argument("--sigma", type=float, default=1.0, help="soft target width in bins")
    ap.add_argument("--w_l1", type=float, default=0.0, help="weight of the relative L1 loss on the decoded speed")
    ap.add_argument("--init", default="", help="vel3 checkpoint in p3/ckpt whose weights start the run")
    ap.add_argument("--phase", action="store_true", help="add the instantaneous-frequency channel")
    args = ap.parse_args()
    {"cache": cache, "train": train, "eval": eval_cmd}[args.cmd](args)
