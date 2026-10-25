"""R1 speed refinement: residual estimation after exact dewarping.

Stage one (vel3) gives a speed v_hat. This module dewarps the stem with v_hat
using the inverse of the renderer's pass-by operator (undo the 1/r spreading
and the propagation delay for the straight-line geometry d0 = 2 m, closest
approach at mid-segment), and a second network estimates the relative
residual eps = v / v_hat - 1 from the dewarped signal alone. If v_hat is
right, the dewarped signal carries no Doppler glide and no loudness bump
centred on t0; if it is off by eps the residual glide and bump scale with
eps. The network therefore solves a translation-invariant local problem
instead of absolute regression over 5 to 40 m/s. At inference the estimate
is refined iteratively.

Training: for each pass-by stem (ground truth with bleed, or cached sep3
stem, as in p3_vel3.augment) draw eps ~ N(0, sigma_eps) clipped to +-eps_max,
dewarp with v (1 + eps) and predict -eps... written as the residual grid
over [-eps_max, eps_max] with soft targets and expectation decoding.

Commands: train, eval (two-stage error on the val cache, by K).
"""
import argparse
import json
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/workspace/CF_ASG/src")
from p3_train import SR  # noqa: E402
from p3_vel3 import (CK, PassbySet, Vel3, augment, NBIN)  # noqa: E402

C_SOUND = 343.0
D0 = 2.0


# ----------------------------------------------------------------- dewarp
def dewarp(x, v):
    """Batched inverse pass-by on GPU: x (B, N), v (B,) -> dry estimate
    (B, N). Mirrors p3_edit_eval.inverse_passby (air absorption neglected)."""
    B, N = x.shape
    t = torch.arange(N, device=x.device, dtype=torch.float32)
    tc = N / 2.0
    pos = v[:, None] * (t[None] - tc) / SR                       # (B, N) along-track position
    r = torch.sqrt(pos ** 2 + D0 ** 2)
    t_read = t[None] - (r / C_SOUND) * SR                        # emission sample for output t
    xs = x * r                                                   # undo spreading
    # linear interpolation of xs (sampled at t_read, monotone) onto t
    idx = torch.searchsorted(t_read.contiguous(), t[None].expand(B, N).contiguous()) # first t_read >= t
    i1 = idx.clamp(1, N - 1)
    i0 = i1 - 1
    x0 = torch.gather(xs, 1, i0); x1 = torch.gather(xs, 1, i1)
    t0 = torch.gather(t_read, 1, i0); t1 = torch.gather(t_read, 1, i1)
    w = ((t[None] - t0) / (t1 - t0).clamp_min(1e-6)).clamp(0, 1)
    d = x0 + w * (x1 - x0)
    inside = (t[None] >= t_read[:, :1]) & (t[None] <= t_read[:, -1:])
    return d * inside.float()


class Vel4(nn.Module):
    """Vel3 backbone over the dewarped stem; output: residual grid."""
    def __init__(self, eps_max=0.10, nres=81, ch=(32, 48, 64, 96, 128), hid=192, phase=False):
        super().__init__()
        self.eps_max, self.nres = eps_max, nres
        self.net = Vel3(ch=ch, hid=hid, nbins=nres, phase=phase)
        self.register_buffer("rgrid", torch.linspace(-eps_max, eps_max, nres))

    def forward(self, d):
        return self.net(d)

    def residual(self, logits):
        p = torch.softmax(logits, -1)
        return (p * self.rgrid.to(logits.device)).sum(-1)

    def soft_targets(self, eps, sigma=1.0):
        g = self.rgrid.to(eps.device)
        z = (eps[:, None] - g[None]) / (sigma * (g[1] - g[0]))
        t = torch.exp(-0.5 * z ** 2)
        return t / t.sum(-1, keepdim=True)

    @torch.no_grad()
    def refine(self, x, v_hat, iters=3):
        """x (B, N) stems, v_hat (B,) first-stage speeds -> refined speeds."""
        v = v_hat.clone()
        for _ in range(iters):
            d = dewarp(x, v)
            eps = self.residual(self(d).float())
            v = (v * (1 + eps)).clamp(5.0, 40.0)
        return v


def load_stage1(name, dev):
    st = torch.load(CK / name, map_location=dev)
    m = Vel3(mix_in=False, nbins=st["args"].get("nbins", NBIN)).to(dev)
    m.load_state_dict(st["model"], strict=False)
    m.eval()
    return m


@torch.no_grad()
def evaluate(model, s1, vds, dev, iters=3, n=None):
    model.eval()
    dl = torch.utils.data.DataLoader(vds, batch_size=32, shuffle=False, num_workers=4)
    err = {"gt1": [], "gt2": [], "sep1": [], "sep2": []}
    Ks = []
    seen = 0
    for bi, (gt, y, sep, v) in enumerate(dl):
        gt, y, sep, v = gt.to(dev), y.to(dev), sep.to(dev), v.to(dev)
        for name, x in (("gt", gt), ("sep", sep)):
            v1 = s1.speed(s1(x).float())
            v2 = model.refine(x, v1, iters)
            err[name + "1"] += ((v1 - v).abs() / v).cpu().tolist()
            err[name + "2"] += ((v2 - v).abs() / v).cpu().tolist()
        Ks += [vds.items[i]["K"] for i in range(seen, seen + gt.shape[0])]
        seen += gt.shape[0]
        if n and seen >= n:
            break
    model.train()
    Ks = np.array(Ks)
    out = {k: float(np.median(e)) for k, e in err.items()}
    for k in (1, 2, 3):
        for name in ("gt2", "sep2"):
            e = np.array(err[name])
            out[f"{name}_K{k}"] = float(np.median(e[Ks == k]))
    out["n"] = len(err["gt1"])
    return out


def train(args):
    torch.manual_seed(args.seed)
    dev = "cuda"
    model = Vel4(eps_max=args.eps_max, nres=args.nres, phase=args.phase).to(dev)
    print(f"vel4 params={sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
    s1 = load_stage1(args.stage1, dev)
    ds, vds = PassbySet("train"), PassbySet("val")
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=6,
                                     drop_last=True, persistent_workers=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    def lr_at(st):
        if st < 0.5 * args.steps:
            return 1.0
        q = (st - 0.5 * args.steps) / (0.5 * args.steps)
        return 0.02 + 0.98 * 0.5 * (1 + np.cos(np.pi * q))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    step, t0, log, best = 0, time.time(), [], 1e9
    model.train()
    while step < args.steps:
        for gt, y, sep, v in dl:
            gt, y, sep, v = gt.to(dev), y.to(dev), sep.to(dev), v.to(dev)
            ramp = min(1.0, step / max(1.0, args.ramp * args.steps))
            cur = argparse.Namespace(p_sep=args.p_sep * ramp, bleed=args.bleed * ramp, mix_in=False)
            inp = augment(gt, y, sep, cur, dev)
            # perturbed first-stage estimate: either the real stage-one
            # output (p_real) or v scaled by a random relative error
            B = v.shape[0]
            eps_draw = (torch.randn(B, device=dev) * args.sigma_eps).clamp(-args.eps_max, args.eps_max)
            v_pert = v * (1 + eps_draw)
            if args.p_real > 0:
                with torch.no_grad():
                    v1 = s1.speed(s1(inp).float())
                use = (torch.rand(B, device=dev) < args.p_real)
                v_pert = torch.where(use, v1, v_pert)
            eps = (v / v_pert - 1).clamp(-args.eps_max, args.eps_max)   # target residual
            d = dewarp(inp, v_pert)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(d)
            loss = -(model.soft_targets(eps, args.sigma) * F.log_softmax(logits.float(), -1)).sum(-1).mean()
            if args.w_l1 > 0:
                loss = loss + args.w_l1 * (model.residual(logits.float()) - eps).abs().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            step += 1
            if step % 200 == 0:
                with torch.no_grad():
                    e = (model.residual(logits.float()) - eps).abs().median()
                print(f"{step} loss={float(loss):.3f} batch |res err| {float(e):.4f} ({(time.time()-t0)/200:.2f}s/it)", flush=True)
                t0 = time.time()
            if step % args.val_every == 0 or step >= args.steps:
                r = evaluate(model, s1, vds, dev, iters=args.iters)
                print(f"  val rel err: gt {r['gt1']:.4f} -> {r['gt2']:.4f}  sep3 {r['sep1']:.4f} -> {r['sep2']:.4f} "
                      f"| by K gt2 {r['gt2_K1']:.4f}/{r['gt2_K2']:.4f}/{r['gt2_K3']:.4f} "
                      f"sep2 {r['sep2_K1']:.4f}/{r['sep2_K2']:.4f}/{r['sep2_K3']:.4f} n {r['n']}", flush=True)
                log.append({"step": step, **r})
                state = {"model": model.state_dict(), "step": step, "args": vars(args), "val": r}
                torch.save(state, CK / f"vel4{args.tag}_latest.pt")
                if r["sep2"] < best:
                    best = r["sep2"]
                    torch.save(state, CK / f"vel4{args.tag}_best.pt")
                    print(f"  new best sep3 rel err {best:.4f} at step {step}", flush=True)
                (CK / f"vel4{args.tag}_log.json").write_text(json.dumps(log, indent=1))
            if step >= args.steps:
                break
    print("TRAINING DONE vel4", flush=True)


def eval_cmd(args):
    dev = "cuda"
    st = torch.load(CK / f"vel4{args.tag}_best.pt", map_location=dev)
    model = Vel4(eps_max=st["args"]["eps_max"], nres=st["args"]["nres"], phase=st["args"].get("phase", False)).to(dev)
    model.load_state_dict(st["model"])
    s1 = load_stage1(args.stage1, dev)
    r = evaluate(model, s1, PassbySet(args.split), dev, iters=args.iters)
    print(json.dumps({"split": args.split, "step": st["step"], "stage1": args.stage1, "iters": args.iters, **r}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval"])
    ap.add_argument("--split", default="val")
    ap.add_argument("--stage1", default="vel3_v8_best.pt")
    ap.add_argument("--tag", default="")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_every", type=int, default=1000)
    ap.add_argument("--p_sep", type=float, default=0.5)
    ap.add_argument("--bleed", type=float, default=0.6)
    ap.add_argument("--ramp", type=float, default=0.2)
    ap.add_argument("--eps_max", type=float, default=0.10, help="half width of the residual grid")
    ap.add_argument("--nres", type=int, default=81)
    ap.add_argument("--sigma", type=float, default=1.0, help="soft target width in residual bins")
    ap.add_argument("--sigma_eps", type=float, default=0.03, help="spread of the synthetic first-stage error")
    ap.add_argument("--p_real", type=float, default=0.3, help="share of samples perturbed by the real stage-one estimate")
    ap.add_argument("--w_l1", type=float, default=5.0)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--phase", action="store_true", help="add the instantaneous-frequency channel")
    a = ap.parse_args()
    {"train": train, "eval": eval_cmd}[a.cmd](a)
