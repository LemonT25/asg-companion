"""R1 speed-conditioned stem extraction in the target-stationary domain.

For a pass-by source with speed estimate v_hat, the mixture and the
sep3 stem are both dewarped with v_hat (inverse pass-by: undo delay and
spreading). In that domain the target is close to a stationary source
while every other source is warped twice (its own motion plus the
inverse of the target's), which gives the extractor a cue that the
original-domain separator does not have. The extractor is the sep3
dual-path network with two input signals (dewarped mixture, dewarped
sep3 stem) and one complex mask applied to the dewarped mixture; its
output d_hat is the target in the dewarped domain. Warping d_hat forward
with v_hat gives a stem estimate (used for the subtraction), warping it
with the new speed gives the edited node. Because warp(dewarp(.)) is the
identity the stem estimate does not depend on the accuracy of v_hat.

Training target: dewarp(gt stem, v_hat) with v_hat drawn around the true
speed (synthetic relative error) or taken from the vel3 head on the
sep3 stem. Loss: negative SI-SDR plus a scale term. Warm start from a
sep3 checkpoint (blocks copied; input / output convolutions adapted).

  python src/p3_dex.py train --tag _x1 --init sep3_d3_best.pt --steps 8000
  python src/p3_dex.py eval  --tag _x1
"""
import argparse
import json
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

sys.path.insert(0, "/workspace/CF_ASG/src")
from p3_train import SR, MAXK, SEG  # noqa: E402
from p3_sep3 import DualPathBlock, N_FFT, si_sdr_t  # noqa: E402
from p3_vel3 import PassbySet, CK  # noqa: E402
from p3_vel4 import dewarp, load_stage1, C_SOUND, D0  # noqa: E402


def warp(x, v):
    """Batched forward pass-by (delay + 1/r spreading, no air absorption),
    the inverse of p3_vel4.dewarp: x (B, N) dry, v (B,) -> stem (B, N)."""
    B, N = x.shape
    t = torch.arange(N, device=x.device, dtype=torch.float32)
    tc = N / 2.0
    pos = v[:, None] * (t[None] - tc) / SR
    r = torch.sqrt(pos ** 2 + D0 ** 2)
    t_read = t[None] - (r / C_SOUND) * SR
    i0 = torch.floor(t_read).long().clamp(0, N - 2)
    w = (t_read - i0.float()).clamp(0, 1)
    x0 = torch.gather(x, 1, i0); x1 = torch.gather(x, 1, i0 + 1)
    y = (x0 + w * (x1 - x0)) / r
    return y * (t_read >= 0).float()


def r_of_t(v, N, device):
    t = torch.arange(N, device=device, dtype=torch.float32)
    pos = v[:, None] * (t[None] - N / 2.0) / SR
    return torch.sqrt(pos ** 2 + D0 ** 2)


def dewarp_ng(x, v):
    """Undo the Doppler delay only (spreading left in place): the network
    domain. dewarp(x, v) restores 1/r as well, which inflates the far
    edges by up to 30x and lets them dominate any waveform loss."""
    return dewarp(x, v) / r_of_t(v, x.shape[-1], x.device)


def rewarp_ng(d, v_dom, v_new):
    """d in the network domain (defined by v_dom) -> stem rendered at v_new."""
    return warp(d * r_of_t(v_dom, d.shape[-1], d.device), v_new)


V_EVAL = 20.0   # alternative speed for the node metric


class Dex(nn.Module):
    def __init__(self, dim=128, hidden=128, blocks=6, hop=160, mask_bound=2.0, use_ckpt=True, amp=True):
        super().__init__()
        self.hop, self.mask_bound, self.use_ckpt, self.amp = hop, mask_bound, use_ckpt, amp
        self.inp = nn.Sequential(nn.Conv2d(6, dim, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(dim, dim, 3, padding=1))
        self.blocks = nn.ModuleList([DualPathBlock(dim, hidden) for _ in range(blocks)])
        self.out = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.GELU(), nn.Conv2d(dim, 2, 1))
        self.register_buffer("win", torch.hann_window(N_FFT))

    def stft(self, x):
        return torch.stft(x, N_FFT, self.hop, window=self.win, return_complex=True)

    def feats(self, Z, scale):
        Zn = Z / scale
        return torch.stack([Zn.real, Zn.imag, Zn.abs().clamp_min(1e-5).log()], 1)

    def forward(self, ym, sm):                 # dewarped mixture, dewarped sep3 stem: (B, N) each
        N = ym.shape[-1]
        Z = self.stft(ym)
        scale = Z.abs().pow(2).mean((1, 2), keepdim=True).sqrt().clamp_min(1e-5)
        feat = torch.cat([self.feats(Z, scale), self.feats(self.stft(sm), scale)], 1)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
            h = self.inp(feat).permute(0, 2, 3, 1)
            for b in self.blocks:
                if self.use_ckpt and self.training:
                    h = torch.utils.checkpoint.checkpoint(b, h, use_reentrant=False)
                else:
                    h = b(h)
            m = self.out(h.permute(0, 3, 1, 2))
        m = m.float()
        if self.mask_bound > 0:
            m = self.mask_bound * torch.tanh(m / self.mask_bound)
        S = torch.complex(m[:, 0], m[:, 1]) * Z
        return torch.istft(S, N_FFT, self.hop, window=self.win, length=N)

    def init_from_sep3(self, name):
        st = torch.load(CK / name, map_location="cpu")
        sd = st["model"]
        own = self.state_dict()
        for k, v in sd.items():
            if k.startswith("blocks.") and k in own and own[k].shape == v.shape:
                own[k] = v
        w = sd["inp.0.weight"]                                   # (dim, 3, 3, 3) -> (dim, 6, 3, 3)
        own["inp.0.weight"] = torch.cat([w, 0.5 * w], 1)
        own["inp.0.bias"] = sd["inp.0.bias"]
        own["inp.2.weight"], own["inp.2.bias"] = sd["inp.2.weight"], sd["inp.2.bias"]
        own["out.0.weight"], own["out.0.bias"] = sd["out.0.weight"], sd["out.0.bias"]
        own["out.2.weight"] = sd["out.2.weight"][[0, MAXK]]      # slot 0 real / imag rows
        own["out.2.bias"] = sd["out.2.bias"][[0, MAXK]]
        self.load_state_dict(own)
        return st["step"]


class MultiPassbySet(torch.utils.data.Dataset):
    """PassbySet restricted to multi-source scenes (K >= 2)."""
    def __init__(self, split, multi_only=True):
        self.base = PassbySet(split)
        self.sel = [i for i, it in enumerate(self.base.items) if (it["K"] >= 2 or not multi_only)]
        self.items = [self.base.items[i] for i in self.sel]

    def __len__(self):
        return len(self.sel)

    def __getitem__(self, i):
        return self.base[self.sel[i]]


def draw_vhat(v, sep, s1, args, dev):
    B = v.shape[0]
    eps = (torch.randn(B, device=dev) * args.sigma_eps).clamp(-args.eps_max, args.eps_max)
    vh = v * (1 + eps)
    if args.p_real > 0 and s1 is not None:
        with torch.no_grad():
            v1 = s1.speed(s1(sep).float())
        use = torch.rand(B, device=dev) < args.p_real
        vh = torch.where(use, v1, vh)
    return vh.clamp(5.0, 40.0)


def snr_t(est, ref, eps=1e-8):
    return 10 * torch.log10(ref.pow(2).sum(-1) / ((est - ref).pow(2).sum(-1) + eps) + eps)


def loss_fn(d, tm, gt, vh, args, dev):
    """d, tm: network-domain output and target (dewarp_ng); gt: true stem.
    Stem-domain SI-SDR at v_hat plus the node rendered at a random speed."""
    st = rewarp_ng(d, vh, vh)
    crit = snr_t if args.loss == "snr" else si_sdr_t     # snr: sign and scale are learned
    l = -crit(st, gt).clamp(max=30.0).mean()
    if args.w_node > 0:
        v2 = (5.0 + 35.0 * torch.rand(vh.shape[0], device=dev))
        l = l + args.w_node * (-crit(rewarp_ng(d, vh, v2), rewarp_ng(tm, vh, v2)).clamp(max=30.0).mean())
    if args.w_scale > 0:
        e = 10 * torch.log10((st.pow(2).sum(-1) + 1e-8) / (gt.pow(2).sum(-1) + 1e-8))
        l = l + args.w_scale * e.abs().mean() / 10
    return l


@torch.no_grad()
def evaluate(model, s1, vds, dev, n=None):
    """Stem-domain SI-SDR of the rewarped extractor output and of the sep3
    stem against the true stem (oracle speed and vel3 estimate), and the
    node rendered at V_EVAL against the same from the true stem; by K."""
    model.eval()
    dl = torch.utils.data.DataLoader(vds, batch_size=16, shuffle=False, num_workers=4)
    keys = ("stem_dex_or", "stem_dex", "stem_sep", "node_dex", "node_sep")
    res = {k: [] for k in keys}
    Ks, seen = [], 0
    for gt, y, sep, v in dl:
        gt, y, sep, v = gt.to(dev), y.to(dev), sep.to(dev), v.to(dev)
        v1 = s1.speed(s1(sep).float()).clamp(5.0, 40.0)
        d_or = model(dewarp_ng(y, v), dewarp_ng(sep, v))
        res["stem_dex_or"] += si_sdr_t(rewarp_ng(d_or, v, v), gt).cpu().tolist()
        ym, sm, tm = dewarp_ng(y, v1), dewarp_ng(sep, v1), dewarp_ng(gt, v1)
        d = model(ym, sm)
        v2 = torch.full_like(v1, V_EVAL)
        res["stem_dex"] += si_sdr_t(rewarp_ng(d, v1, v1), gt).cpu().tolist()
        res.setdefault("stem_dex_snr", []).extend(snr_t(rewarp_ng(d, v1, v1), gt).cpu().tolist())
        res.setdefault("stem_sep_snr", []).extend(snr_t(sep, gt).cpu().tolist())
        res["stem_sep"] += si_sdr_t(sep, gt).cpu().tolist()
        res["node_dex"] += si_sdr_t(rewarp_ng(d, v1, v2), rewarp_ng(tm, v1, v2)).cpu().tolist()
        res["node_sep"] += si_sdr_t(rewarp_ng(sm, v1, v2), rewarp_ng(tm, v1, v2)).cpu().tolist()
        Ks += [vds.items[i]["K"] for i in range(seen, seen + gt.shape[0])]
        seen += gt.shape[0]
        if n and seen >= n:
            break
    model.train()
    Ks = np.array(Ks)
    out = {k: float(np.median(e)) for k, e in res.items()}
    for k in (2, 3):
        for name in ("stem_dex", "stem_sep", "node_dex", "node_sep"):
            e = np.array(res[name])
            out[f"{name}_K{k}"] = float(np.median(e[Ks == k])) if (Ks == k).any() else float("nan")
    out["n"] = len(res["stem_dex"])
    return out


def fmt(r):
    return (f"stem SI-SDR dex {r['stem_dex']:.2f} (oracle v {r['stem_dex_or']:.2f}) / sep3 {r['stem_sep']:.2f} "
            f"| K2 {r['stem_dex_K2']:.2f}/{r['stem_sep_K2']:.2f} K3 {r['stem_dex_K3']:.2f}/{r['stem_sep_K3']:.2f} "
            f"| node@20 dex {r['node_dex']:.2f} / sep3 {r['node_sep']:.2f} | SNR stem dex {r.get('stem_dex_snr', float('nan')):.2f} / sep3 {r.get('stem_sep_snr', float('nan')):.2f} n {r['n']}")


def train(args):
    torch.manual_seed(args.seed)
    dev = "cuda"
    model = Dex(dim=args.dim, hidden=args.hidden, blocks=args.blocks, hop=args.hop, mask_bound=args.mask_bound).to(dev)
    if args.init:
        print(f"init from {args.init} (step {model.init_from_sep3(args.init)})", flush=True)
    print(f"dex params={sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
    s1 = load_stage1(args.vel3, dev)
    ds, vds = MultiPassbySet("train"), MultiPassbySet("val")
    print(f"train items {len(ds)}, val items {len(vds)}", flush=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=6,
                                     drop_last=True, persistent_workers=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    def lr_at(st):
        if st < args.warm:
            return (st + 1) / args.warm
        q = (st - args.warm) / max(1, args.steps - args.warm)
        return 0.03 + 0.97 * 0.5 * (1 + np.cos(np.pi * q))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    step, t0, log, best = 0, time.time(), [], -1e9
    model.train()
    while step < args.steps:
        for gt, y, sep, v in dl:
            gt, y, sep, v = gt.to(dev), y.to(dev), sep.to(dev), v.to(dev)
            vh = draw_vhat(v, sep, s1, args, dev)
            ym, sm, tm = dewarp_ng(y, vh), dewarp_ng(sep, vh), dewarp_ng(gt, vh)
            d = model(ym, sm)
            loss = loss_fn(d, tm, gt, vh, args, dev)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            step += 1
            if step % 200 == 0:
                with torch.no_grad():
                    base = si_sdr_t(sep, gt).clamp(max=30).mean()
                    cur = si_sdr_t(rewarp_ng(d, vh, vh), gt).clamp(max=30).mean()
                print(f"{step} loss={float(loss):.3f} batch stem dex {float(cur):.2f} sep3 {float(base):.2f} ({(time.time()-t0)/200:.2f}s/it)", flush=True)
                t0 = time.time()
            if step % args.val_every == 0 or step >= args.steps:
                r = evaluate(model, s1, vds, dev)
                print("  val " + fmt(r), flush=True)
                log.append({"step": step, **r})
                state = {"model": model.state_dict(), "step": step, "args": vars(args), "val": r}
                torch.save(state, CK / f"dex{args.tag}_latest.pt")
                if r["stem_dex"] > best:
                    best = r["stem_dex"]
                    torch.save(state, CK / f"dex{args.tag}_best.pt")
                    print(f"  new best rewarped stem {best:.2f} at step {step}", flush=True)
                (CK / f"dex{args.tag}_log.json").write_text(json.dumps(log, indent=1))
            if step >= args.steps:
                break
    print("TRAINING DONE dex", flush=True)


def load_dex(name, dev):
    st = torch.load(CK / name, map_location=dev)
    a = st["args"]
    m = Dex(dim=a["dim"], hidden=a["hidden"], blocks=a["blocks"], hop=a["hop"], mask_bound=a["mask_bound"]).to(dev)
    m.load_state_dict(st["model"])
    m.eval()
    return m, st["step"]


def eval_cmd(args):
    dev = "cuda"
    model, step = load_dex(f"dex{args.tag}_best.pt", dev)
    s1 = load_stage1(args.vel3, dev)
    r = evaluate(model, s1, MultiPassbySet(args.split, multi_only=not args.all_k), dev)
    print(json.dumps({"split": args.split, "step": step, **r}, indent=1))
    print(fmt(r))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval"])
    ap.add_argument("--split", default="val")
    ap.add_argument("--tag", default="")
    ap.add_argument("--init", default="sep3_d3_best.pt", help="sep3 checkpoint for the warm start ('' = scratch)")
    ap.add_argument("--vel3", default="vel3_v8_best.pt")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--hop", type=int, default=160)
    ap.add_argument("--mask_bound", type=float, default=2.0)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--warm", type=int, default=200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--val_every", type=int, default=500)
    ap.add_argument("--sigma_eps", type=float, default=0.06, help="spread of the synthetic speed error")
    ap.add_argument("--eps_max", type=float, default=0.15)
    ap.add_argument("--p_real", type=float, default=0.3, help="share of items whose v_hat is the vel3 estimate")
    ap.add_argument("--w_scale", type=float, default=0.3)
    ap.add_argument("--loss", default="sisdr", choices=["sisdr", "snr"])
    ap.add_argument("--w_node", type=float, default=0.5, help="weight of the node term at a random new speed")
    ap.add_argument("--all_k", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"train": train, "eval": eval_cmd}[a.cmd](a)
