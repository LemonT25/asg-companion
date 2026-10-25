"""R1 slot naming head: which separated stem is which source class.

The evaluation addresses a source by its class name (the queries the
external separators receive). Our separator returns unnamed slots, and
until now the result files matched them to the edited source by the
best permutation against the ground-truth stems. This module learns the
naming: a small time-frequency classifier (the speed estimator's
backbone with a 23-way output) reads a posterior stem (and optionally
the mixture) and scores the 23 classes. At inference the K live slots
of a scene are assigned to the K class names of the scene by the
Hungarian algorithm on the log-probabilities, so no ground truth is
used.

Training items are the posterior stems of the frozen separator on the
training scenes, labeled through the oracle permutation, mixed with
ground-truth stems plus bleed. Validation reports per-slot accuracy
and per-scene agreement with the oracle permutation.

  python src/p3_cls.py train --tag _c1
  python src/p3_cls.py train --tag _c2 --mix_in
  python src/p3_cls.py eval --tag _c1
"""
import argparse
import itertools
import json
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, "/workspace/CF_ASG/src")
from p3_train import SceneSet, SEG, MAXK, SR  # noqa: E402
from p3_sep3 import Sep3, si_sdr_t  # noqa: E402
from p3_vel3 import Vel3, CK  # noqa: E402
from p3_scenes import BROAD  # noqa: E402

CLASSES = list(BROAD)
CIDX = {c: i for i, c in enumerate(CLASSES)}
PERMS = list(itertools.permutations(range(MAXK)))


def load_sep3(name, dev):
    st = torch.load(CK / name, map_location=dev)
    a = st["args"]
    m = Sep3(a["dim"], a["hidden"], a["blocks"], hop=a.get("hop", 128), mask_bound=a.get("mask_bound", 0.0),
             k_cond=a.get("k_cond", False)).to(dev)
    m.load_state_dict(st["model"])
    m.eval()
    return m


def posterior_t(y, stems):
    """Wiener consistency projection on GPU: y (B, N), stems (B, K, N) -> (B, K, N)."""
    B, K, N = stems.shape
    w = torch.hann_window(1024, device=y.device)
    Zy = torch.stft(y, 1024, 256, window=w, return_complex=True)
    Zs = torch.stft(stems.reshape(B * K, N), 1024, 256, window=w, return_complex=True).reshape(B, K, *Zy.shape[1:])
    P = Zs.abs() ** 2
    M = P / (P.sum(1, keepdim=True) + 1e-10)
    S = M * Zy[:, None]
    return torch.istft(S.reshape(B * K, *S.shape[2:]), 1024, 256, window=w, length=N).reshape(B, K, N)


def pit_perm(post, gts, K):
    """Best permutation of post (MAXK, N) against gts (K, N) by summed SI-SDR."""
    best, bp = -1e9, PERMS[0]
    for p in PERMS:
        v = float(sum(si_sdr_t(post[p[k]][None], gts[k][None]) for k in range(K)))
        if v > best:
            best, bp = v, p
    return bp


class ClsScenes(torch.utils.data.Dataset):
    def __init__(self, split, n=None):
        self.ds = SceneSet(split)
        self.n = len(self.ds) if n is None else min(n, len(self.ds))

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        y, stems, dries, exist, traj, thn, alpha, env, K = self.ds[i]
        meta = json.loads(str(np.load(self.ds.files[i], allow_pickle=True)["meta"]))
        cls = torch.full((MAXK,), -1, dtype=torch.long)
        for k, m in enumerate(meta[:MAXK]):
            cls[k] = CIDX[m["cls"]]
        return y[:SEG], stems[..., :SEG], torch.tensor(int(K)), cls


class SlotNamer(nn.Module):
    def __init__(self, mix_in=False):
        super().__init__()
        self.mix_in = mix_in
        self.net = Vel3(mix_in=mix_in, nbins=len(CLASSES))

    def forward(self, x, y=None):
        return self.net(x, y) if self.mix_in else self.net(x)

    @torch.no_grad()
    def assign(self, post, y, live, classes):
        """post (MAXK, N) posterior stems, y (N,), live: slot indices, classes:
        class indices of the K sources -> tuple perm with perm[k] = slot of source k."""
        K = len(classes)
        x = post[live]
        lp = F.log_softmax(self(x, y[None].expand(len(live), -1) if self.mix_in else None).float(), -1)
        cost = -lp[:, classes].cpu().numpy()            # (slots, sources)
        r, c = linear_sum_assignment(cost)
        perm = [None] * K
        for ri, ci in zip(r, c):
            perm[ci] = live[ri]
        return tuple(perm)


def make_items(sep3, y, stems, K, cls, args, dev, train=True):
    """One training item per live source: (input stem, mixture, label)."""
    B = y.shape[0]
    with torch.no_grad():
        pred = sep3(y)[..., :SEG]
        post = posterior_t(y, pred)
    xs, ys, ls = [], [], []
    for b in range(B):
        Kb = int(K[b])
        if Kb == 1:
            cand = [y[b]]
        else:
            p = pit_perm(post[b], stems[b, :Kb], Kb)
            cand = [post[b, p[k]] for k in range(Kb)]
        for k in range(Kb):
            x = cand[k]
            if train and float(torch.rand(())) < args.p_gt:
                lam = float(torch.rand(())) * args.bleed
                x = stems[b, k] + lam * (y[b] - stems[b, k])
            xs.append(x); ys.append(y[b]); ls.append(int(cls[b, k]))
    X, Y = torch.stack(xs), torch.stack(ys)
    if train and (args.aug_tilt > 0 or args.aug_lp > 0):
        # robustness to a mismatched world: random spectral tilt (dB at 4 kHz
        # relative to 1 kHz) and a random low-pass, applied to stem and mixture alike
        X, Y = tilt_aug(X, Y, args)
    return X, Y, torch.tensor(ls, device=dev)


def tilt_aug(X, Y, args):
    B, N = X.shape
    n = N
    f = torch.fft.rfftfreq(n, 1.0 / SR).to(X.device)
    tilt = (torch.rand(B, 1, device=X.device) * 2 - 1) * args.aug_tilt           # dB at 4 kHz vs 1 kHz
    g = 10 ** (tilt * torch.log2(f[None].clamp_min(50.0) / 1000.0) / 2.0 / 20.0)  # log-linear tilt per octave
    if args.aug_lp > 0:
        fc = 2000.0 + torch.rand(B, 1, device=X.device) * (8000.0 - 2000.0)
        g = g * torch.sigmoid((fc - f[None]) / 300.0) if float(torch.rand(())) < args.aug_lp else g
    Xf, Yf = torch.fft.rfft(X), torch.fft.rfft(Y)
    return torch.fft.irfft(Xf * g, n=n), torch.fft.irfft(Yf * g, n=n)


@torch.no_grad()
def evaluate(model, sep3, vds, dev, n=None, all_slots=False):
    model.eval()
    dl = torch.utils.data.DataLoader(vds, batch_size=8, shuffle=False, num_workers=4)
    slot_ok, slot_n, scene_ok, scene_n, byK = 0, 0, 0, 0, {2: [0, 0], 3: [0, 0]}
    seen = 0
    for y, stems, K, cls in dl:
        y, stems = y.to(dev), stems.to(dev)
        pred = sep3(y)[..., :SEG]
        post = posterior_t(y, pred)
        for b in range(y.shape[0]):
            Kb = int(K[b])
            if Kb < 2:
                continue
            en = pred[b].pow(2).mean(-1)
            live = list(range(MAXK)) if all_slots else sorted(torch.argsort(-en)[:Kb].tolist())
            classes = [int(cls[b, k]) for k in range(Kb)]
            perm = model.assign(post[b], y[b], live, classes)
            orac = pit_perm(post[b], stems[b, :Kb], Kb)
            same = all(perm[k] == orac[k] for k in range(Kb))
            scene_ok += int(same); scene_n += 1
            byK[Kb][0] += int(same); byK[Kb][1] += 1
            slot_ok += sum(int(perm[k] == orac[k]) for k in range(Kb)); slot_n += Kb
        seen += y.shape[0]
        if n and seen >= n:
            break
    model.train()
    return {"scene_agree": scene_ok / max(1, scene_n), "slot_agree": slot_ok / max(1, slot_n), "n_scenes": scene_n,
            "K2": byK[2][0] / max(1, byK[2][1]), "K3": byK[3][0] / max(1, byK[3][1])}


def train(args):
    torch.manual_seed(args.seed)
    dev = "cuda"
    sep3 = load_sep3(args.sep3, dev)
    model = SlotNamer(mix_in=args.mix_in).to(dev)
    print(f"slot namer params={sum(p.numel() for p in model.parameters())/1e6:.2f}M mix_in={args.mix_in}", flush=True)
    ds, vds = ClsScenes("train"), ClsScenes("val", n=args.n_val)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=6, drop_last=True, persistent_workers=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda st: min(1.0, (st + 1) / 200) * (0.03 + 0.97 * 0.5 * (1 + np.cos(np.pi * min(1.0, st / args.steps)))))
    step, t0, best, log = 0, time.time(), -1.0, []
    model.train()
    while step < args.steps:
        for y, stems, K, cls in dl:
            y, stems = y.to(dev), stems.to(dev)
            x, ym, lab = make_items(sep3, y, stems, K, cls, args, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x, ym if args.mix_in else None)
            loss = F.cross_entropy(logits.float(), lab)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step(); step += 1
            if step % 100 == 0:
                acc = float((logits.argmax(-1) == lab).float().mean())
                print(f"{step} loss={float(loss):.3f} batch acc {acc:.2f} ({(time.time()-t0)/100:.2f}s/it)", flush=True)
                t0 = time.time()
            if step % args.val_every == 0 or step >= args.steps:
                r = evaluate(model, sep3, vds, dev)
                print(f"  val scene agreement {r['scene_agree']:.3f} (K2 {r['K2']:.3f} K3 {r['K3']:.3f}) slot {r['slot_agree']:.3f} n {r['n_scenes']}", flush=True)
                log.append({"step": step, **r})
                state = {"model": model.state_dict(), "step": step, "args": vars(args), "val": r, "classes": CLASSES}
                torch.save(state, CK / f"cls{args.tag}_latest.pt")
                if r["scene_agree"] > best:
                    best = r["scene_agree"]
                    torch.save(state, CK / f"cls{args.tag}_best.pt")
                    print(f"  new best scene agreement {best:.3f} at step {step}", flush=True)
                (CK / f"cls{args.tag}_log.json").write_text(json.dumps(log, indent=1))
            if step >= args.steps:
                break
    print("TRAINING DONE cls", flush=True)


def load_namer(name, dev):
    st = torch.load(CK / name, map_location=dev)
    m = SlotNamer(mix_in=st["args"].get("mix_in", False)).to(dev)
    m.load_state_dict(st["model"])
    m.eval()
    return m, st.get("step")


def eval_cmd(args):
    dev = "cuda"
    model, step = load_namer(f"cls{args.tag}_best.pt", dev)
    sep3 = load_sep3(args.sep3, dev)
    r = evaluate(model, sep3, ClsScenes(args.split, n=args.n_val), dev, all_slots=args.all_slots)
    print(json.dumps({"split": args.split, "step": step, "all_slots": args.all_slots, **r}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval"])
    ap.add_argument("--split", default="val")
    ap.add_argument("--tag", default="")
    ap.add_argument("--sep3", default="sep3_d3_best.pt")
    ap.add_argument("--mix_in", action="store_true")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--val_every", type=int, default=500)
    ap.add_argument("--n_val", type=int, default=400)
    ap.add_argument("--p_gt", type=float, default=0.3, help="share of items that use the ground-truth stem plus bleed")
    ap.add_argument("--bleed", type=float, default=0.6)
    ap.add_argument("--aug_tilt", type=float, default=0.0, help="train: random spectral tilt range in dB at 4 kHz vs 1 kHz")
    ap.add_argument("--aug_lp", type=float, default=0.0, help="train: probability of a random low-pass between 2 and 8 kHz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all_slots", action="store_true", help="eval: assign over all three slots instead of the top-K by energy")
    a = ap.parse_args()
    {"train": train, "eval": eval_cmd}[a.cmd](a)
