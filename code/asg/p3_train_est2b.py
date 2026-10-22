"""A10 est2b: T-F velocity estimator, final form.
- input: 2-channel log-spectrogram (slot stem + mixture context)
- v as 72-bin classification with Gaussian-smoothed targets,
  expectation decoding (multimodal-ambiguity-friendly)
- training inputs drawn 50/50 from PRECOMPUTED ownership-posterior
  stems (deployment domain, p3/po_train) and GT stems with bleed
  augmentation; random time-segment dropout on both.
-> p3/ckpt/est2b_latest.pt
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/workspace/CF_ASG/src")
from p3_train import SceneSet, SEG  # noqa

PO = Path("/workspace/CF_ASG/p3/po_train")
CK = Path("/workspace/CF_ASG/p3/ckpt")
NBINS = 72


class Est2B(nn.Module):
    def __init__(self, ch=64):
        super().__init__()

        def blk(ci, co):
            return nn.Sequential(nn.Conv2d(ci, co, 3, 2, 1),
                                 nn.GroupNorm(4, co), nn.SiLU())
        self.net = nn.Sequential(blk(2, ch), blk(ch, 2 * ch),
                                 blk(2 * ch, 4 * ch), blk(4 * ch, 4 * ch),
                                 blk(4 * ch, 4 * ch))
        self.head = nn.Linear(4 * ch, 2 + NBINS)

    def forward(self, s, y):
        w = torch.hann_window(1024, device=s.device)
        X = torch.stack([s, y], 1).reshape(-1, s.shape[-1])
        Z = torch.stft(X, 1024, 256, window=w, return_complex=True)
        X = Z.abs().pow(2).clamp_min(1e-10).log() \
            .reshape(s.shape[0], 2, 513, -1)
        X = X - X.mean(dim=(2, 3), keepdim=True)
        h = self.net(X).mean(dim=(2, 3))
        o = self.head(h)
        return o[:, :2], o[:, 2:]

    @staticmethod
    def decode_v(logits):
        p = torch.softmax(logits, -1)
        centers = torch.linspace(0, 1, NBINS, device=logits.device)
        return (p * centers).sum(-1)          # v_norm expectation


class SlotSet(torch.utils.data.Dataset):
    def __init__(self):
        self.ds = SceneSet("train")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        y, stems, dries, exist, traj, thn, alpha, env, K = self.ds[i]
        K = int(K)
        rng = np.random.default_rng()
        k = int(rng.integers(K))
        yn = y.numpy()
        if rng.random() < 0.5 and (PO / f"{i:06d}.npz").exists():
            po = np.load(PO / f"{i:06d}.npz")["po"]
            s = po[k].astype(np.float32)
        else:
            lam = rng.random() * 0.6
            s = (stems[k].numpy()
                 + lam * (yn[:SEG] - stems[k].numpy())).astype(np.float32)
        # random time-segment dropout (x2)
        for _ in range(2):
            if rng.random() < 0.5:
                wlen = int(0.1 * SEG)
                st = int(rng.integers(SEG - wlen))
                s[st:st + wlen] = 0.0
        return (torch.tensor(s), y[:SEG].float(), traj[k],
                thn[k].float())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(0)
    dev = "cuda"
    model = Est2B().to(dev)
    print("params: %.1fM" % (sum(p.numel() for p in model.parameters())
                             / 1e6), flush=True)
    dl = torch.utils.data.DataLoader(SlotSet(), batch_size=args.batch,
                                     shuffle=True, num_workers=6,
                                     drop_last=True,
                                     persistent_workers=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    centers = torch.linspace(0, 1, NBINS, device=dev)
    step, t0, log = 0, time.time(), []
    while step < args.steps:
        for s, y, tr, tn in dl:
            s, y = s.to(dev), y.to(dev)
            tr, tn = tr.to(dev), tn.to(dev)
            ptr, pv = model(s, y)
            l_tr = F.cross_entropy(ptr, tr)
            mv = (tr == 0)
            if mv.any():
                tgt = torch.exp(-0.5 * ((centers[None, :] - tn[mv, None])
                                        / (1.5 / NBINS)) ** 2)
                tgt = tgt / tgt.sum(-1, keepdim=True)
                l_v = -(tgt * torch.log_softmax(pv[mv], -1)).sum(-1).mean()
                vhat = Est2B.decode_v(pv[mv])
                l_reg = torch.abs(vhat - tn[mv]).mean()
            else:
                l_v = l_reg = l_tr * 0
            loss = 0.3 * l_tr + l_v + 0.5 * l_reg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            step += 1
            if step % 200 == 0:
                parts = {"tr": float(l_tr), "v": float(l_v),
                         "reg": float(l_reg)}
                log.append({"step": step, **parts})
                print(f"{step} loss={float(loss):.4f} "
                      f"{json.dumps(parts)} "
                      f"({(time.time()-t0)/200:.2f}s/it)", flush=True)
                t0 = time.time()
            if step % 2000 == 0 or step >= args.steps:
                torch.save({"model": model.state_dict(), "step": step},
                           CK / "est2b_latest.pt")
                (CK / "est2b_log.json").write_text(json.dumps(log))
            if step >= args.steps:
                break
    print("TRAINING DONE est2b")


if __name__ == "__main__":
    main()
