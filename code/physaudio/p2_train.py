"""Phase C trainer: variational model of wet audio with swappable decoder.

    enc(y) -> z ~ q(z|y);   decoder:
      --decoder phi   : d = D_dry(z);  y_hat = Phi_torch(d, theta)   [ours]
      --decoder film  : y_hat = D_neu(z, FiLM(theta))               [twin]
      --decoder phires: y_hat = Phi(d, theta) + g(Phi(d,theta), theta)
                        with differentiable-audit invariance penalty [residual]

Identical encoder, latent, data, budget across variants; only where the
physics lives differs. Loss: multi-res STFT + beta*KL (+ audit penalty for
phires). Axes trained jointly with an axis token. theta is teacher-forced
in training, commanded at generation.
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

sys.path.insert(0, "/workspace/CF_PhysAudio/src")
sys.path.insert(0, "/workspace/audiocf_render")
from phi_torch import phi_velocity_t, phi_distance_t, RIRBank, SR  # noqa: E402

ROOT = Path("/workspace/CF_PhysAudio")
CORPUS = ROOT / "p2" / "corpus"


def set_corpus(name):
    """Point NPZSet at a different corpus dir (e.g. corpus_v2)."""
    global CORPUS
    CORPUS = ROOT / "p2" / name
SEG = 4 * SR
AXES = {"velocity": 0, "distance": 1, "reverb": 2}
THETA_NORM = {"velocity": (5.0, 40.0), "distance": (1.0, 32.0),
              "reverb": (0.1, 0.5)}


class NPZSet(torch.utils.data.Dataset):
    def __init__(self, split):
        self.files = sorted((CORPUS / split).glob("*.npz"))
        assert self.files, f"empty corpus split: {CORPUS / split}"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        z = np.load(self.files[i], allow_pickle=True)
        axis = str(z["axis"])
        lo, hi = THETA_NORM[axis]
        th = (float(z["theta"]) - lo) / (hi - lo)
        return (torch.tensor(z["y"][:SEG]), torch.tensor(z["d"][:SEG]),
                torch.tensor(th, dtype=torch.float32),
                torch.tensor(float(z["theta"])), AXES[axis])


def mrstft_loss(x, y):
    loss = 0.0
    for nfft in (512, 1024, 2048):
        w = torch.hann_window(nfft, device=x.device)
        X = torch.stft(x, nfft, nfft // 4, window=w, return_complex=True).abs()
        Y = torch.stft(y, nfft, nfft // 4, window=w, return_complex=True).abs()
        loss = loss + (X - Y).abs().mean() + \
            (torch.log(X + 1e-5) - torch.log(Y + 1e-5)).abs().mean()
    return loss / 3


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, 8, stride, 3)
        self.c2 = nn.Conv1d(cout, cout, 7, 1, 3)
        self.gn = nn.GroupNorm(8, cout)

    def forward(self, x):
        h = F.gelu(self.c1(x))
        return F.gelu(self.gn(self.c2(h)) + h)


class Encoder(nn.Module):
    """y (B,N) -> mu, logvar over latent sequence (B, C, N/1024)."""
    def __init__(self, width=1024, zdim=192):
        super().__init__()
        ch = [1, 64, 160, 320, 640, width]
        self.blocks = nn.ModuleList(
            [ConvBlock(ch[i], ch[i + 1], 4) for i in range(5)])
        self.head = nn.Conv1d(width, 2 * zdim, 1)

    def forward(self, y):
        h = y[:, None, :]
        for b in self.blocks:
            h = b(h)
        mu, lv = self.head(h).chunk(2, 1)
        return mu, lv.clamp(-8, 8)


class UpBlock(nn.Module):
    def __init__(self, cin, cout, film_dim=None):
        super().__init__()
        self.up = nn.ConvTranspose1d(cin, cout, 8, 4, 2)
        self.c = nn.Conv1d(cout, cout, 7, 1, 3)
        self.gn = nn.GroupNorm(8, cout)
        self.film = nn.Linear(film_dim, 2 * cout) if film_dim else None

    def forward(self, x, cond=None):
        h = F.gelu(self.up(x))
        h2 = self.gn(self.c(h))
        if self.film is not None and cond is not None:
            g, b = self.film(cond)[:, :, None].chunk(2, 1)
            h2 = h2 * (1 + g) + b
        return F.gelu(h2 + h)


class Decoder(nn.Module):
    """z (B,zdim,T) -> waveform (B,N). If film_dim, conditions every stage."""
    def __init__(self, zdim=192, width=1024, film_dim=None, mult=1.0):
        super().__init__()
        ch = [zdim] + [max(8, int(c * mult) // 8 * 8)
                       for c in (width, 640, 320, 160, 64)]
        self.blocks = nn.ModuleList(
            [UpBlock(ch[i], ch[i + 1], film_dim) for i in range(5)])
        self.out = nn.Conv1d(ch[-1], 1, 7, 1, 3)

    def forward(self, z, cond=None):
        h = z
        for b in self.blocks:
            h = b(h, cond)
        return torch.tanh(self.out(h))[:, 0, :]


class ResidualNet(nn.Module):
    """Post-Phi corrector conditioned on (theta, axis)."""
    def __init__(self, film_dim):
        super().__init__()
        self.enc = nn.ModuleList([ConvBlock(1, 96, 4), ConvBlock(96, 192, 4)])
        self.dec = nn.ModuleList([UpBlock(192, 96, film_dim),
                                  UpBlock(96, 48, film_dim)])
        self.out = nn.Conv1d(48, 1, 7, 1, 3)

    def forward(self, y, cond):
        h = y[:, None, :]
        for b in self.enc:
            h = b(h)
        for b in self.dec:
            h = b(h, cond)
        r = self.out(h)[:, 0, :]
        return r[:, :y.shape[-1]]


class Model(nn.Module):
    def __init__(self, decoder_kind, zdim=192):
        super().__init__()
        self.kind = decoder_kind
        self.enc = Encoder(zdim=zdim)
        self.cond_dim = 32
        self.embed = nn.Sequential(nn.Linear(1 + 3, 64), nn.GELU(),
                                   nn.Linear(64, self.cond_dim))
        film = self.cond_dim if decoder_kind in ("film", "film2x") else None
        if decoder_kind == "film2x":
            self.dec = Decoder(zdim=zdim, film_dim=film, mult=1.6)
        elif decoder_kind == "filmcat":
            self.dec = Decoder(zdim=zdim + self.cond_dim, film_dim=None)
        else:
            self.dec = Decoder(zdim=zdim, film_dim=film)
        self.rir = RIRBank() if decoder_kind in ("phi", "phires") else None
        self.res = ResidualNet(self.cond_dim) if decoder_kind == "phires" else None

    def cond(self, th_norm, axis_id):
        onehot = F.one_hot(axis_id, 3).float()
        return self.embed(torch.cat([th_norm[:, None], onehot], -1))

    def phi(self, d, theta, axis_id):
        outs = []
        for a, fn in ((0, phi_velocity_t), (1, phi_distance_t)):
            m = axis_id == a
            if m.any():
                outs.append((m, fn(d[m], theta[m])))
        m = axis_id == 2
        if m.any():
            outs.append((m, self.rir(d[m], theta[m])[:, :d.shape[-1]]))
        y = torch.zeros_like(d)
        for m, o in outs:
            y[m] = o[:, :d.shape[-1]].to(y.dtype)
        return y

    def forward(self, y, th_norm, theta, axis_id):
        mu, lv = self.enc(y)
        z = mu + torch.randn_like(mu) * (0.5 * lv).exp()
        kl = 0.5 * (mu ** 2 + lv.exp() - 1 - lv).mean()
        c = self.cond(th_norm, axis_id)
        if self.kind in ("film", "film2x"):
            y_hat = self.dec(z, c)
            return y_hat, None, kl, None
        if self.kind == "filmcat":
            zc = torch.cat([z, c[:, :, None].expand(-1, -1, z.shape[-1])], 1)
            y_hat = self.dec(zc)
            return y_hat, None, kl, None
        d_hat = self.dec(z)
        y_phi = self.phi(d_hat, theta, axis_id)
        peak = y_phi.abs().amax(-1, keepdim=True).clamp(min=1e-6)
        y_phi = y_phi / peak
        if self.kind == "phires":
            y_hat = y_phi + 0.1 * self.res(y_phi, c)
            return y_hat, y_phi, kl, d_hat
        return y_phi, None, kl, d_hat


def _env_db(x):
    e = F.avg_pool1d(x[:, None, :] ** 2, 2048, 1024)[:, 0, :]
    return 10 * torch.log10(e + 1e-8)


def dryness_penalty(d_hat, axis_id):
    """Differentiable form of the two dryness audits (R3 diagnostic):
    a dry static source has no seconds-scale level trajectory (motion
    audit; velocity items) and no decay tail beyond its event
    (reverberation audit; reverb items). Distance items unconstrained
    (absolute scale is Phi's)."""
    env = _env_db(d_hat)                      # (B, T) frames of 64 ms
    total = d_hat.new_zeros(())
    m_v = axis_id == 0
    if m_v.any():
        ev = env[m_v]
        sm = F.avg_pool1d(ev[:, None, :], 15, 1, padding=7)[:, 0, :]
        total = total + (sm - sm.mean(-1, keepdim=True)).pow(2).mean().sqrt()
    m_r = axis_id == 2
    if m_r.any():
        er = env[m_r]
        peak_val, peak_idx = er.max(-1)
        t = torch.arange(er.shape[-1], device=er.device)[None, :]
        mask = t >= (peak_idx[:, None] + 23)      # ~1.5 s after the peak
        tail = F.relu(er - (peak_val[:, None] - 35.0)) * mask
        total = total + tail.sum() / mask.sum().clamp(min=1)
    return total


def audit_penalty(y_hat, y_phi):
    """Differentiable invariance: residual must not change level trajectory
    or decay envelope (soft audits) beyond epsilon."""
    def env(x):
        e = F.avg_pool1d(x[:, None, :] ** 2, 2048, 1024)[:, 0, :]
        return 10 * torch.log10(e + 1e-8)
    return (env(y_hat) - env(y_phi)).abs().mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoder",
                    choices=["phi", "film", "phires", "phidry", "phiauditd",
                             "phienv", "film2x", "filmcat"],
                    required=True)
    ap.add_argument("--lam_dry", type=float, default=1.0)
    ap.add_argument("--lam_audit", type=float, default=0.05)
    ap.add_argument("--lam_env", type=float, default=0.05)
    ap.add_argument("--corpus", default="corpus",
                    help="corpus dir under p2/ (corpus | corpus_v2)")
    ap.add_argument("--tag", default="",
                    help="suffix for ckpt/log names (e.g. _v2)")
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--beta", type=float, default=1e-3)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(0)
    dev = "cuda"
    set_corpus(args.corpus)
    skind = ("phi" if args.decoder in ("phidry", "phiauditd", "phienv")
             else args.decoder)  # film2x/filmcat are structural kinds
    model = Model(skind).to(dev)
    nparam = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"decoder={args.decoder} params={nparam/1e6:.1f}M", flush=True)
    ds = NPZSet("train")
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True,
                                     num_workers=6, drop_last=True,
                                     persistent_workers=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    ckdir = ROOT / "p2" / "ckpt"
    ckdir.mkdir(parents=True, exist_ok=True)
    step, t0, log = 0, time.time(), []
    name = args.decoder + args.tag
    ckp = ckdir / f"{name}_latest.pt"
    if args.resume and ckp.exists():
        state = torch.load(ckp, map_location=dev)
        model.load_state_dict(state["model"])
        step = state["step"]
        for _ in range(step):
            sched.step()
        lp = ckdir / f"{name}_log.json"
        if lp.exists():
            log = [e for e in json.loads(lp.read_text()) if e["step"] <= step]
        print(f"resumed {name} from step {step}", flush=True)
    if step >= args.steps:
        print("TRAINING DONE", args.decoder)
        return
    while step < args.steps:
        for y, d, thn, th, ax in dl:
            y, thn, th, ax = (y.to(dev), thn.to(dev), th.to(dev), ax.to(dev))
            y_hat, y_phi, kl, d_hat = model(y, thn, th, ax)
            L = min(y_hat.shape[-1], y.shape[-1])
            loss = mrstft_loss(y_hat[:, :L], y[:, :L]) + args.beta * kl
            dcomp = 0.0
            if model.kind == "phires" and y_phi is not None:
                loss = loss + 0.5 * audit_penalty(y_hat[:, :L], y_phi[:, :L])
            if args.decoder == "phidry":
                d = d.to(dev)
                Ld = min(d_hat.shape[-1], d.shape[-1])
                pk_h = d_hat[:, :Ld].abs().amax(-1, keepdim=True).clamp(min=1e-6)
                pk_t = d[:, :Ld].abs().amax(-1, keepdim=True).clamp(min=1e-6)
                dpen = mrstft_loss(d_hat[:, :Ld] / pk_h, d[:, :Ld] / pk_t)
                loss = loss + args.lam_dry * dpen
                dcomp = float(dpen)
            elif args.decoder == "phiauditd":
                dpen = dryness_penalty(d_hat, ax)
                loss = loss + args.lam_audit * dpen
                dcomp = float(dpen)
            elif args.decoder == "phienv":
                # A': supervise the low-dimensional observable itself --
                # the dB envelope of the true dry source (scale removed by
                # per-sample peak norm). A real target, not a threshold
                # penalty: no headroom for surrogate gaming, and low-dim
                # enough to transfer across sources.
                d = d.to(dev)
                Ld = min(d_hat.shape[-1], d.shape[-1])
                pk_h = d_hat[:, :Ld].abs().amax(-1, keepdim=True).clamp(min=1e-6)
                pk_t = d[:, :Ld].abs().amax(-1, keepdim=True).clamp(min=1e-6)
                dpen = (_env_db(d_hat[:, :Ld] / pk_h)
                        - _env_db(d[:, :Ld] / pk_t)).abs().mean()
                loss = loss + args.lam_env * dpen
                dcomp = float(dpen)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            step += 1
            if step % 200 == 0:
                log.append({"step": step, "loss": float(loss),
                            "kl": float(kl), "dry": dcomp})
                print(f"{step} loss={float(loss):.4f} "
                      f"({(time.time()-t0)/200:.2f}s/it)", flush=True)
                t0 = time.time()
            if step % 1000 == 0:
                (ckdir / f"{name}_log.json").write_text(
                    json.dumps(log))
            if step % 10000 == 0 or step == args.steps:
                torch.save({"model": model.state_dict(), "step": step,
                            "args": vars(args)},
                           ckdir / f"{name}_latest.pt")
            if step >= args.steps:
                break
    print("TRAINING DONE", args.decoder)


if __name__ == "__main__":
    main()
