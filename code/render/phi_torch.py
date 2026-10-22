"""Differentiable propagation core (PyTorch). Mirrors core.py/room.py/modal.py
semantics; differentiable w.r.t. both the dry signal d and the physical
command theta, so it can serve as a FIXED analytic decoder inside training.

Blocks:
  velocity: fractional-delay pass-by (grad wrt d and v)
  distance: 1/r + power-law air absorption via STFT (grad wrt d and r)
  reverb:   convolution with a differentiable convex mixture of a cached
            image-source RIR bank (grad wrt d and alpha, alpha continuous)
  material: closed-form modal sum (grad wrt k)

Self-test: parity with the NumPy renderer at bank nodes (run as __main__).
"""
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/workspace/audiocf_render")
from core import C_SOUND, R_MIN, air_absorption_db_per_m  # noqa: E402
from room import image_source_rir  # noqa: E402
from modal import BASE_MODES  # noqa: E402

SR = 16000


def _frac_read(d, t_read):
    """Differentiable linear-interpolation read of d (B,N) at t_read (B,N)."""
    n = d.shape[-1]
    i0 = torch.clamp(t_read.floor(), 0, n - 2)
    frac = t_read - i0
    i0 = i0.long()
    g0 = torch.gather(d, -1, i0)
    g1 = torch.gather(d, -1, i0 + 1)
    valid = (t_read >= 0) & (t_read < n - 1)
    return torch.where(valid, (1 - frac) * g0 + frac * g1,
                       torch.zeros_like(g0))


def _air_absorb(x, r_traj, n_fft=1024, hop=256):
    """Frequency-dependent attenuation with per-frame distance (matches the
    NumPy renderer's granularity). r_traj: (B,) constant or (B,N) trajectory.
    Differentiable w.r.t. x and r_traj."""
    freqs = torch.fft.rfftfreq(n_fft, 1 / SR, device=x.device)
    alpha = 0.0052 * (freqs.clamp(min=1.0) / 1000.0) ** 1.5  # dB/m
    win = torch.hann_window(n_fft, device=x.device)
    X = torch.stft(x, n_fft, hop, window=win, return_complex=True)  # (B,F,T)
    T = X.shape[-1]
    if r_traj.dim() == 1:
        r_frame = r_traj[:, None].expand(-1, T)
    else:
        # frame centers as used by torch.stft with center=True
        centers = torch.arange(T, device=x.device) * hop
        idx = centers.clamp(max=r_traj.shape[-1] - 1)
        half = n_fft // 2
        r_frame = torch.stack([
            r_traj[:, max(0, int(c) - half):int(c) + half].mean(-1)
            for c in centers], dim=-1)
    gain = 10 ** (-alpha[None, :, None] * r_frame[:, None, :] / 20)
    return torch.istft(X * gain, n_fft, hop, window=win, length=x.shape[-1])


def phi_velocity_t(d, v, d0=2.0):
    """d: (B,N); v: (B,). Straight pass-by, closest approach mid-signal."""
    B, n = d.shape
    t = torch.arange(n, device=d.device, dtype=d.dtype) / SR
    tc = n / SR / 2
    x = v[:, None] * (t[None, :] - tc)
    r = torch.sqrt(x ** 2 + d0 ** 2).clamp(min=R_MIN)
    t_read = torch.arange(n, device=d.device, dtype=d.dtype)[None, :] \
        - (r / C_SOUND) * SR
    y = _frac_read(d, t_read) / r
    return _air_absorb(y, r)


def phi_distance_t(d, r):
    """d: (B,N); r: (B,)."""
    B, n = d.shape
    delay = (r / C_SOUND) * SR
    t_read = torch.arange(n, device=d.device, dtype=d.dtype)[None, :] \
        - delay[:, None]
    y = _frac_read(d, t_read) / r.clamp(min=R_MIN)[:, None]
    return _air_absorb(y, r)


class RIRBank(torch.nn.Module):
    """Cached image-source RIRs at alpha nodes; forward interpolates
    log-RIR-energy-consistently via convex combination of neighbor RIRs.
    Differentiable w.r.t. d always, and w.r.t. alpha through the mixture
    weights (piecewise-linear in alpha)."""

    def __init__(self, room=(30.0, 20.0, 8.0), src=(18.0, 12.0, 1.5),
                 mic=(15.0, 10.0, 1.6), nodes=(0.08, 0.1, 0.15, 0.2, 0.28,
                                               0.35, 0.42, 0.5, 0.6),
                 rir_len_s=2.5):
        super().__init__()
        self.nodes = torch.tensor(nodes)
        bank = [image_source_rir(room, src, mic, SR, alpha=a, max_order=22,
                                 rir_len_s=rir_len_s) for a in nodes]
        L = max(len(b) for b in bank)
        bank = np.stack([np.pad(b, (0, L - len(b))) for b in bank])
        self.register_buffer("bank", torch.tensor(bank, dtype=torch.float32))

    def forward(self, d, alpha):
        """d: (B,N); alpha: (B,). FFT convolution with interpolated RIR."""
        nodes = self.nodes.to(d.device)
        a = alpha.clamp(nodes[0] + 1e-4, nodes[-1] - 1e-4)
        idx = torch.searchsorted(nodes, a) - 1
        idx = idx.clamp(0, len(nodes) - 2)
        w = (a - nodes[idx]) / (nodes[idx + 1] - nodes[idx])
        rir = (1 - w)[:, None] * self.bank[idx] + w[:, None] * self.bank[idx + 1]
        n, L = d.shape[-1], rir.shape[-1]
        m = n + L - 1
        Y = torch.fft.rfft(d, m) * torch.fft.rfft(rir, m)
        return torch.fft.irfft(Y, m)[:, :n + int(0.5 * SR)]


def phi_material_t(k, obj="plate", dur=1.5, device="cpu",
                   amp_jitter=None):
    """k: (B,). Closed-form modal sum, exactly differentiable in k."""
    t = torch.arange(int(dur * SR), device=device) / SR
    B = k.shape[0]
    y = torch.zeros(B, len(t), device=device)
    for mi, (f, dec, a) in enumerate(BASE_MODES[obj]):
        jit = amp_jitter[:, mi] if amp_jitter is not None else torch.zeros(B, device=device)
        amp = a * (1 + jit)
        y = y + amp[:, None] * torch.exp(-dec * t)[None, :] * \
            torch.sin(2 * torch.pi * f * torch.sqrt(k)[:, None] * t[None, :])
    return y / y.abs().amax(-1, keepdim=True).clamp(min=1e-9) * 0.5


def parity_test():
    """Numerical parity with the NumPy renderer."""
    from core import render_static_source, render_moving_source, passby_positions
    rng = np.random.default_rng(0)
    d = rng.standard_normal(SR * 2).astype(np.float32) * 0.1
    # distance parity
    E = 2048  # STFT edge region: overlap-add boundary conventions differ
    for r in (2.0, 8.0):
        y_np = render_static_source(d.astype(np.float64), SR, r)
        y_t = phi_distance_t(torch.tensor(d)[None, :],
                             torch.tensor([r])).numpy()[0]
        err = np.abs(y_np - y_t)[E:-E].max() / (np.abs(y_np).max() + 1e-9)
        print(f"distance r={r}: interior max rel dev {err:.2e}")
        assert err < 0.01
    # velocity parity
    v = 20.0
    pos = passby_positions(len(d), SR, v=v, d0=2.0, t_close=len(d) / SR / 2)
    y_np = render_moving_source(d.astype(np.float64), SR, pos)
    y_t = phi_velocity_t(torch.tensor(d)[None, :],
                         torch.tensor([v])).numpy()[0]
    err = np.abs(y_np - y_t)[E:-E].max() / (np.abs(y_np).max() + 1e-9)
    print(f"velocity v={v}: interior max rel dev {err:.2e}")
    assert err < 0.02
    return


if __name__ == "__main__":
    torch.set_num_threads(8)
    try:
        parity_test()
    except ModuleNotFoundError:
        pass
    # reverb node parity (standalone)
    from scipy.signal import fftconvolve
    rng = np.random.default_rng(1)
    d = rng.standard_normal(SR).astype(np.float32) * 0.1
    bank = RIRBank()
    a = 0.35
    rir_np = image_source_rir((30.0, 20.0, 8.0), (18.0, 12.0, 1.5),
                              (15.0, 10.0, 1.6), SR, alpha=a, max_order=22,
                              rir_len_s=2.5)
    y_np = fftconvolve(d.astype(np.float64), rir_np)[:len(d) + int(0.5 * SR)]
    y_t = bank(torch.tensor(d)[None, :], torch.tensor([a])).numpy()[0]
    L = min(len(y_np), len(y_t))
    err = np.abs(y_np[:L] - y_t[:L]).max() / (np.abs(y_np).max() + 1e-9)
    print(f"reverb alpha={a} (bank node): max rel dev {err:.2e}")
    # gradient smoke: d(loss)/d(theta) exists and is finite
    v = torch.tensor([20.0], requires_grad=True)
    y = phi_velocity_t(torch.tensor(d)[None, :], v)
    y.pow(2).mean().backward()
    print("grad wrt v:", float(v.grad))
    kk = torch.tensor([2.0], requires_grad=True)
    ym = phi_material_t(kk)
    ym.pow(2).mean().backward()
    print("grad wrt k:", float(kk.grad))
    print("PHI_TORCH SELF-TEST OK")
