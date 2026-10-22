"""PWG: physics-warped Gaussian scene model (protocol A7).

Dry sources are locally-stationary Gaussian processes on the STFT
grid, D_k(f,t) ~ CN(0, W_k(f) c_k(t)); the physics acts as PSD-domain
operators derived EXACTLY from phi_torch's waveform operators:

  passby(v):  gamma(t) = 1 - r'(t)/c   (instantaneous Doppler factor)
              S_wet(f,t) = c(t) * W(f/gamma(t))/gamma(t)
                           * Aair(f, r(t))^2 / r(t)^2
  static(r):  delay drops from the PSD; 1/r^2 and Aair(f,r)^2 fold
              into (W, c)  ->  the E3 gain confound as a theorem.
  room(a):    S -> sum_tau H_a(f,tau) S(:, t-tau),  H_a = |STFT(rir_a)|^2
              (energy kernel of the image-source RIR bank).

The dry waveforms are marginalised exactly: Y(f,t) ~
CN(0, sum_k S_k + sig2), giving the Whittle log-likelihood
  ll = sum_{f,t} [ -log(pi V) - |Y|^2 / V ].
theta is identified by this marginal likelihood; (W, c) fit by IS-NMF
multiplicative updates (= exact M-steps of EM); env/alpha by
likelihood model selection.  Posterior stems are the model's Wiener
means m_k Y -- the A5 projection, now derived rather than bolted on.
"""
import sys

import numpy as np
import torch

sys.path.insert(0, "/workspace/audiocf_render")
sys.path.insert(0, "/workspace/CF_ASG/src")
from core import C_SOUND, R_MIN  # noqa: E402
from phi_torch import RIRBank  # noqa: E402

SR = 16000
NFFT = 1024
HOP = 256
F = NFFT // 2 + 1
EPS = 1e-10
ALPHA_NODES = (0.08, 0.1, 0.15, 0.2, 0.28, 0.35, 0.42, 0.5, 0.6)


def stft_pow(x, dev="cpu"):
    """|STFT|^2 (F,T) and the complex STFT."""
    t = torch.as_tensor(x, dtype=torch.float32, device=dev)
    w = torch.hann_window(NFFT, device=dev)
    Z = torch.stft(t, NFFT, HOP, window=w, return_complex=True)
    return (Z.abs() ** 2).clamp_min(EPS), Z


def passby_geometry(T, v, d0=2.0):
    """Per-frame r(t), gamma(t) for frame centres (torch.stft center=True
    puts frame t at sample t*HOP)."""
    tt = torch.arange(T, dtype=torch.float64) * HOP / SR
    tc = (T - 1) * HOP / SR / 2
    x = v * (tt - tc)
    r = torch.sqrt(x ** 2 + d0 ** 2).clamp(min=R_MIN)
    rdot = v * x / r
    gamma = (1.0 - rdot / C_SOUND).clamp(0.5, 1.5)
    return r.float(), gamma.float()


_FREQS = torch.fft.rfftfreq(NFFT, 1 / SR)
_AIR_DB = 0.0052 * (_FREQS.clamp(min=1.0) / 1000.0) ** 1.5  # dB/m


def air_pow(r, dev="cpu"):
    """Air-absorption POWER gain (F,T) for per-frame distance r (T,)."""
    return 10 ** (-_AIR_DB.to(dev)[:, None] * r[None, :].to(dev) / 10)


def warp_apply(W, gamma, dev="cpu"):
    """S(f,t) = W(f/gamma_t)/gamma_t via linear interpolation on the
    frequency axis.  W: (F,), gamma: (T,) -> (F,T)."""
    f = torch.arange(F, device=dev, dtype=torch.float32)
    q = f[:, None] / gamma[None, :].to(dev)          # dry-freq coordinate
    qc = q.clamp(0, F - 1 - 1e-4)      # edge-extend beyond Nyquist
    i0 = qc.floor().clamp(0, F - 2)
    frac = qc - i0
    i0 = i0.long()
    Wv = W.to(dev)
    out = (1 - frac) * Wv[i0] + frac * Wv[i0 + 1]
    return out / gamma[None, :].to(dev)


def warp_transpose(S, gamma, dev="cpu"):
    """Adjoint of warp_apply w.r.t. W (for multiplicative updates)."""
    f = torch.arange(F, device=dev, dtype=torch.float32)
    q = f[:, None] / gamma[None, :].to(dev)
    qc = q.clamp(0, F - 1 - 1e-4)
    i0 = qc.floor().clamp(0, F - 2)
    frac = qc - i0
    i0 = i0.long()
    Sv = S / gamma[None, :].to(dev)
    out = torch.zeros(F, device=dev)
    out.scatter_add_(0, i0.reshape(-1), ((1 - frac) * Sv).reshape(-1))
    out.scatter_add_(0, (i0 + 1).reshape(-1), (frac * Sv).reshape(-1))
    return out


class RoomKernel:
    """Frequency-resolved RIR energy kernels H_a(f,tau) at bank nodes."""

    def __init__(self, dev="cpu", max_tau=None):
        bank = RIRBank()
        self.dev = dev
        Hs = []
        w = torch.hann_window(NFFT)
        for i in range(bank.bank.shape[0]):
            rir = bank.bank[i]
            Z = torch.stft(rir, NFFT, HOP, window=w, return_complex=True)
            H = (Z.abs() ** 2)
            Hs.append(H)
        L = max(h.shape[-1] for h in Hs)
        Hs = [torch.nn.functional.pad(h, (0, L - h.shape[-1])) for h in Hs]
        H = torch.stack(Hs).to(dev)                 # (A, F, TAU)
        if max_tau is not None:
            H = H[..., :max_tau]
        # window energy normalisation: sum_tau H approximates the
        # frequency response energy of the RIR at hop resolution
        self.H = H / (w.pow(2).sum() / HOP)
        self.nodes = list(ALPHA_NODES)

    def kernel(self, alpha):
        n = torch.tensor(self.nodes)
        a = float(np.clip(alpha, self.nodes[0] + 1e-4, self.nodes[-1] - 1e-4))
        i = int(torch.searchsorted(n, torch.tensor(a))) - 1
        i = max(0, min(i, len(self.nodes) - 2))
        w = (a - self.nodes[i]) / (self.nodes[i + 1] - self.nodes[i])
        return (1 - w) * self.H[i] + w * self.H[i + 1]

    def apply(self, S, alpha):
        """S (F,T) -> sum_tau H(f,tau) S(f,t-tau), causal."""
        H = self.kernel(alpha)                       # (F,TAU)
        TAU = H.shape[-1]
        T = S.shape[-1]
        Sp = torch.nn.functional.pad(S, (TAU - 1, 0))
        # conv along time with per-frequency kernels
        out = torch.nn.functional.conv1d(
            Sp[None, :, :], H.flip(-1)[:, None, :], groups=F)[0]
        return out[:, :T]

    def transpose(self, S, alpha):
        H = self.kernel(alpha)
        TAU = H.shape[-1]
        T = S.shape[-1]
        Sp = torch.nn.functional.pad(S, (0, TAU - 1))
        out = torch.nn.functional.conv1d(
            Sp[None, :, :], H[:, None, :], groups=F)[0]
        return out[:, :T]


class Slot:
    """One source's PSD parameters and physics tags."""

    def __init__(self, W, c, kind="static", v=None):
        self.W, self.c, self.kind, self.v = W, c, kind, v

    def wet_pre_room(self, T, dev):
        if self.kind == "passby":
            r, gamma = passby_geometry(T, float(self.v))
            B = warp_apply(self.W, gamma, dev) * air_pow(r, dev) \
                / (r[None, :].to(dev) ** 2)
            return self.c[None, :].to(dev) * B
        # static: gain/absorption folded into (W, c) -- confound theorem
        return self.W[:, None].to(dev) * self.c[None, :].to(dev)


def scene_psd(slots, T, alpha, env, rk, dev):
    S = []
    for s in slots:
        Sp = s.wet_pre_room(T, dev)
        if env > 0.5:
            Sp = rk.apply(Sp, alpha)
        S.append(Sp)
    return S


def whittle_ll(P, V):
    """Mean Whittle log-likelihood (up to const) of |Y|^2=P under PSD V."""
    return float((-V.clamp_min(EPS).log() - P / V.clamp_min(EPS)).mean())


def em_fit(P, slots, alpha, env, rk, sig2, dev, iters=8, spars=0.0,
           omega=None):
    """Multiplicative IS-NMF updates for (W_k, c_k) with physics warps
    and room kernel fixed.  P = |Y|^2 (F,T).  sig2=None -> data-scaled
    noise floor (1e-4 of mean power).  spars: MAP-EM exponential prior
    on the envelopes c (event-sparse dries -- restores env/alpha
    identifiability: reverb tails must be explained by the kernel,
    not mimicked by c).  spars is specified relative to mean power."""
    T = P.shape[-1]
    if sig2 is None:
        sig2 = 1e-4 * float(P.mean())
    lam = spars * float(P.mean())
    for _ in range(iters):
        Ss = scene_psd(slots, T, alpha, env, rk, dev)
        V = sum(Ss) + sig2
        R1 = P / V.pow(2).clamp_min(EPS)             # numerator carrier
        R2 = 1.0 / V.clamp_min(EPS)                  # denominator carrier
        if omega is not None:
            R1 = R1 * omega
            R2 = R2 * omega
        for s in slots:
            if env > 0.5:
                G1 = rk.transpose(R1, alpha)
                G2 = rk.transpose(R2, alpha)
            else:
                G1, G2 = R1, R2
            if s.kind == "passby":
                r, gamma = passby_geometry(T, float(s.v))
                A2 = air_pow(r, dev) / (r[None, :].to(dev) ** 2)
                base = warp_apply(s.W, gamma, dev) * A2   # dV/dc
                num_c = (G1 * base).sum(0)
                den_c = ((G2 * base).sum(0) + lam).clamp_min(EPS)
                s.c = (s.c.to(dev) * num_c / den_c).clamp_min(EPS).cpu()
                CA = s.c[None, :].to(dev) * A2            # dV/d(warp W)
                num_W = warp_transpose(G1 * CA, gamma, dev)
                den_W = warp_transpose(G2 * CA, gamma, dev).clamp_min(EPS)
                s.W = (s.W.to(dev) * num_W / den_W).clamp_min(EPS).cpu()
            else:
                Wd = s.W[:, None].to(dev)
                num_c = (G1 * Wd).sum(0)
                den_c = ((G2 * Wd).sum(0) + lam).clamp_min(EPS)
                s.c = (s.c.to(dev) * num_c / den_c).clamp_min(EPS).cpu()
                cd = s.c[None, :].to(dev)
                num_W = (G1 * cd).sum(1)
                den_W = (G2 * cd).sum(1).clamp_min(EPS)
                s.W = (s.W.to(dev) * num_W / den_W).clamp_min(EPS).cpu()
    Ss = scene_psd(slots, T, alpha, env, rk, dev)
    V = sum(Ss) + sig2
    if omega is not None:
        ll = float(((-V.clamp_min(EPS).log() - P / V.clamp_min(EPS))
                    * omega).sum() / omega.sum().clamp_min(EPS))
    else:
        ll = whittle_ll(P, V)
    return ll, slots


def init_slot_from_stem(stem, kind, v0, dev):
    """Rank-1 init of (W, c) from a network stem estimate, unwarped to
    the dry frame for pass-by slots."""
    P, _ = stft_pow(stem, dev)
    T = P.shape[-1]
    if kind == "passby":
        r, gamma = passby_geometry(T, float(v0))
        P = P / (air_pow(r, dev) / (r[None, :].to(dev) ** 2)).clamp_min(EPS)
        # unwarp: dry-freq q = f/gamma -> sample wet at f = q*gamma
        f = torch.arange(F, device=dev, dtype=torch.float32)
        q = f[:, None] * gamma[None, :].to(dev)
        i0 = q.floor().clamp(0, F - 2)
        frac = q - i0
        i0 = i0.long()
        P = (1 - frac) * P.gather(0, i0) + frac * P.gather(0, i0 + 1)
    W = P.mean(1)
    c = (P * W[:, None]).sum(0) / W.pow(2).sum().clamp_min(EPS)
    W = (W / W.mean().clamp_min(EPS)).clamp_min(EPS)
    return Slot(W.cpu(), c.clamp_min(EPS).cpu(), kind, v0)


def unwarp_mean_W(P, v, dev):
    """Moment init: unwarp the observed power to the dry frame at
    candidate v and average over time.  At the true v the per-frame
    spectra align (sharp W); at a wrong v they smear."""
    T = P.shape[-1]
    r, gamma = passby_geometry(T, float(v))
    Pn = P / (air_pow(r, dev) / (r[None, :].to(dev) ** 2)).clamp_min(EPS)
    f = torch.arange(F, device=dev, dtype=torch.float32)
    q = (f[:, None] * gamma[None, :].to(dev)).clamp(0, F - 1 - 1e-4)
    i0 = q.floor().clamp(0, F - 2)
    frac = q - i0
    i0 = i0.long()
    Pd = (1 - frac) * Pn.gather(0, i0) + frac * Pn.gather(0, i0 + 1)
    W = Pd.mean(1).clamp_min(EPS)
    return (W / W.mean().clamp_min(EPS)).cpu()


def fit_v(P, slots, k, v_grid, alpha, env, rk, sig2, dev, iters=4):
    if sig2 is None:
        sig2 = 1e-4 * float(P.mean())
    """Grid search on the marginal likelihood over slot k's velocity,
    refitting (W_k, c_k) briefly at each candidate."""
    import copy
    best = (-np.inf, None, None)
    for v in v_grid:
        trial = copy.deepcopy(slots)
        trial[k].v = float(v)
        ll, trial = em_fit(P, trial, alpha, env, rk, sig2, dev, iters=iters)
        if ll > best[0]:
            best = (ll, float(v), trial)
    return best


def pwg_theta(stem, dev="cpu", grid=None, iters=12):
    """Deployment config (A7): single-source Whittle fit on a
    posterior-isolated stem -- moment-initialised W, full grid on the
    marginal likelihood, parabolic refinement.  Returns v_hat."""
    if grid is None:
        grid = np.linspace(5, 40, 29)
    P, _ = stft_pow(stem, dev)
    T = P.shape[-1]
    rk = None
    lls = []
    for v in grid:
        W0 = unwarp_mean_W(P, float(v), dev)
        s = Slot(W0, torch.ones(T), "passby", float(v))
        ll, _ = em_fit(P, [s], 0.0, 0.0, rk, None, dev, iters=iters)
        lls.append(ll)
    lls = np.array(lls)
    i = int(np.argmax(lls))
    v = float(grid[i])
    if 0 < i < len(grid) - 1:
        c0, c1, c2 = lls[i - 1], lls[i], lls[i + 1]
        den = c0 - 2 * c1 + c2
        if den < -1e-12:
            v += 0.5 * (c0 - c2) / den * (grid[1] - grid[0])
    return float(np.clip(v, grid[0], grid[-1]))
