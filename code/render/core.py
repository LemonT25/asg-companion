"""audiocf_render v0 — deterministic analytic propagation core.

Direct-path renderer for a moving point source: fractional-delay Doppler,
1/r spreading, frequency-dependent air absorption (power-law approximation
of ISO 9613-1 at 20C/50%RH; exact ISO tables are a v1 upgrade, documented).
No learned components, no randomness. All angles in meters/seconds.

Reception-time approximation for the delay (t_emit = t - r(t)/c) — valid for
v << c; at v = 40 m/s the induced frequency error is ~1%.
"""
import numpy as np

C_SOUND = 343.0
R_MIN = 0.1  # clamp to avoid 1/r blowup


def air_absorption_db_per_m(freqs_hz):
    """Power-law fit: ~0.005 dB/m @1kHz, ~0.04 @4kHz, ~0.12 @8kHz."""
    f = np.maximum(freqs_hz, 1.0) / 1000.0
    return 0.0052 * f ** 1.5


def passby_positions(n, sr, v, d0, t_close, mic=(0.0, 0.0, 0.0)):
    """Straight-line pass-by along +x at speed v, closest distance d0 (y),
    closest approach at t_close. Returns (n,3) source positions."""
    t = np.arange(n) / sr
    pos = np.zeros((n, 3))
    pos[:, 0] = v * (t - t_close)
    pos[:, 1] = d0
    return pos + np.asarray(mic)[None, :]


def render_moving_source(dry, sr, positions, mic=(0.0, 0.0, 0.0),
                         air_absorb=True, spreading=True):
    """Render dry (n,) emitted signal from per-sample positions (n,3) to a
    mic. Returns pressure signal at the mic, same length."""
    dry = np.asarray(dry, np.float64)
    n = len(dry)
    r = np.linalg.norm(np.asarray(positions, np.float64)
                       - np.asarray(mic, np.float64)[None, :], axis=1)
    r = np.maximum(r, R_MIN)
    # fractional-delay read (linear interpolation), reception-time approx
    t_read = np.arange(n) - (r / C_SOUND) * sr
    i0 = np.floor(t_read).astype(int)
    frac = t_read - i0
    valid = (i0 >= 0) & (i0 < n - 1)
    out = np.zeros(n)
    out[valid] = (1 - frac[valid]) * dry[i0[valid]] \
        + frac[valid] * dry[i0[valid] + 1]
    if spreading:
        out = out / r
    if air_absorb:
        out = _apply_air_absorption(out, sr, r)
    return out


def _apply_air_absorption(x, sr, r, nfft=1024, hop=256):
    freqs = np.fft.rfftfreq(nfft, 1 / sr)
    alpha = air_absorption_db_per_m(freqs)
    w = np.hanning(nfft)
    n = len(x)
    out = np.zeros(n + nfft)
    norm = np.zeros(n + nfft)
    for s in range(0, n - nfft + 1, hop):
        r_frame = float(np.mean(r[s:s + nfft]))
        gain = 10 ** (-alpha * r_frame / 20)
        seg = np.fft.irfft(np.fft.rfft(w * x[s:s + nfft]) * gain, nfft)
        out[s:s + nfft] += seg * w
        norm[s:s + nfft] += w ** 2
    return (out[:n] / np.maximum(norm[:n], 1e-8))


def render_static_source(dry, sr, r, air_absorb=True):
    """Static source at distance r: pure delay + 1/r + absorption."""
    n = len(dry)
    pos = np.tile([r, 0.0, 0.0], (n, 1))
    return render_moving_source(dry, sr, pos, air_absorb=air_absorb)
