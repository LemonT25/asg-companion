"""Modal impact synthesis block of Phi. Analytic, deterministic.
impact(t) = sum_i a_i exp(-d_i t) sin(2 pi f_i sqrt(k) t)
do(material) = stiffness scale k: modal frequencies scale by sqrt(k)."""
import numpy as np

BASE_MODES = {  # (freq Hz, decay 1/s, amp) — four-mode plate-like base set
    "plate": [(413.0, 22.0, 1.0), (721.0, 30.0, 0.62), (1153.0, 45.0, 0.40),
              (1618.0, 60.0, 0.25)],
    "bar": [(292.0, 18.0, 1.0), (806.0, 34.0, 0.55), (1580.0, 55.0, 0.33),
            (2612.0, 80.0, 0.18)],
    "bowl": [(524.0, 12.0, 1.0), (1214.0, 20.0, 0.5), (2109.0, 35.0, 0.3),
             (3170.0, 55.0, 0.15)],
    "shell": [(347.0, 28.0, 1.0), (901.0, 42.0, 0.58), (1497.0, 66.0, 0.35),
              (2233.0, 95.0, 0.2)],
}


def modal_impact(sr, k=1.0, obj="plate", dur=1.5, strike_jitter=0.0, rng=None):
    """Render an impact with stiffness scale k. strike_jitter adds small
    per-mode amplitude variation (excitation randomness), seeded via rng."""
    t = np.arange(int(dur * sr)) / sr
    x = np.zeros_like(t)
    for (f, d, a) in BASE_MODES[obj]:
        aj = a * (1.0 + (rng.uniform(-strike_jitter, strike_jitter)
                         if rng is not None and strike_jitter > 0 else 0.0))
        x += aj * np.exp(-d * t) * np.sin(2 * np.pi * f * np.sqrt(k) * t)
    return x / (np.max(np.abs(x)) + 1e-9) * 0.5


def modal_peak_khat(x, sr, obj="plate"):
    """Recover stiffness scale from the fundamental modal peak."""
    f0 = BASE_MODES[obj][0][0]
    n = len(x)
    spec = np.abs(np.fft.rfft(x * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1 / sr)
    lo, hi = f0 * 0.6, f0 * 2.4  # covers k in [0.36, 5.7]
    band = (freqs >= lo) & (freqs <= hi)
    fhat = freqs[band][np.argmax(spec[band])]
    return (fhat / f0) ** 2
