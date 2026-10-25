"""Shoebox image-source RIR (static source) + Sabine closed form.
Deterministic; uniform wall absorption alpha; rigid-walk image method with
1/r spreading and (1-alpha)^k reflection loss. v0: no air absorption inside
RIR (applied separately if needed), no fractional-delay on images (nearest
sample), order-capped."""
import numpy as np

from core import C_SOUND


def sabine_rt60(room_dim, alpha):
    L, W, H = room_dim
    V = L * W * H
    S = 2 * (L * W + L * H + W * H)
    return 0.161 * V / (S * alpha)


def image_source_rir(room_dim, src, mic, sr, alpha=0.3, max_order=40,
                     rir_len_s=None):
    L, W, H = room_dim
    if rir_len_s is None:
        rir_len_s = min(8.0, 2.5 * sabine_rt60(room_dim, alpha))
    n = int(rir_len_s * sr)
    rir = np.zeros(n)
    refl = np.sqrt(1 - alpha)  # pressure reflection magnitude
    dims = np.array([L, W, H])
    src = np.asarray(src, float)
    mic = np.asarray(mic, float)
    for ix in range(-max_order, max_order + 1):
        for iy in range(-max_order, max_order + 1):
            for iz in range(-max_order, max_order + 1):
                k = abs(ix) + abs(iy) + abs(iz)
                img = np.empty(3)
                for d, m in enumerate((ix, iy, iz)):
                    p = src[d] if m % 2 == 0 else dims[d] - src[d]
                    img[d] = m * dims[d] + p
                r = np.linalg.norm(img - mic)
                idx = int(round(r / C_SOUND * sr))
                if idx < n:
                    rir[idx] += (refl ** k) / max(r, 0.1)
    return rir
