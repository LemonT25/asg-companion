"""P2 shared: command grids, class/prompt sets, Phi wrappers, verifiers.
All operational constants locked by PROTOCOL_P2.md v1.0."""
import sys

import numpy as np
from scipy.signal import fftconvolve

sys.path.insert(0, "/workspace/audiocf_render")
sys.path.insert(0, "/workspace/CF_PhysAudio/src")
from core import (render_moving_source, render_static_source,  # noqa: E402
                  passby_positions)
from room import image_source_rir, sabine_rt60  # noqa: E402
from modal import modal_impact, modal_peak_khat, BASE_MODES  # noqa: E402
from g0b_verify import (doppler_vhat_ridge, rms_dbfs,  # noqa: E402
                        rt60_schroeder, stft_mag, split_segments)

SR = 16000
SEEDS = list(range(8))
V_GRID = [5, 10, 20, 30, 40]
R_GRID = [1, 2, 4, 8, 16, 32]
ALPHA_GRID = [0.1, 0.2, 0.35, 0.5]
K_GRID = [0.5, 1.0, 2.0, 4.0]
ROOM = (30.0, 20.0, 8.0)
RIR_SRC, RIR_MIC = (18.0, 12.0, 1.5), (15.0, 10.0, 1.6)

CLASSES = {
    "velocity": {
        "siren": "a continuous ambulance siren, steady tone, dry recording, no reverberation",
        "engine": "a sustained car engine hum, constant speed, dry recording, no reverberation",
        "organ": "a single sustained church organ note, dry recording, no reverberation",
        "drill": "a continuous electric drill sound, steady, dry recording, no reverberation",
    },
    "distance": {
        "dog": "a single dog bark, dry recording, no reverberation",
        "horn": "a single car horn honk, dry recording, no reverberation",
        "clap": "a single hand clap, dry recording, no reverberation",
        "drum": "a single drum hit, dry recording, no reverberation",
    },
    "reverb": {
        "clap": "a single hand clap, dry recording, no reverberation",
        "knock": "a single knock on wood, dry recording, no reverberation",
        "drum": "a single drum hit, dry recording, no reverberation",
        "snap": "a single finger snap, dry recording, no reverberation",
    },
}
B1_TMPL = {
    "velocity": "a {cls} passing by at {v} meters per second",
    "distance": "a {cls} {r} meters away from the microphone",
    "reverb": "a {cls} in a room with a reverberation time of {rt:.1f} seconds",
}
B2_TMPL = {
    "velocity": {5: "slowly", 10: "at moderate speed", 20: "fast",
                 30: "very fast", 40: "extremely fast"},
    "distance": {1: "very close to the microphone", 2: "close by",
                 4: "a few meters away", 8: "some distance away",
                 16: "far away", 32: "very far away in the distance"},
    "reverb": {0.1: "in a very reverberant huge cathedral",
               0.2: "in a large reverberant hall",
               0.35: "in a medium room with some reverberation",
               0.5: "in a small damped room with little reverberation"},
}
B_CLS = {"velocity": {"siren": "ambulance siren", "engine": "car engine",
                      "organ": "church organ note", "drill": "electric drill"},
         "distance": {"dog": "dog barking", "horn": "car horn",
                      "clap": "hand clap", "drum": "drum hit"},
         "reverb": {"clap": "hand clap", "knock": "knock on wood",
                    "drum": "drum hit", "snap": "finger snap"}}

_RIR_CACHE = {}


def phi_velocity(dry, v):
    n = len(dry)
    pos = passby_positions(n, SR, v=v, d0=2.0, t_close=n / SR / 2)
    return render_moving_source(dry, SR, pos)


def phi_distance(dry, r):
    return render_static_source(dry, SR, r)


def phi_reverb(dry, alpha):
    if alpha not in _RIR_CACHE:
        _RIR_CACHE[alpha] = image_source_rir(ROOM, RIR_SRC, RIR_MIC, SR,
                                             alpha=alpha, max_order=22,
                                             rir_len_s=2.5)
    y = fftconvolve(dry, _RIR_CACHE[alpha])[:len(dry) + int(2.5 * SR)]
    return y


def rt60_theory(alpha):
    """Measured RT60 of the image-source RIR itself (exact for geometry)."""
    rir = _RIR_CACHE.get(alpha)
    if rir is None:
        phi_reverb(np.zeros(10), alpha)
        rir = _RIR_CACHE[alpha]
    rt, _ = rt60_schroeder(rir)
    return rt


def rt60_after_offset(y, offset_idx):
    """RT60 measured only on the decay after the (known) dry offset —
    the event-isolated variant demanded by PROTOCOL_P2 (G0-B lesson)."""
    tail = y[offset_idx:]
    return rt60_schroeder(tail)
