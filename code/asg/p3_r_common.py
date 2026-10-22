"""A11 real-domain protocol R: shared helpers for p3_r_mixtures /
p3_r_eval_ours / p3_r_e2.

Import-light on purpose (numpy + stdlib only at module level): this
module is imported inside five different venvs (audiocf, flowsep,
soloaudio, omnisep, audioomni, aumetrics).
"""
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path("/workspace/CF_ASG")
PN = ROOT / "paper_numbers"
CK = ROOT / "p3" / "ckpt"
RSET = ROOT / "p3" / "rset"
RREF = ROOT / "p3" / "rset_ref"
ROUT = ROOT / "p3" / "rset_out"
SR = 16000
SEG = 64000                      # 4 s @ 16 kHz, same as synthetic protocol

VS = Path("/workspace/data/vs13")
IDMT = Path("/workspace/data/idmt_traffic/IDMT_Traffic/audio")
FSD_AUDIO = Path("/workspace/data/fsd50k/FSD50K.eval_audio")
FSD_GT = Path("/workspace/data/fsd50k/FSD50K.ground_truth/eval.csv")

VEH_QUERY = "car passing by"     # external systems' vehicle query (R-E1a)

# FSD50K distractor exclusion: any clip whose label set contains one of
# these substrings is excluded (the distractor must not itself be a
# vehicle/engine sound, or vehicle extraction is ill-posed).
VEHICLE_KEYWORDS = ("car", "vehicle", "bus", "truck", "motorcycle",
                    "engine", "traffic", "train", "rail", "subway",
                    "aircraft", "airplane", "helicopter", "boat",
                    "bicycle", "skateboard", "accelerating", "idling")


def fsd_query(label):
    """FSD label -> natural-language query string, same convention as the
    synthetic harnesses' q_of (first '_and_' alternative, spaces)."""
    lab = label.split("_and_")[0]
    lab = lab.replace("_", " ").replace("(", "").replace(")", "")
    return " ".join(lab.split()).lower()


def si_sdr(est, ref):
    ref = ref - ref.mean()
    est = est - est.mean()
    a = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    e = est - a * ref
    return float(10 * np.log10((a * ref @ (a * ref) + 1e-12)
                               / (e @ e + 1e-12)))


def _stft_mag(x, nfft=1024, hop=256):
    w = np.hanning(nfft)
    n = 1 + (len(x) - nfft) // hop
    frames = np.stack([x[i * hop:i * hop + nfft] * w for i in range(n)])
    return np.abs(np.fft.rfft(frames, axis=-1))


def lsd(a, b):
    """Log-spectral distance (dB): LPAPS substitute, documented in the
    ledgers (no trained LPAPS weights available for 16 kHz general
    audio; LSD + MRSTFT jointly cover the locality axis)."""
    L = min(len(a), len(b))
    Sa = _stft_mag(np.asarray(a, np.float64)[:L])
    Sb = _stft_mag(np.asarray(b, np.float64)[:L])
    la = 10 * np.log10(Sa ** 2 + 1e-10)
    lb = 10 * np.log10(Sb ** 2 + 1e-10)
    return float(np.mean(np.sqrt(np.mean((la - lb) ** 2, axis=-1))))


def peak03(x):
    """Peak-normalize to 0.3, the VS13 eval convention
    (p3_est2_vs13.VS13Set, train=False)."""
    return x / (np.abs(x).max() + 1e-9) * 0.3


def list_rset():
    return sorted(glob.glob(str(RSET / "mix_*.npz")))


def load_mix(path):
    z = np.load(path, allow_pickle=True)
    return {"mix": z["mix"].astype(np.float64),
            "stem_vehicle": z["stem_vehicle"].astype(np.float64),
            "stem_distractor": z["stem_distractor"].astype(np.float64),
            "snr": float(z["snr"]),
            "meta": json.loads(str(z["meta"]))}


def agg(rows, key="sisdr"):
    """Median + bootstrap CI, same shape as the synthetic ledgers."""
    rng = np.random.default_rng(0)
    v = [r[key] for r in rows if r.get(key) is not None] \
        if rows and isinstance(rows[0], dict) else list(rows)
    if not v:
        return None
    m = np.median(rng.choice(v, (5000, len(v))), axis=1)
    return {"median": float(np.median(v)),
            "ci95": [float(np.percentile(m, 2.5)),
                     float(np.percentile(m, 97.5))], "n": len(v)}


def load_vs13_items():
    """All VS13 items (same parse as p3_est2_vs13.load_vs13)."""
    items = []
    for wav in sorted(glob.glob(str(VS / "**/*.wav"), recursive=True)):
        txt = wav.replace(".wav", ".txt")
        try:
            parts = open(txt).read().split()
            v_kmh, cpa = float(parts[0]), float(parts[1])
        except Exception:
            continue
        car = Path(wav).stem.rsplit("_", 1)[0]
        items.append({"wav": wav, "v": v_kmh / 3.6, "v_kmh": v_kmh,
                      "cpa": cpa, "car": car})
    return items


def idmt_car_passes():
    """Dry car pass-bys on the mid mic (type code C in <A|M>_D_<C*>),
    excluding backgrounds. 2.0-s clips (dataset property, documented)."""
    out = []
    for wav in sorted(glob.glob(str(IDMT / "*_ME_CH12.wav"))):
        name = Path(wav).stem
        if name.endswith("-BG"):
            continue
        parts = name.split("_")
        # ..._<speed>Kmh_<pos>_<A|M>_<D|W>_<TypeDir>_ME_CH12
        try:
            typ_dir = parts[-3]
            cond = parts[-4]
            speed = [p for p in parts if p.endswith("Kmh")][0]
        except IndexError:
            continue
        if cond == "D" and typ_dir.startswith("C"):
            out.append({"wav": wav, "speed_kmh": float(speed[:-3]),
                        "type_dir": typ_dir})
    return out
