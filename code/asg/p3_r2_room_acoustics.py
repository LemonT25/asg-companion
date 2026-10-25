"""TASLP R1 (reviewer 1): room-acoustics descriptors of the simulated room.

The corpus room is rendered by an image-source model (room.py), i.e. a
GEOMETRIC-acoustics model: every reflection order up to 22 contributes
a phantom source with 1/r spreading and (1-alpha)^k reflection loss.
This script measures, per absorption node alpha:
  * Sabine RT60 (statistical acoustics closed form)
  * Schroeder-integrated T20/T30 of the image-source RIR (geometric)
  * direct-to-reverberant ratio, early (<50 ms) energy fraction,
    number of phantom sources arriving within 50 ms and 100 ms
  * Schroeder frequency f_s = 2000 sqrt(RT60 / V): the boundary below
    which wave acoustics (isolated modes) dominates; above it the modal
    overlap is high and the geometric / statistical description holds
  * modal density per Hz at 50/100/200 Hz for the shoebox (Bonello-type
    count), and the fraction of corpus signal energy below f_s
Deterministic, no learning. -> paper_numbers/p3_room_acoustics_v0.json
"""
import datetime
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/workspace/CF_PhysAudio/src")
sys.path.insert(0, "/workspace/audiocf_render")
from room import image_source_rir, sabine_rt60  # noqa: E402
from core import C_SOUND  # noqa: E402

ROOM = (30.0, 20.0, 8.0)
RIR_SRC, RIR_MIC = (18.0, 12.0, 1.5), (15.0, 10.0, 1.6)
SR = 16000
ALPHAS = [round(0.30 + 0.05 * i, 2) for i in range(5)]
PN = Path("/workspace/CF_ASG/paper_numbers")


def schroeder(rir, sr):
    e = rir ** 2
    edc = np.cumsum(e[::-1])[::-1]
    edc_db = 10 * np.log10(edc / edc[0] + 1e-30)
    t = np.arange(len(rir)) / sr
    out = {}
    for name, lo, hi in (("T20", -5, -25), ("T30", -5, -35)):
        m = (edc_db <= lo) & (edc_db >= hi)
        if m.sum() < 10:
            out[name] = None
            continue
        p = np.polyfit(t[m], edc_db[m], 1)
        out[name] = float(-60.0 / p[0])
    return out


def mode_count(fmax, room):
    """Number of shoebox modes with f <= fmax (exact enumeration)."""
    L, W, H = room
    n = 0
    nx = int(2 * L * fmax / C_SOUND) + 1
    ny = int(2 * W * fmax / C_SOUND) + 1
    nz = int(2 * H * fmax / C_SOUND) + 1
    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz + 1):
                if i == j == k == 0:
                    continue
                f = (C_SOUND / 2) * np.sqrt((i / L) ** 2 + (j / W) ** 2
                                            + (k / H) ** 2)
                if f <= fmax:
                    n += 1
    return n


def main():
    L, W, H = ROOM
    V = L * W * H
    res = {"meta": {"date": str(datetime.date.today()),
                    "script": "src/p3_r2_room_acoustics.py",
                    "room_m": ROOM, "volume_m3": V,
                    "src": RIR_SRC, "mic": RIR_MIC, "sr": SR,
                    "renderer": "image-source, max_order 22, rir_len 2.5 s,"
                                " uniform alpha, 1/r spreading,"
                                " (1-alpha)^k reflection loss",
                    "notes": "geometric acoustics = image sources (phantom"
                             " sources); wave acoustics matters below the"
                             " Schroeder frequency f_s; statistical"
                             " acoustics = Sabine closed form"},
           "per_alpha": {}}
    d_direct = float(np.linalg.norm(np.array(RIR_SRC) - np.array(RIR_MIC)))
    res["meta"]["direct_path_m"] = d_direct
    for a in ALPHAS:
        rir = image_source_rir(ROOM, RIR_SRC, RIR_MIC, SR, alpha=a,
                               max_order=22, rir_len_s=2.5)
        sch = schroeder(rir, SR)
        rt_sab = sabine_rt60(ROOM, a)
        t_rt = sch["T30"] or sch["T20"] or rt_sab
        f_s = 2000.0 * np.sqrt(t_rt / V)
        i0 = int(round(d_direct / C_SOUND * SR))
        e = rir ** 2
        e_dir = e[max(0, i0 - 8):i0 + 8].sum()
        e_tot = e.sum()
        n50 = int(SR * 0.050)
        n100 = int(SR * 0.100)
        early = e[:i0 + n50].sum()
        drr = 10 * np.log10(e_dir / max(e_tot - e_dir, 1e-30))
        nz = np.nonzero(rir)[0]
        res["per_alpha"][str(a)] = {
            "sabine_rt60_s": float(rt_sab),
            "schroeder_T20_s": sch["T20"], "schroeder_T30_s": sch["T30"],
            "schroeder_frequency_hz": float(f_s),
            "drr_db": float(drr),
            "early_energy_fraction_50ms": float(early / e_tot),
            "n_image_arrivals_le_50ms": int(((nz - i0) <= n50).sum()),
            "n_image_arrivals_le_100ms": int(((nz - i0) <= n100).sum()),
            "n_image_arrivals_total": int(len(nz)),
        }
    res["modal"] = {
        "modes_below_50hz": mode_count(50.0, ROOM),
        "modes_below_100hz": mode_count(100.0, ROOM),
        "modes_below_200hz": mode_count(200.0, ROOM),
        "modal_density_per_hz_at_100hz": (mode_count(105.0, ROOM)
                                          - mode_count(95.0, ROOM)) / 10.0,
        "notes": "exact shoebox mode enumeration; above f_s modal overlap"
                 " (density x bandwidth) exceeds 3 and the room behaves"
                 " statistically; renderer passband is 0-8 kHz at 16 kHz"}
    # fraction of corpus energy below f_s (val scenes, if present)
    scenes = sorted(Path("/workspace/CF_ASG/p3/scenes3/val").glob("*.npz"))
    if scenes:
        fs_max = max(v["schroeder_frequency_hz"]
                     for v in res["per_alpha"].values())
        fr = []
        for p in scenes[:200]:
            z = np.load(p, allow_pickle=True)
            y = z["y"].astype(np.float64)
            Y = np.abs(np.fft.rfft(y)) ** 2
            f = np.fft.rfftfreq(len(y), 1 / SR)
            fr.append(float(Y[f <= fs_max].sum() / Y.sum()))
        res["corpus_energy_fraction_below_fs"] = {
            "n_scenes": len(fr), "fs_max_hz": float(fs_max),
            "median": float(np.median(fr)), "p95": float(np.percentile(fr, 95))}
    (PN / "p3_room_acoustics_v0.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
