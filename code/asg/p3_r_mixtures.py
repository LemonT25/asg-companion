"""A11 protocol R: construct the R-E1/R-E3 real-domain evaluation set.

120 additive two-source mixtures, fully real content:
  vehicle side  = 4-s VS13 held-out-car crop around the labeled CPA
                  (i < 60; held-out cars from the EDITOR checkpoint
                  est2_vs13.pt val_cars key), OR an IDMT-Traffic dry car
                  pass on the mid mic (i >= 60).  NOTE (documented
                  dataset property): IDMT-Traffic clips in this copy are
                  fixed 2.0-s files, so the pass is center-embedded in
                  the 4-s window with zero padding; SNR is computed on
                  the pass's active support.
  distractor    = one FSD50K.eval clip (>= 4 s, max-RMS 4-s window,
                  resampled to 16 kHz mono, vehicle-like classes
                  excluded -- see p3_r_common.VEHICLE_KEYWORDS), scaled
                  to a drawn SNR ~ U[0, 10] dB below the vehicle.
  mix = stem_vehicle + stem_distractor exactly (additive), so removal
  and gain-edit references are exact with fully real content.

Deterministic: per-mixture rng seed 1000+i (SNR), fixed permutation
seeds for source assignment -- a smoke run is a strict prefix of the
full run.

Also writes 200 UNMIXED real 4-s crops (140 VS13 across all cars + 60
IDMT car passes disjoint from the mixture set) to p3/rset_ref/ as the
FAD-CLAP reference set (float wavs, 16 kHz).

Run (audiocf venv):
  HF_HOME=/workspace/hf_cache OMP_NUM_THREADS=8 nice -n 15 \
    /workspace/venvs/audiocf/bin/python src/p3_r_mixtures.py
-> p3/rset/mix_XXX.npz, p3/rset/manifest.json,
   p3/rset_ref/ref_XXX.wav, p3/rset_ref/manifest_ref.json
"""
import argparse
import csv
import datetime
import json
import sys

import numpy as np
import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, "/workspace/CF_ASG/src")
from p3_r_common import (ROOT, RSET, RREF, SR, SEG, VS, FSD_AUDIO,  # noqa
                         FSD_GT, CK, VEHICLE_KEYWORDS, fsd_query,
                         peak03, load_vs13_items, idmt_car_passes)


def load_16k(path):
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    if sr != SR:
        x = torchaudio.functional.resample(
            torch.tensor(x), sr, SR).numpy()
    return x.astype(np.float64)


def vs13_crop(item):
    """4-s crop at the labeled CPA, peak-norm 0.3 (VS13 eval convention,
    no jitter -- mirrors p3_est2_vs13.VS13Set train=False)."""
    x = load_16k(item["wav"])
    c = int(item["cpa"] * SR)
    half = SEG // 2
    c = max(half, min(len(x) - half, c))
    return peak03(x[c - half:c + half])


def idmt_embed(item):
    """2.0-s IDMT pass -> center of a 4-s zero-padded window."""
    x = load_16k(item["wav"])
    x = x[:SEG]
    out = np.zeros(SEG)
    st = (SEG - len(x)) // 2
    out[st:st + len(x)] = x
    return peak03(out), (st, st + len(x))


def fsd_window(path):
    """Max-RMS 4-s window (hop 0.5 s) of the 16 kHz mono clip,
    peak-normalized to 1.0. None if unusable."""
    x = load_16k(path)
    if len(x) < SEG:
        return None, None
    hop = SR // 2
    n = 1 + (len(x) - SEG) // hop
    offs = [i * hop for i in range(n)]
    rms = [float(np.sqrt((x[o:o + SEG] ** 2).mean())) for o in offs]
    o = offs[int(np.argmax(rms))]
    w = x[o:o + SEG]
    w = w / (np.abs(w).max() + 1e-9)
    if float(np.sqrt((w ** 2).mean())) < 0.01:
        return None, None       # essentially silent even at peak norm
    return w, o / SR


def fsd_eligible():
    rows = list(csv.DictReader(open(FSD_GT)))
    out = []
    for r in rows:
        labs = r["labels"].split(",")
        low = ",".join(labs).lower()
        if any(k in low for k in VEHICLE_KEYWORDS):
            continue
        out.append({"fname": r["fname"], "labels": labs})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_mix", type=int, default=120)
    ap.add_argument("--n_ref", type=int, default=200)
    ap.add_argument("--skip_ref", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(8)
    RSET.mkdir(parents=True, exist_ok=True)
    RREF.mkdir(parents=True, exist_ok=True)

    ck = torch.load(CK / "est2_vs13.pt", map_location="cpu",
                    weights_only=False)
    val_cars = list(ck["val_cars"])
    items = load_vs13_items()
    held = [it for it in items if it["car"] in val_cars]
    idmt = idmt_car_passes()
    print(f"editor val_cars={val_cars}  held-out clips={len(held)}  "
          f"idmt dry car passes={len(idmt)}", flush=True)

    p_vs = np.random.default_rng(3001).permutation(len(held))
    p_id = np.random.default_rng(3002).permutation(len(idmt))
    fsd = fsd_eligible()
    p_fsd = np.random.default_rng(3000).permutation(len(fsd))
    print(f"fsd eligible clips={len(fsd)}", flush=True)

    n_half = args.n_mix // 2 if args.n_mix >= 2 else args.n_mix
    manifest = []
    fi = 0                       # cursor into the FSD permutation
    for i in range(args.n_mix):
        if i < n_half:
            it = held[int(p_vs[i % len(held)])]
            veh = vs13_crop(it)
            support = (0, SEG)
            vmeta = {"veh_source": "vs13", "wav": it["wav"],
                     "car": it["car"], "v_kmh": it["v_kmh"],
                     "cpa": it["cpa"]}
        else:
            it = idmt[int(p_id[(i - n_half) % len(idmt)])]
            veh, support = idmt_embed(it)
            vmeta = {"veh_source": "idmt", "wav": it["wav"],
                     "speed_limit_kmh": it["speed_kmh"],
                     "type_dir": it["type_dir"],
                     "note": "2.0-s clip center-embedded, zero-padded"}
        # next valid FSD distractor
        dist = None
        while dist is None:
            if fi >= len(fsd):
                raise RuntimeError("ran out of FSD candidates")
            cand = fsd[int(p_fsd[fi])]
            fi += 1
            path = FSD_AUDIO / f"{cand['fname']}.wav"
            try:
                if sf.info(str(path)).duration < 4.05:
                    continue
                w, off = fsd_window(str(path))
            except Exception:
                continue
            if w is None:
                continue
            dist = (cand, w, off)
        cand, w, off = dist
        rng = np.random.default_rng(1000 + i)
        snr = float(rng.uniform(0.0, 10.0))
        act = veh[support[0]:support[1]]
        v_rms = float(np.sqrt((act ** 2).mean()) + 1e-12)
        d_rms = float(np.sqrt((w ** 2).mean()) + 1e-12)
        g = v_rms / d_rms * 10 ** (-snr / 20)
        sd = (g * w).astype(np.float64)
        mix = veh + sd
        meta = {"idx": i, **vmeta,
                "veh_support": list(support),
                "fsd_fname": cand["fname"], "fsd_labels": cand["labels"],
                "fsd_query": fsd_query(cand["labels"][0]),
                "fsd_offset_s": off, "fsd_gain": g, "snr_db": snr}
        np.savez_compressed(
            RSET / f"mix_{i:03d}.npz",
            mix=mix.astype(np.float32),
            stem_vehicle=veh.astype(np.float32),
            stem_distractor=sd.astype(np.float32),
            snr=np.float64(snr), meta=json.dumps(meta))
        manifest.append(meta)
        if (i + 1) % 20 == 0:
            print(f"mix {i + 1}/{args.n_mix}", flush=True)

    (RSET / "manifest.json").write_text(json.dumps(
        {"meta": {"date": str(datetime.date.today()),
                  "script": "src/p3_r_mixtures.py", "seed": 0,
                  "n_mix": args.n_mix, "editor_val_cars": val_cars,
                  "snr_range_db": [0, 10],
                  "notes": ("additive real mixtures: mix = stem_vehicle "
                            "+ stem_distractor exactly; VS13 vehicle = "
                            "4-s CPA crop of an editor-held-out car, "
                            "peak-norm 0.3; IDMT vehicle = 2.0-s dry "
                            "car pass (dataset ships fixed 2-s files) "
                            "center-embedded in the 4-s window with "
                            "zero padding, SNR on active support; "
                            "distractor = FSD50K.eval max-RMS 4-s "
                            "window, vehicle-like classes excluded, "
                            "SNR ~ U[0,10] dB below vehicle")},
         "mixtures": manifest}, indent=1))

    if args.skip_ref:
        return
    # ---- FAD reference set: 200 unmixed real crops ----
    n_vs = int(round(args.n_ref * 0.7))
    n_id = args.n_ref - n_vs
    rngr = np.random.default_rng(2000)
    pick = rngr.choice(len(items), size=min(n_vs, len(items)),
                       replace=False)
    refs = []
    j = 0
    for k in pick:
        it = items[int(k)]
        sf.write(str(RREF / f"ref_{j:03d}.wav"),
                 vs13_crop(it).astype(np.float32), SR, subtype="FLOAT")
        refs.append({"idx": j, "source": "vs13", "wav": it["wav"],
                     "car": it["car"], "held_out_editor":
                     it["car"] in val_cars})
        j += 1
    for k in range(n_id):
        it = idmt[int(p_id[(n_half + k) % len(idmt)])]  # disjoint from mixes
        veh, _ = idmt_embed(it)
        sf.write(str(RREF / f"ref_{j:03d}.wav"),
                 veh.astype(np.float32), SR, subtype="FLOAT")
        refs.append({"idx": j, "source": "idmt", "wav": it["wav"]})
        j += 1
    (RREF / "manifest_ref.json").write_text(json.dumps(
        {"meta": {"n_ref": j, "date": str(datetime.date.today()),
                  "notes": ("FAD-CLAP reference: unmixed real 4-s "
                            "crops; 70% VS13 (all cars, CPA crops), "
                            "30% IDMT dry car passes (2-s, "
                            "center-embedded), peak-norm 0.3")},
         "refs": refs}, indent=1))
    print(f"wrote {len(manifest)} mixtures, {j} refs", flush=True)


if __name__ == "__main__":
    main()
