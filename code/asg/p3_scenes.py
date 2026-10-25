"""P3 corpus: multi-source physical scenes rendered through Phi.

Scene: K ~ U{1,2,3} sources; per source a trajectory (static at r, or
passby at v), relative gain U[-6,6] dB; shared room alpha on a 9-point
grid applied per stem (LTI => mix = sum of wet stems exactly).
Stores mix, wet stems, dry sources, all parameters. GT for every edit
is renderable from the stored graph.
-> p3/scenes/{train,val}/scene_<idx>.npz

A12.1 confirmation cohort:
  python p3_scenes.py --split conf1 --n 250 --seed 1
renders 250 fresh scenes into p3/scenes3/conf1 from the val clip pool
(never trained on).  The conf1 job stream is seeded
SeedSequence([2, seed]): the legacy train/val streams used plain
seeds 0/1, so plain seed 1 would REPLICATE the first val scenes --
the namespaced stream keeps the pre-registered seed while
guaranteeing fresh, disjoint scene configurations (verified
post-render against the main-cohort scenes).
"""
import argparse
import csv
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, "/workspace/CF_PhysAudio/src")
sys.path.insert(0, "/workspace/audiocf_render")
from p2_common import SR, phi_velocity, phi_distance, phi_reverb  # noqa: E402

FSD = Path("/workspace/data/fsd50k")
AUDIO = FSD / "FSD50K.dev_audio"
OUT = Path("/workspace/CF_ASG/p3/scenes3")
SEG = int(4.0 * SR)
N_TRAIN, N_VAL = 24000, 2000
ALPHAS = [round(0.30 + 0.05 * i, 2) for i in range(5)]
TONAL = ["Siren", "Vehicle_horn_and_car_horn_and_honking", "Church_bell",
         "Bell", "Doorbell", "Cowbell", "Bicycle_bell", "Alarm", "Engine"]
BROAD = TONAL + ["Knock", "Clapping", "Slam", "Drum", "Snare_drum",
                 "Bass_drum", "Glass", "Gunshot_and_gunfire",
                 "Thump_and_thud", "Hammer", "Tap", "Door",
                 "Engine_starting", "Sliding_door"]


def build_pools():
    rows = list(csv.DictReader(open(FSD / "FSD50K.ground_truth" / "dev.csv")))
    pools = {"train": {"tonal": [], "broad": []},
             "val": {"tonal": [], "broad": []}}
    for r in rows:
        split = "val" if r["split"] == "val" else "train"
        labels = set(r["labels"].split(","))
        for pname, pool in (("tonal", TONAL), ("broad", BROAD)):
            hit = sorted(labels & set(pool))
            if hit:
                pools[split][pname].append((r["fname"], hit[0]))
    return pools


def load_clip(fname):
    x, sr = sf.read(str(AUDIO / f"{fname}.wav"), dtype="float64")
    if x.ndim > 1:
        x = x.mean(1)
    if sr != SR:
        x = resample_poly(x, SR, sr)
    r = np.sqrt(np.mean(x ** 2) + 1e-12)
    return x / r * 0.1


def seg_of(x, rng):
    if len(x) >= SEG:
        i = rng.integers(0, len(x) - SEG + 1)
        return x[i:i + SEG]
    out = np.zeros(SEG)
    i = rng.integers(0, SEG - len(x) + 1)
    out[i:i + len(x)] = x
    return out


def render_source(d, traj, theta, alpha, gain_db):
    if traj == "passby":
        w = phi_velocity(d, theta)
    else:
        w = phi_distance(d, theta)
    if alpha is not None:
        w = phi_reverb(w[:SEG], alpha)[:SEG]
    return w[:SEG] * (10 ** (gain_db / 20.0))


def render_one(job):
    split, idx, srcs, alpha, seed = job
    p = OUT / split / f"scene_{idx:05d}.npz"
    if p.exists():
        return p.name
    rng = np.random.default_rng(seed)
    stems, dries, meta = [], [], []
    for fname, cls, traj in srcs:
        try:
            d = seg_of(load_clip(fname), rng)
        except Exception as e:
            return f"SKIP {fname}: {e}"
        if traj == "passby":
            theta = float(rng.uniform(5, 40))
        else:
            theta = float(np.exp(rng.uniform(np.log(1), np.log(32))))
        gain_db = float(rng.uniform(-6, 6))
        stems.append(render_source(d, traj, theta, alpha, gain_db))
        dries.append(d * (10 ** (gain_db / 20.0)))
        meta.append({"clip": fname, "cls": cls, "traj": traj,
                     "theta": theta, "gain_db": gain_db})
    mix = np.sum(stems, axis=0)
    peak = np.abs(mix).max() + 1e-9
    np.savez_compressed(
        p, y=(mix / peak).astype(np.float32),
        stems=(np.stack(stems) / peak).astype(np.float32),
        dries=(np.stack(dries) / peak).astype(np.float32),
        alpha=np.float32(-1.0 if alpha is None else alpha),
        env=(0 if alpha is None else 1),
        gain=np.float32(peak), K=np.int64(len(srcs)),
        meta=json.dumps(meta))
    return p.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="all",
                    choices=("all", "conf1", "mmB"),
                    help="'all' = legacy train+val render; 'conf1' = "
                         "A12.1 confirmation cohort; 'mmB' = A13 "
                         "physics-mismatch cohort (run via p3_phiB)")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    pools = build_pools()
    if args.split == "mmB":
        # A13 cohort: namespaced stream [3, seed], val clip pool
        renders = [("mmB", args.n,
                    np.random.default_rng(
                        np.random.SeedSequence([3, args.seed])), "val")]
    elif args.split == "conf1":
        # fresh cohort from the (never-trained-on) val clip pool;
        # namespaced seed stream, disjoint from train(0)/val(1)
        renders = [("conf1", args.n,
                    np.random.default_rng(
                        np.random.SeedSequence([2, args.seed])), "val")]
    else:
        renders = [("train", N_TRAIN, np.random.default_rng(0), "train"),
                   ("val", N_VAL, np.random.default_rng(1), "val")]
    for split, _, _, pool_split in renders:
        (OUT / split).mkdir(parents=True, exist_ok=True)
        print(split, {k: len(v) for k, v in pools[pool_split].items()},
              flush=True)
    jobs = []
    for split, n, rng, pool_split in renders:
        tonal = pools[pool_split]["tonal"]
        broad = pools[pool_split]["broad"]
        for i in range(n):
            K = int(rng.integers(1, 4))
            indoor = rng.random() < 0.5
            srcs, used = [], set()
            for k in range(K):
                # pass-bys only in open scenes (readable Doppler; also
                # the physically natural configuration)
                traj = ("static" if indoor
                        else ("passby" if rng.random() < 0.5 else "static"))
                pool = tonal if traj == "passby" else broad
                # pairwise-distinct classes within a scene (A2):
                # same-class simultaneous sources are unidentifiable
                for _ in range(30):
                    fname, cls = pool[int(rng.integers(len(pool)))]
                    if cls not in used:
                        break
                used.add(cls)
                srcs.append((fname, cls, traj))
            alpha = (ALPHAS[int(rng.integers(len(ALPHAS)))]
                     if indoor else None)
            jobs.append((split, i, srcs, alpha, int(rng.integers(2 ** 31))))
    print(f"{len(jobs)} scenes", flush=True)
    done = 0
    import os
    with Pool(1 if os.environ.get('PHIB') else 8) as pool:
        for name in pool.imap(render_one, jobs, 32):
            done += 1
            if done % 2000 == 0:
                print(done, "/", len(jobs), flush=True)
    print("scenes done")


if __name__ == "__main__":
    main()
