"""A12.4 (R-A): existence-threshold calibration for sep2's existence
head.

Pre-registration discipline: the threshold is swept and selected
(F1-max) on the TRAIN split ONLY; the val split is touched exactly
once, with the frozen calibrated threshold, to re-report the main-eval
inference existence F1 and re-test gate C5 (F1 >= 0.90).

Val reporting uses (a) the exact 250-scene main-eval cohort
(reproduced: np.random.default_rng(0).choice(len(val), 250,
replace=False), the first draw of p3_edit_eval's module rng) and
(b) the full 2000-scene val split (robustness).

-> paper_numbers/p3_exist_calib_v0.json
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/workspace/CF_ASG/src")
from p3_train import SceneModel, SceneSet, MAXK  # noqa: E402

ROOT = Path("/workspace/CF_ASG")
PN = ROOT / "paper_numbers"
CK = ROOT / "p3" / "ckpt"


def collect_probs(model, ds, dev, batch=32, limit=0):
    """Existence sigmoid probs + GT on-flags for every slot of every
    scene (deterministic order)."""
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False,
                                     num_workers=6)
    probs, gts = [], []
    n_done = 0
    with torch.no_grad():
        for y, stems, dries, exist, traj, thn, alpha, env, K in dl:
            _, e, *_ = model.slots(y.to(dev))
            probs.append(torch.sigmoid(e).cpu().numpy())
            gts.append(exist.numpy())
            n_done += y.shape[0]
            if n_done % 3200 < batch:
                print(f"{n_done}/{len(ds)}", flush=True)
            if limit and n_done >= limit:
                break
    return np.concatenate(probs), np.concatenate(gts)


def f1_at(probs, gts, thr):
    pred = probs > thr
    on = gts > 0.5
    tp = int((pred & on).sum())
    fp = int((pred & ~on).sum())
    fn = int((~pred & on).sum())
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    return {"threshold": float(thr), "precision": p, "recall": r,
            "f1": 2 * p * r / max(1e-9, p + r),
            "tp": tp, "fp": fp, "fn": fn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="p3_exist_calib_v0.json")
    args = ap.parse_args()
    torch.set_num_threads(8)
    dev = "cuda"

    models = {}
    smodel = SceneModel("sep2").to(dev)
    smodel.load_state_dict(torch.load(CK / "sep2_latest.pt",
                                      map_location=dev)["model"])
    smodel.eval()
    models["sep2"] = smodel
    gmodel = SceneModel("graph").to(dev)
    gmodel.load_state_dict(torch.load(CK / "graph_latest.pt",
                                      map_location=dev)["model"])
    gmodel.eval()
    models["graph"] = gmodel   # context only; calibration target is sep2

    tr = SceneSet("train")
    va = SceneSet("val")
    cohort = np.random.default_rng(0).choice(len(va), 250, replace=False)
    thrs = np.linspace(0.01, 0.99, 197)

    out = {"meta": {
        "date": str(datetime.date.today()),
        "script": "src/p3_exist_calib.py",
        "protocol": ("PROTOCOL_P3.md A12.4 / fix plan R-A: threshold "
                     "selected F1-max on the FULL TRAIN split; val "
                     "touched once with the frozen threshold; gate C5 "
                     "F1>=0.90 re-tested on the main-eval 250-scene "
                     "cohort (np.random.default_rng(0).choice(2000,250) "
                     "-- identical to p3_edit_eval's draw) and on full "
                     "val"),
        "ckpts": {"sep2": "p3/ckpt/sep2_latest.pt",
                  "graph": "p3/ckpt/graph_latest.pt"},
        "n_train": len(tr), "n_val": len(va),
        "cohort_n": int(len(cohort))}}

    for name, model in models.items():
        print(f"== {name}: train sweep", flush=True)
        ptr, gtr = collect_probs(model, tr, dev)
        curve = [f1_at(ptr, gtr, t) for t in thrs]
        best = max(curve, key=lambda c: c["f1"])
        print(f"{name} train best thr={best['threshold']:.3f} "
              f"f1={best['f1']:.4f}", flush=True)
        print(f"== {name}: val (frozen thresholds)", flush=True)
        pva, gva = collect_probs(model, va, dev)
        pco, gco = pva[cohort], gva[cohort]
        out[name] = {
            "train_best": best,
            "train_f1_at_0.5": f1_at(ptr, gtr, 0.5),
            "train_curve": [{"threshold": c["threshold"],
                             "f1": c["f1"]} for c in curve],
            "val_cohort250_at_0.5": f1_at(pco, gco, 0.5),
            "val_cohort250_at_calib": f1_at(pco, gco, best["threshold"]),
            "val_full_at_0.5": f1_at(pva, gva, 0.5),
            "val_full_at_calib": f1_at(pva, gva, best["threshold"])}

    c5 = out["sep2"]["val_cohort250_at_calib"]["f1"]
    out["verdict"] = {
        "C5_gate": "existence F1 >= 0.90 on val (PROTOCOL_P3.md C5)",
        "sep2_val_cohort250_f1_calibrated": c5,
        "C5_pass": bool(c5 >= 0.90),
        "sep2_val_full_f1_calibrated":
            out["sep2"]["val_full_at_calib"]["f1"]}
    (PN / args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out["sep2"].items()
                      if k != "train_curve"}, indent=1))
    print(json.dumps(out["verdict"], indent=1))


if __name__ == "__main__":
    main()
