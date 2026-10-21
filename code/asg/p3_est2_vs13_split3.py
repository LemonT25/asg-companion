"""A12.3 (R-D): three-way VS13 vehicle split + editor/verifier
fine-tunes with NO weight sharing.

Split: the 13 VS13 vehicles are permuted by a seeded RNG
(np.random.default_rng(12), documented) -- first 5 = EDITOR fine-tune
cars, next 5 = VERIFIER fine-tune cars, last 3 = EVALUATION cars
(unseen by both).  Written to p3/ckpt/vs13_split3.json.

Roles:
  --role editor  : base = p3/ckpt/est2_latest.pt (synthetic est2),
                   fine-tuned on the 5 editor cars only
                   -> p3/ckpt/est2_vs13_ed3.pt
  --role verifier: base = p3/ckpt/est2_scratch_v.pt (Est2 retrained
                   FROM SCRATCH on the synthetic corpus with a fresh
                   seed via p3_train.py --kind est2 --seed 1234 --
                   no weight sharing with the editor at ANY stage),
                   fine-tuned on the 5 verifier cars only
                   -> p3/ckpt/est2_vs13_ver3.pt

Recipe identical to p3_est2_vs13.py (30 epochs, batch 16, lr 5e-5,
CPA-jittered 4 s crops, gain jitter, IDMT background mixing); only the
car roster and base ckpt differ.  After training, held-out eval on the
3 evaluation cars is printed and stored in the ckpt.
"""
import argparse
import datetime
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/workspace/CF_ASG/src")
from p3_train import Est2  # noqa: E402
from p3_est2_vs13 import VS13Set, load_vs13 as _unused_load  # noqa: E402,F401
from p3_r_common import load_vs13_items, IDMT  # noqa: E402

CK = Path("/workspace/CF_ASG/p3/ckpt")
SPLIT_SEED = 12
SPLIT_JSON = CK / "vs13_split3.json"


def build_split():
    items = load_vs13_items()
    cars = sorted({it["car"] for it in items})
    assert len(cars) == 13, f"expected 13 vehicles, got {len(cars)}"
    perm = list(np.random.default_rng(SPLIT_SEED).permutation(cars))
    split = {"meta": {"date": str(datetime.date.today()),
                      "script": "src/p3_est2_vs13_split3.py",
                      "seed": SPLIT_SEED,
                      "rule": "default_rng(12).permutation(sorted cars); "
                              "[:5]=editor, [5:10]=verifier, [10:]=eval"},
             "editor_cars": [str(c) for c in perm[:5]],
             "verifier_cars": [str(c) for c in perm[5:10]],
             "eval_cars": [str(c) for c in perm[10:]]}
    SPLIT_JSON.write_text(json.dumps(split, indent=1))
    return split


def get_split():
    if SPLIT_JSON.exists():
        return json.loads(SPLIT_JSON.read_text())
    return build_split()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, choices=("editor", "verifier"))
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(0 if args.role == "editor" else 1)
    dev = "cuda"

    split = get_split()
    print(json.dumps(split, indent=1), flush=True)
    items = load_vs13_items()
    train_cars = set(split[f"{args.role}_cars"])
    tr = [it for it in items if it["car"] in train_cars]
    ev = [it for it in items if it["car"] in set(split["eval_cars"])]
    base = ("est2_latest.pt" if args.role == "editor"
            else "est2_scratch_v.pt")
    out = ("est2_vs13_ed3.pt" if args.role == "editor"
           else "est2_vs13_ver3.pt")
    print(f"role={args.role} base={base} train {len(tr)} clips "
          f"({sorted(train_cars)}), eval {len(ev)} clips "
          f"({split['eval_cars']})", flush=True)

    bgs = sorted(glob.glob(str(IDMT / "*_BG.wav")))[:200]
    model = Est2().to(dev)
    model.load_state_dict(torch.load(CK / base, map_location=dev,
                                     weights_only=False)["model"])
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5,
                            weight_decay=1e-4)
    dl = torch.utils.data.DataLoader(VS13Set(tr, bgs), batch_size=16,
                                     shuffle=True, num_workers=4,
                                     drop_last=True)
    step = 0
    for epoch in range(args.epochs):
        for s, vn in dl:
            s, vn = s.to(dev), vn.to(dev)
            _, pv = model(s)
            loss = torch.abs(pv - vn).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step % 100 == 0:
                print(f"{step} l={float(loss):.4f}", flush=True)
    # held-out eval on the 3 eval cars
    model.eval()
    rels, maes = [], []
    vs = VS13Set(ev, None, train=False)
    with torch.no_grad():
        for i in range(len(vs)):
            s, vn = vs[i]
            _, pv = model(s[None].to(dev))
            v_hat = 5.0 + 35.0 * float(pv[0].clamp(0, 1))
            v_gt = 5.0 + 35.0 * float(vn)
            rels.append(abs(v_hat - v_gt) / v_gt)
            maes.append(abs(v_hat - v_gt) * 3.6)
    ev_stats = {"n_eval": len(rels),
                "rel_median": float(np.median(rels)),
                "mae_kmh_median": float(np.median(maes)),
                "mae_kmh_mean": float(np.mean(maes))}
    torch.save({"model": model.state_dict(), "step": step,
                "role": args.role, "base": base,
                "train_cars": sorted(train_cars),
                "eval_cars": split["eval_cars"],
                "split_seed": SPLIT_SEED,
                "eval_stats": ev_stats}, CK / out)
    print(json.dumps(ev_stats, indent=1))
    print("SAVED", str(CK / out))


if __name__ == "__main__":
    main()
