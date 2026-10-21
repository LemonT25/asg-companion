"""No-edit anchor: the unmodified mixture scored against each edit
reference, over the main ledger's specs.  A12.5 adds the E4 anchor
(mixture vs the newa reference re-render).
-> paper_numbers/p3_noop_anchor_v0.json (or --out)"""
import argparse
import datetime
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/workspace/CF_ASG/src")
from p3_train import SceneSet, theta_denorm_t  # noqa
from p3_edit_eval import si_sdr, render_node  # noqa

PN = "/workspace/CF_ASG/paper_numbers"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default="p3_edit_eval_v0.json")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default="p3_noop_anchor_v0.json")
    ap.add_argument("--k2", action="store_true",
                    help="revision rule: E3 anchor on K>=2 scenes only")
    args = ap.parse_args()
    led = json.load(open(f"{PN}/{args.ledger}"))
    ds = SceneSet(args.split)
    noop = {"E1": [], "E2": [], "E3": [], "E4": []}
    for spec in led["edit_specs"]:
        i = spec["scene"]
        y, stems, dries, exist, traj, thn, alpha, env, K = ds[int(i)]
        yn = y.numpy().astype(np.float64)
        gts = stems.numpy().astype(np.float64)
        gtd = dries.numpy().astype(np.float64)
        alpha, env, K = float(alpha), float(env), int(K)
        if "E1_slot" in spec:
            k = spec["E1_slot"]
            noop["E1"].append(si_sdr(yn, yn - gts[k]))
        if "E2_slot" in spec:
            k, newv = spec["E2_slot"], spec["E2_newv"]
            ref = yn - gts[k] + render_node(
                gtd[k], "passby", newv, alpha, env)
            noop["E2"].append(si_sdr(yn, ref))
        if "E3_slot" in spec and not (args.k2 and K < 2):
            k, fm = spec["E3_slot"], spec["E3_factor"]
            noop["E3"].append(si_sdr(yn, yn - gts[k] + gts[k] / fm))
        if "E4_newa" in spec:      # A12.5 E4 anchor (mixture vs newa ref)
            newa = spec["E4_newa"]
            ref = sum(render_node(
                gtd[j], "passby" if int(traj[j]) == 0 else "static",
                float(theta_denorm_t(traj[j][None], thn[j][None])),
                newa, 1.0) for j in range(K))
            noop["E4"].append(si_sdr(yn, ref))
    out = {"meta": {"date": str(datetime.date.today()),
                    "script": "src/p3_noop_anchor.py",
                    "split": args.split,
                    "ledger": args.ledger, "k2_rule": bool(args.k2),
                    "E4_anchor": ("A12.5/M7: unmodified mixture vs the "
                                  "GT re-render at the commanded newa")},
           "median": {e: float(np.median(v)) for e, v in noop.items()
                      if v},
           "n": {e: len(v) for e, v in noop.items()}}
    open(f"{PN}/{args.out}", "w").write(json.dumps(out, indent=1))
    print(json.dumps(out["median"], indent=1))


if __name__ == "__main__":
    main()
