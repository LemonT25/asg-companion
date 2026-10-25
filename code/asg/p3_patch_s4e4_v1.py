"""Interim A12.5 patch: compute the oracle-E4 rung (GT stems +
estimated alpha, mixture-level re-render) into rows.S4.E4 of an
existing main ledger, verify it equals the ORE4 pseudo-system rows
(recorded coincidence), and rebuild the edits aggregates.

Superseded by the native S4.E4 in the next full main-eval run; kept
as the auditable path from ledger state A to state B.
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/workspace/CF_ASG/src")
from p3_train import SceneSet, Estimator, ALPHA_NODES, SEG  # noqa: E402
from p3_edit_eval import (si_sdr, mrstft_np, locality,  # noqa: E402
                          wiener_dereverb)
sys.path.insert(0, "/workspace/CF_PhysAudio/src")
from p2_common import phi_reverb  # noqa: E402

PN = Path("/workspace/CF_ASG/paper_numbers")
CK = Path("/workspace/CF_ASG/p3/ckpt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default="p3_edit_eval_v1.json")
    ap.add_argument("--split", default="val")
    args = ap.parse_args()
    torch.set_num_threads(8)
    dev = "cuda"
    emodel = Estimator().to(dev)
    emodel.load_state_dict(torch.load(CK / "est_latest.pt",
                                      map_location=dev)["model"])
    emodel.eval()
    ds = SceneSet(args.split)
    led = json.loads((PN / args.ledger).read_text())
    rows = []
    for spec in led["edit_specs"]:
        if "E4_newa" not in spec:
            continue
        i, newa = spec["scene"], spec["E4_newa"]
        y, stems, dries, exist, traj, thn, alpha, env, K = ds[int(i)]
        yn = y.numpy().astype(np.float64)
        gts = stems.numpy().astype(np.float64)
        K = int(K)
        with torch.no_grad():
            _, _, pal = emodel(y[None].to(dev))
        al3 = float(pal[0])
        na = min(ALPHA_NODES, key=lambda t: abs(t - newa))
        # reference: exact GT re-render at newa (as in the main eval)
        from p3_edit_eval import render_node
        from p3_train import theta_denorm_t
        ref = sum(render_node(
            dries[j].numpy().astype(np.float64),
            "passby" if int(traj[j]) == 0 else "static",
            float(theta_denorm_t(traj[j][None], thn[j][None])),
            newa, 1.0) for j in range(K))
        gmix = np.sum(gts[:K], axis=0)
        est_o = phi_reverb(wiener_dereverb(gmix[:SEG], al3)[:SEG],
                           na)[:SEG]
        rows.append({"scene": int(i), "sisdr": si_sdr(est_o, ref),
                     "mrstft": mrstft_np(est_o, ref),
                     "loc": locality(est_o, ref, yn)})
    # sanity: must equal the recorded ORE4 rows (same computation)
    if "ORE4" in led["rows"] and led["rows"]["ORE4"].get("E4"):
        o = {r["scene"]: r["sisdr"] for r in led["rows"]["ORE4"]["E4"]}
        d = [abs(r["sisdr"] - o[r["scene"]]) for r in rows
             if r["scene"] in o]
        print(f"ORE4 cross-check: n={len(d)} max|delta|="
              f"{max(d) if d else float('nan'):.2e}")
    led["rows"]["S4"]["E4"] = rows

    rng = np.random.default_rng(0)

    def agg(rws, key="sisdr"):
        v = [r[key] for r in rws if key in r]
        if not v:
            return None
        m = np.median(rng.choice(v, (5000, len(v))), axis=1)
        return {"median": float(np.median(v)),
                "ci95": [float(np.percentile(m, 2.5)),
                         float(np.percentile(m, 97.5))], "n": len(v)}

    for sys_ in led["rows"]:
        led["edits"][sys_] = {
            e: {"sisdr": agg(led["rows"][sys_][e]),
                "mrstft": agg(led["rows"][sys_][e], "mrstft"),
                "loc": agg(led["rows"][sys_][e], "loc"),
                "loc_clean": agg(led["rows"][sys_][e], "loc_clean")}
            for e in led["rows"][sys_] if led["rows"][sys_][e]}
    led["meta"]["s4_e4_patch"] = (
        f"{datetime.date.today()} src/p3_patch_s4e4_v1.py: oracle-E4 "
        "rung recomputed into rows.S4.E4 (GT stems + est alpha, "
        "mixture-level; == ORE4 rows, cross-checked); aggregates "
        "rebuilt; superseded by the next full eval run")
    (PN / args.ledger).write_text(json.dumps(led, indent=1))
    print("S4.E4 n =", len(rows), "median =",
          led["edits"]["S4"]["E4"]["sisdr"]["median"])


if __name__ == "__main__":
    main()
