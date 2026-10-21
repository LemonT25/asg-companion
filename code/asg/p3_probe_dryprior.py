"""Probe: line-search theta with the Dryer (stem->dry amortized
inverse) as prior -- the deployable variant of diag D/G/H.
Reports theta rel-err and dry-prior quality vs GT dry.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/workspace/CF_ASG/src")
from p3_train import SceneSet, SceneModel, Dryer, theta_denorm_t, SEG  # noqa
from p3_edit_eval import (si_sdr, mrstft_np, posterior_stems,  # noqa
                          pit_by_sisdr, theta_linesearch)

CK = Path("/workspace/CF_ASG/p3/ckpt")
rng = np.random.default_rng(1)


def main():
    torch.set_num_threads(8)
    dev = "cuda"
    smodel = SceneModel("sep").to(dev)
    smodel.load_state_dict(torch.load(CK / "sep_latest.pt",
                                      map_location=dev,
                                      weights_only=False)["model"])
    smodel.eval()
    dmodel = Dryer().to(dev)
    st = torch.load(CK / "dryer_latest.pt", map_location=dev,
                    weights_only=False)
    dmodel.load_state_dict(st["model"])
    dmodel.eval()
    print("dryer ckpt step:", st["step"])

    ds = SceneSet("val")
    idxs = rng.choice(len(ds), size=200, replace=False)
    rels, dmr, rels_gt = [], [], []
    n_used = 0
    for i in idxs:
        if n_used >= 30:
            break
        y, stems, dries, exist, traj, thn, alpha, env, K = ds[int(i)]
        K = int(K)
        pb = [k for k in range(K) if int(traj[k]) == 0]
        if not pb or float(env) > 0.5:
            continue
        n_used += 1
        yn = y.numpy().astype(np.float64)
        gts = stems.numpy().astype(np.float64)
        gtd = dries.numpy().astype(np.float64)
        with torch.no_grad():
            sepst = smodel.slots(y[None].to(dev))[0][0, ..., :SEG]
        po = posterior_stems(yn, [sepst[k].cpu().numpy().astype(np.float64)
                                  for k in range(3)])
        pp = pit_by_sisdr(po, gts, K)
        for k in pb:
            gt_v = float(theta_denorm_t(traj[k][None], thn[k][None]))
            pk = po[pp[k]]
            with torch.no_grad():
                dp = dmodel(torch.tensor(pk[None, :SEG],
                                         dtype=torch.float32, device=dev)
                            )[0].cpu().numpy().astype(np.float64)
                dpg = dmodel(torch.tensor(gts[k][None, :SEG],
                                          dtype=torch.float32, device=dev)
                             )[0].cpu().numpy().astype(np.float64)
            vh, _ = theta_linesearch(pk, dp)
            rels.append(abs(vh - gt_v) / gt_v)
            vg, _ = theta_linesearch(gts[k], dpg)
            rels_gt.append(abs(vg - gt_v) / gt_v)
            L = min(len(dp), len(gtd[k]))
            dmr.append(mrstft_np(dp[:L], gtd[k][:L]))
    print(json.dumps({
        "n_slots": len(rels),
        "theta_rel_median_po": float(np.median(rels)),
        "theta_rel_median_gtstem": float(np.median(rels_gt)),
        "theta_rel_mean_po": float(np.mean(rels)),
        "dryprior_mrstft_median": float(np.median(dmr)),
    }, indent=1))


if __name__ == "__main__":
    main()
