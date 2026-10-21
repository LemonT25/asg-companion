"""A10 gate: est2 (T-F velocity regressor) vs est on ownership
posterior stems.  Target: median rel <= 0.10."""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/workspace/CF_ASG/src")
from p3_train import (SceneSet, SceneModel, Estimator, Est2,  # noqa
                      theta_denorm_t, SEG)
from p3_edit_eval import posterior_stems_own, pit_by_sisdr  # noqa

CK = "/workspace/CF_ASG/p3/ckpt"


def main():
    torch.set_num_threads(8)
    dev = "cuda"
    sm = SceneModel("sep2").to(dev)
    sm.load_state_dict(torch.load(f"{CK}/sep2_latest.pt", map_location=dev,
                                  weights_only=False)["model"])
    sm.eval()
    em = Estimator().to(dev)
    em.load_state_dict(torch.load(f"{CK}/est_latest.pt", map_location=dev,
                                  weights_only=False)["model"])
    em.eval()
    e2 = Est2().to(dev)
    st = torch.load(f"{CK}/est2_latest.pt", map_location=dev,
                    weights_only=False)
    e2.load_state_dict(st["model"])
    e2.eval()
    print("est2 step:", st["step"])

    ds = SceneSet("val")
    rng = np.random.default_rng(7)
    r_est, r_e2 = [], []
    n = 0
    for i in rng.permutation(len(ds)):
        if n >= 40:
            break
        y, stems, dries, exist, traj, thn, alpha, env, K = ds[int(i)]
        K = int(K)
        pb = [k for k in range(K) if int(traj[k]) == 0]
        if not pb or float(env) > 0.5:
            continue
        n += 1
        yn = y.numpy().astype(np.float64)
        gts = stems.numpy().astype(np.float64)
        with torch.no_grad():
            dsep, esep, _, _, _, _ = sm.slots(y[None].to(dev))
            sepst = dsep[0, ..., :SEG].cpu().numpy().astype(np.float64)
            es = torch.sigmoid(esep[0]).cpu()
        po = posterior_stems_own(sm, yn, [sepst[k] for k in range(3)],
                                 es, dev)
        pp = pit_by_sisdr(po, gts, K)
        for k in pb:
            gt_v = float(theta_denorm_t(traj[k][None], thn[k][None]))
            pk = torch.tensor(po[pp[k]][None, :SEG],
                              dtype=torch.float32, device=dev)
            with torch.no_grad():
                _, ptn, _ = em(pk)
                _, pv2 = e2(pk)
            ve = float(theta_denorm_t(torch.tensor([0]),
                                      ptn[0][None].cpu().clamp(0, 1)))
            v2 = float(theta_denorm_t(torch.tensor([0]),
                                      pv2[0][None].cpu().clamp(0, 1)))
            r_est.append(abs(ve - gt_v) / gt_v)
            r_e2.append(abs(v2 - gt_v) / gt_v)
    print(json.dumps({
        "n_slots": len(r_est),
        "est_rel_median": float(np.median(r_est)),
        "est2_rel_median": float(np.median(r_e2)),
        "est_rel_mean": float(np.mean(r_est)),
        "est2_rel_mean": float(np.mean(r_e2))}, indent=1))


if __name__ == "__main__":
    main()
