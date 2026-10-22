"""A7 pre-registered gate: does the Whittle marginal likelihood
identify theta?

G1  K=1 open pass-by GT scenes: v by full-grid marginal likelihood,
    moment-init W per candidate (no oracle theta) -> rel err (gate: median <= 0.05).
G2  K=2 open scenes (passby+static): coordinate EM + v grid.
G3  room static scenes: alpha by likelihood model selection over the
    node grid; env by likelihood comparison.
-> paper_numbers/p3_pwg_gate_v0.json
"""
import datetime
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/workspace/CF_ASG/src")
from p3_train import SceneSet, theta_denorm_t  # noqa: E402
import p3_pwg as pwg  # noqa: E402

PN = "/workspace/CF_ASG/paper_numbers/p3_pwg_gate_v1.json"
rng = np.random.default_rng(3)


def flat_slot(T, kind, v):
    W = torch.ones(pwg.F)
    c = torch.ones(T)
    return pwg.Slot(W, c, kind, v)


def main():
    torch.set_num_threads(8)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rk = pwg.RoomKernel(dev=dev, max_tau=80)
    ds = SceneSet("val")
    v_grid = np.linspace(5, 40, 29)
    out = {"G1": [], "G2": [], "G3_alpha": [], "G3_env": []}
    n1 = n2 = n3 = 0
    for i in rng.permutation(len(ds)):
        if n1 >= 15 and n2 >= 10 and n3 >= 10:
            break
        y, stems, dries, exist, traj, thn, alpha, env, K = ds[int(i)]
        K = int(K)
        pb = [k for k in range(K) if int(traj[k]) == 0]
        yn = y.numpy().astype(np.float64)
        P, _ = pwg.stft_pow(yn, dev)
        T = P.shape[-1]

        if K == 1 and pb and float(env) < 0.5 and n1 < 15:
            n1 += 1
            gt_v = float(theta_denorm_t(traj[0][None], thn[0][None]))
            lls = []
            for v in v_grid:
                W0 = pwg.unwarp_mean_W(P, float(v), dev)
                slots = [pwg.Slot(W0, torch.ones(T), "passby",
                                  float(v))]
                ll, _ = pwg.em_fit(P, slots, 0.0, 0.0, rk, None, dev,
                                   iters=12)
                lls.append(ll)
            vh = float(v_grid[int(np.argmax(lls))])
            out["G1"].append({"gt": gt_v, "vh": vh,
                              "rel": abs(vh - gt_v) / gt_v})
            print(f"G1 scene {i}: gt {gt_v:.1f} vh {vh:.1f}", flush=True)

        elif K == 2 and len(pb) == 1 and float(env) < 0.5 and n2 < 10:
            n2 += 1
            kpb = pb[0]
            gt_v = float(theta_denorm_t(traj[kpb][None], thn[kpb][None]))
            slots = [flat_slot(T, "passby", 20.0), flat_slot(T, "static", 0)]
            ll, slots = pwg.em_fit(P, slots, 0.0, 0.0, rk, None, dev,
                                   iters=8)
            ll, vh, slots = pwg.fit_v(P, slots, 0, v_grid, 0.0, 0.0, rk,
                                      1e-8, dev, iters=4)
            out["G2"].append({"gt": gt_v, "vh": vh,
                              "rel": abs(vh - gt_v) / gt_v})
            print(f"G2 scene {i}: gt {gt_v:.1f} vh {vh:.1f}", flush=True)

        elif float(env) > 0.5 and K <= 2 and n3 < 10:
            n3 += 1
            slots = [flat_slot(T, "static", 0) for _ in range(K)]
            lls = []
            for a in pwg.ALPHA_NODES:
                import copy
                tr = copy.deepcopy(slots)
                ll, _ = pwg.em_fit(P, tr, float(a), 1.0, rk, None, dev,
                                   iters=6)
                lls.append(ll)
            ah = float(pwg.ALPHA_NODES[int(np.argmax(lls))])
            out["G3_alpha"].append({"gt": float(alpha), "ah": ah,
                                    "abserr": abs(ah - float(alpha))})
            # env selection: best room ll vs open ll
            import copy
            tr = copy.deepcopy(slots)
            ll_open, _ = pwg.em_fit(P, tr, 0.0, 0.0, rk, None, dev, iters=6)
            out["G3_env"].append({"gt": 1.0,
                                  "pred": 1.0 if max(lls) > ll_open else 0.0})
            print(f"G3 scene {i}: gt_a {float(alpha):.2f} ah {ah:.2f} "
                  f"env {'ok' if max(lls) > ll_open else 'MISS'}",
                  flush=True)

    res = {"meta": {"date": str(datetime.date.today()),
                    "script": "src/p3_pwg_gate.py",
                    "config": ("v1: moment-init W (unwarp at candidate v), "
                               "Nyquist edge-extension, 12 EM iters; "
                               "v0 was flat-init pre-moment"),
                    "gate": "G1 median rel <= 0.05"},
           "G1_rel_median": float(np.median([r["rel"] for r in out["G1"]]))
           if out["G1"] else None,
           "G2_rel_median": float(np.median([r["rel"] for r in out["G2"]]))
           if out["G2"] else None,
           "G3_alpha_abserr_median": float(np.median(
               [r["abserr"] for r in out["G3_alpha"]]))
           if out["G3_alpha"] else None,
           "G3_env_acc": float(np.mean(
               [r["pred"] == r["gt"] for r in out["G3_env"]]))
           if out["G3_env"] else None,
           "rows": out}
    open(PN, "w").write(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "rows"},
                     indent=1))


if __name__ == "__main__":
    main()
