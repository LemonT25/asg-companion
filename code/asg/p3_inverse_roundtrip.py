"""R2-16: round-trip accuracy of the semi-analytic inverses on
ground-truth stems -> p3_inverse_roundtrip_v0.json"""
import datetime
import json
import sys

import numpy as np

sys.path.insert(0, "/workspace/CF_ASG/src")
from p3_train import SceneSet, theta_denorm_t, SEG  # noqa
from p3_edit_eval import (si_sdr, inverse_passby, render_node)  # noqa

PN = "/workspace/CF_ASG/paper_numbers"


def main():
    ds = SceneSet("val")
    rng = np.random.default_rng(5)
    vals = []
    n = 0
    for i in rng.permutation(len(ds)):
        if n >= 60:
            break
        y, stems, dries, exist, traj, thn, alpha, env, K = ds[int(i)]
        K = int(K)
        pb = [k for k in range(K) if int(traj[k]) == 0]
        if not pb or float(env) > 0.5:
            continue
        n += 1
        for k in pb:
            gt_v = float(theta_denorm_t(traj[k][None], thn[k][None]))
            stem = stems[k].numpy().astype(np.float64)
            d = inverse_passby(stem, gt_v)
            re = render_node(d, "passby", gt_v, 0.0, 0.0)
            L = min(len(re), len(stem))
            vals.append(si_sdr(re[:L], stem[:L]))
    out = {"meta": {"date": str(datetime.date.today()),
                    "script": "src/p3_inverse_roundtrip.py",
                    "def": ("render(inverse_passby(GT wet stem, GT v), "
                            "GT v) vs the GT wet stem, open scenes")},
           "sisdr_median": float(np.median(vals)),
           "sisdr_p10": float(np.percentile(vals, 10)),
           "sisdr_p90": float(np.percentile(vals, 90)),
           "n_slots": len(vals)}
    open(f"{PN}/p3_inverse_roundtrip_v0.json", "w").write(
        json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
