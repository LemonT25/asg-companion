"""Confirmation cohort (conf1) re-scored under the revision rule that E3
is evaluated on scenes with two or more sources only (single-source
distance edits are pure gains, invisible to SI-SDR). Everything else
(E1, E2) is unchanged from p3_confirm_verdict_v0.json.
-> paper_numbers/p3_r2_confirm_k2.json
Run with /workspace/venvs/audiocf python.
"""
import json
import sys

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, "/workspace/CF_ASG/src")
from p3_train import SceneSet  # noqa: E402

PN = "/workspace/CF_ASG/paper_numbers"
XFILES = {"x1": ("p3_baselines_x1_c1.json", "S5"),
          "x2": ("p3_baselines_x2_c1.json", "X2"),
          "x3": ("p3_baselines_x3_c1.json", "X3"),
          "x4": ("p3_baselines_x4_c1.json", "X4")}


def holm(ps):
    idx = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(idx):
        run = max(run, (m - r) * ps[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def main():
    ds = SceneSet("conf1")
    ours = json.load(open(f"{PN}/p3_edit_eval_c1.json"))["rows"]["S3"]
    out = {"meta": {"cohort": "conf1", "rule": "E3 on K>=2 only (revision)",
                    "source": "p3_edit_eval_c1.json, p3_baselines_x*_c1.json"},
           "E3": {}}
    K = {}

    def k_of(i):
        if i not in K:
            K[i] = int(ds[int(i)][-1])
        return K[i]
    o = {r["scene"]: r["sisdr"] for r in ours["E3"] if k_of(r["scene"]) >= 2}
    out["E3"]["ours"] = {"median": float(np.median(list(o.values()))),
                         "n": len(o)}
    out["E3"]["ours_all"] = {"median": float(np.median(
        [r["sisdr"] for r in ours["E3"]])), "n": len(ours["E3"])}
    res, ps, ks = {}, [], []
    for k, (f, key) in XFILES.items():
        x = {r["scene"]: r["sisdr"] for r in
             json.load(open(f"{PN}/{f}"))[key]["rows"]["E3"]
             if k_of(r["scene"]) >= 2}
        com = sorted(set(o) & set(x))
        a = np.array([o[s] for s in com])
        b = np.array([x[s] for s in com])
        p = float(wilcoxon(a, b).pvalue)
        res[k] = {"median": float(np.median(b)), "n": len(com),
                  "median_diff": float(np.median(a - b)), "p": p}
        ps.append(p)
        ks.append(k)
    for k, pa in zip(ks, holm(ps)):
        res[k]["p_holm"] = pa
    out["E3"]["vs"] = res
    json.dump(out, open(f"{PN}/p3_r2_confirm_k2.json", "w"), indent=1)
    print(json.dumps(out["E3"], indent=1))


if __name__ == "__main__":
    main()
