"""M5: held-out theta rel-err of the three estimator generations on
ownership posterior stems -> p3_theta_estimators_v0.json"""
import datetime
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/workspace/CF_ASG/src")
from p3_train import (SceneSet, SceneModel, Estimator, Est2,  # noqa
                      theta_denorm_t, SEG)
from p3_train_est2b import Est2B  # noqa
from p3_edit_eval import posterior_stems_own, pit_by_sisdr  # noqa

CK = "/workspace/CF_ASG/p3/ckpt"
PN = "/workspace/CF_ASG/paper_numbers"


def main():
    torch.set_num_threads(8)
    dev = "cuda"
    sm = SceneModel("sep2").to(dev)
    sm.load_state_dict(torch.load(f"{CK}/sep2_latest.pt",
                                  map_location=dev,
                                  weights_only=False)["model"])
    sm.eval()
    em = Estimator().to(dev)
    em.load_state_dict(torch.load(f"{CK}/est_latest.pt",
                                  map_location=dev,
                                  weights_only=False)["model"])
    em.eval()
    e2 = Est2().to(dev)
    e2.load_state_dict(torch.load(f"{CK}/est2_latest.pt",
                                  map_location=dev,
                                  weights_only=False)["model"])
    e2.eval()
    eb = Est2B().to(dev)
    eb.load_state_dict(torch.load(f"{CK}/est2b_latest.pt",
                                  map_location=dev,
                                  weights_only=False)["model"])
    eb.eval()

    ds = SceneSet("val")
    rng = np.random.default_rng(7)
    r = {"est": [], "est2": [], "est2b": []}
    n = 0
    for i in rng.permutation(len(ds)):
        if n >= 80:
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
        yt = torch.tensor(yn[None, :SEG], dtype=torch.float32,
                          device=dev)
        for k in pb:
            gt_v = float(theta_denorm_t(traj[k][None], thn[k][None]))
            pk = torch.tensor(po[pp[k]][None, :SEG],
                              dtype=torch.float32, device=dev)
            with torch.no_grad():
                _, ptn, _ = em(pk)
                _, pv2 = e2(pk)
                _, pvb = eb(pk, yt)
                vb = float(Est2B.decode_v(pvb)[0])
            for tag, vn in (("est", float(ptn[0].clamp(0, 1))),
                            ("est2", float(pv2[0].clamp(0, 1))),
                            ("est2b", vb)):
                vh = float(theta_denorm_t(
                    torch.tensor([0]), torch.tensor([vn]).clamp(0, 1)))
                r[tag].append(abs(vh - gt_v) / gt_v)
    out = {"meta": {"date": str(datetime.date.today()),
                    "script": "src/p3_theta_estimators.py",
                    "note": ("three estimator generations on ownership "
                             "posterior stems, held-out open scenes; "
                             "est=waveform conv (est_latest 3.4M), "
                             "est2=T-F conv (est2_latest 0.9M), "
                             "est2b=T-F 2ch+bins (est2b_latest 1.6M)"),
                    "n_scenes": n, "n_slots": len(r["est"])},
           "rel_median": {t: float(np.median(v)) for t, v in r.items()},
           "rel_mean": {t: float(np.mean(v)) for t, v in r.items()}}
    open(f"{PN}/p3_theta_estimators_v0.json", "w").write(
        json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("rel_median", "rel_mean")},
                     indent=1))


if __name__ == "__main__":
    main()
