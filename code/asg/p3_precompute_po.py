"""Precompute ownership-posterior stems for the TRAIN split (A10):
est2b trains directly on deployment-domain inputs.  Stems stored
PIT-matched to GT slot order (labels align).  float16, ~9 GB."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/workspace/CF_ASG/src")
from p3_train import SceneSet, SceneModel, SEG  # noqa
from p3_edit_eval import posterior_stems_own, pit_by_sisdr  # noqa

OUT = Path("/workspace/CF_ASG/p3/po_train")
OUT.mkdir(exist_ok=True)


def main():
    torch.set_num_threads(8)
    dev = "cuda"
    sm = SceneModel("sep2").to(dev)
    sm.load_state_dict(torch.load(
        "/workspace/CF_ASG/p3/ckpt/sep2_latest.pt", map_location=dev,
        weights_only=False)["model"])
    sm.eval()
    ds = SceneSet("train")
    for i in range(min(10000, len(ds))):
        op = OUT / f"{i:06d}.npz"
        if op.exists():
            continue
        y, stems, dries, exist, traj, thn, alpha, env, K = ds[i]
        K = int(K)
        yn = y.numpy().astype(np.float64)
        gts = stems.numpy().astype(np.float64)
        with torch.no_grad():
            dsep, esep, _, _, _, _ = sm.slots(y[None].to(dev))
            sepst = dsep[0, ..., :SEG].cpu().numpy().astype(np.float64)
            es = torch.sigmoid(esep[0]).cpu()
        po = posterior_stems_own(sm, yn, [sepst[k] for k in range(3)],
                                 es, dev)
        pp = pit_by_sisdr(po, gts, K)
        arr = np.stack([po[pp[k]] for k in range(K)]
                       + [np.zeros(SEG)] * (3 - K)).astype(np.float16)
        np.savez(op, po=arr, K=K)
        if (i + 1) % 2000 == 0:
            print(f"{i+1}/{len(ds)}", flush=True)
    print("PRECOMPUTE DONE")


if __name__ == "__main__":
    main()
