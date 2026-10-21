"""Graph checkpoint selection: 50k backup vs 60k final, decided by
validation reconstruction (same losses as training, no grad), disclosed.
-> paper_numbers/p3_graph_ckpt_select_v0.json
"""
import datetime
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, "/workspace/CF_ASG/src")
from p3_train import (SceneSet, SceneModel, pit_match, mrstft_loss,  # noqa
                      SEG)

ROOT = Path("/workspace/CF_ASG")
CK = ROOT / "p3" / "ckpt"
PN = ROOT / "paper_numbers"


@torch.no_grad()
def evaluate(ckpt, n=500, batch=12, dev="cuda"):
    model = SceneModel("graph").to(dev)
    st = torch.load(ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(st["model"])
    model.eval()
    ds = SceneSet("val")
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False,
                                     num_workers=4)
    sums, wsum = {}, 0
    seen = 0
    for y, stems, dries, exist, traj, thn, alpha, env, K in dl:
        y, stems, dries = y.to(dev), stems.to(dev), dries.to(dev)
        exist, traj, thn = exist.to(dev), traj.to(dev), thn.to(dev)
        alpha, env = alpha.to(dev), env.to(dev)
        d, e, trj, tn, al, ev = model.slots(y)
        d = d[..., :SEG]
        pstems, ymix = model.compose(d, e, trj, tn, al, env)
        pstems, ymix = pstems[..., :SEG], ymix[..., :SEG]
        idx = pit_match(pstems, stems, exist)
        bi = torch.arange(y.shape[0], device=dev)[:, None]
        ps, pd, ptn = pstems[bi, idx], d[bi, idx], tn[bi, idx]
        m = exist
        L = min(ps.shape[-1], stems.shape[-1])
        sel = m.flatten() > 0
        l_stem = mrstft_loss(ps[..., :L].flatten(0, 1)[sel],
                             stems[..., :L].flatten(0, 1)[sel])
        Ld = min(pd.shape[-1], dries.shape[-1])
        l_dry = mrstft_loss(pd[..., :Ld].flatten(0, 1)[sel],
                            dries[..., :Ld].flatten(0, 1)[sel])
        l_mix = mrstft_loss(ymix[:, :L], y[:, :L])
        mv = m * (traj == 0).float()
        l_th = (torch.abs(ptn - thn) * mv).sum() / mv.sum().clamp(min=1)
        b = y.shape[0]
        for k, v in (("stem", l_stem), ("mix", l_mix), ("dry", l_dry),
                     ("th", l_th)):
            sums[k] = sums.get(k, 0.0) + float(v) * b
        wsum += b
        seen += b
        if seen >= n:
            break
    r = {k: v / wsum for k, v in sums.items()}
    r["recon"] = r["stem"] + r["mix"] + 0.5 * r["dry"]
    r["step"] = int(st["step"])
    r["n_scenes"] = seen
    return r


def main():
    torch.set_num_threads(8)
    cands = {"50k_backup": CK / "graph_50k_backup.pt",
             "60k_final": CK / "graph_latest.pt"}
    res = {k: evaluate(p) for k, p in cands.items()}
    for k, r in res.items():
        print(k, json.dumps(r))
    pick = min(res, key=lambda k: res[k]["recon"])
    out = {"meta": {"date": str(datetime.date.today()),
                    "script": "src/p3_ckpt_select.py",
                    "rule": ("pre-registered: pick lower validation "
                             "reconstruction recon = stem + mix + 0.5*dry "
                             "on first 500 val scenes; th reported, not "
                             "used for selection")},
           "candidates": res, "selected": pick}
    (PN / "p3_graph_ckpt_select_v0.json").write_text(json.dumps(out, indent=1))
    print("SELECTED:", pick)


if __name__ == "__main__":
    main()
