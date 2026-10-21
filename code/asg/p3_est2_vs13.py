"""A11: fine-tune the synthetic-pretrained Est2 on real VS13 pass-bys.
Vehicle-level split (the dataset's own Train_valid_split); 4 s crops
jittered around the labeled CPA; gain jitter; real IDMT-Traffic
background mixed at random SNR.  Eval on held-out vehicles: median
rel err + km/h MAE (reference bar: VS13 paper methods ~4-7 km/h).
-> p3/ckpt/est2_vs13.pt, prints eval metrics
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

sys.path.insert(0, "/workspace/CF_ASG/src")
from p3_train import Est2, SEG  # noqa

VS = Path("/workspace/data/vs13")
IDMT = Path("/workspace/data/idmt_traffic/IDMT_Traffic/audio")
CK = Path("/workspace/CF_ASG/p3/ckpt")
SR = 16000


def load_vs13():
    items = []
    for wav in sorted(glob.glob(str(VS / "**/*.wav"), recursive=True)):
        txt = wav.replace(".wav", ".txt")
        try:
            parts = open(txt).read().split()
            v_kmh, cpa = float(parts[0]), float(parts[1])
        except Exception:
            continue
        car = Path(wav).stem.rsplit("_", 1)[0]
        items.append({"wav": wav, "v": v_kmh / 3.6, "cpa": cpa,
                      "car": car})
    cars = sorted({it["car"] for it in items})
    rng = np.random.default_rng(0)
    rng.shuffle(cars)
    val_cars = set(cars[:3])          # 3 held-out vehicles
    tr = [it for it in items if it["car"] not in val_cars]
    va = [it for it in items if it["car"] in val_cars]
    return tr, va, sorted(val_cars)


def clip_16k(path):
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    return torchaudio.functional.resample(
        torch.tensor(x), sr, SR).numpy()


class VS13Set(torch.utils.data.Dataset):
    def __init__(self, items, bgs, train=True):
        self.items, self.bgs, self.train = items, bgs, train
        self.cache = {}

    def __len__(self):
        return len(self.items) * (8 if self.train else 1)

    def get(self, path):
        if path not in self.cache:
            self.cache[path] = clip_16k(path)
        return self.cache[path]

    def __getitem__(self, i):
        it = self.items[i % len(self.items)]
        x = self.get(it["wav"])
        rng = np.random.default_rng(None if self.train else i)
        jit = rng.uniform(-0.5, 0.5) if self.train else 0.0
        c = int((it["cpa"] + jit) * SR)
        half = SEG // 2
        c = max(half, min(len(x) - half, c))
        seg = x[c - half:c + half].astype(np.float64)
        seg = seg / (np.abs(seg).max() + 1e-9)
        seg = seg * (10 ** (rng.uniform(-12, 0) / 20)
                     if self.train else 0.3)
        if self.train and self.bgs and rng.random() < 0.5:
            bg = self.get(self.bgs[int(rng.integers(len(self.bgs)))])
            if len(bg) >= SEG:
                st = int(rng.integers(len(bg) - SEG + 1))
                b = bg[st:st + SEG].astype(np.float64)
                snr = rng.uniform(5, 25)
                pb = np.sqrt((b ** 2).mean()) + 1e-9
                ps = np.sqrt((seg ** 2).mean()) + 1e-9
                seg = seg + b * (ps / pb) * 10 ** (-snr / 20)
        vn = (it["v"] - 5.0) / 35.0
        return torch.tensor(seg, dtype=torch.float32), \
            torch.tensor(vn, dtype=torch.float32)


def main():
    torch.set_num_threads(8)
    dev = "cuda"
    tr, va, val_cars = load_vs13()
    print(f"train {len(tr)} clips, val {len(va)} (cars: {val_cars})",
          flush=True)
    bgs = sorted(glob.glob(str(IDMT / "*_BG.wav")))[:200]
    model = Est2().to(dev)
    model.load_state_dict(torch.load(CK / "est2_latest.pt",
                                     map_location=dev,
                                     weights_only=False)["model"])
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5,
                            weight_decay=1e-4)
    dl = torch.utils.data.DataLoader(VS13Set(tr, bgs), batch_size=16,
                                     shuffle=True, num_workers=4,
                                     drop_last=True)
    step = 0
    for epoch in range(30):
        for s, vn in dl:
            s, vn = s.to(dev), vn.to(dev)
            _, pv = model(s)
            loss = torch.abs(pv - vn).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step % 100 == 0:
                print(f"{step} l={float(loss):.4f}", flush=True)
    torch.save({"model": model.state_dict(), "step": step,
                "val_cars": val_cars}, CK / "est2_vs13.pt")
    # held-out eval
    model.eval()
    rels, maes = [], []
    vs = VS13Set(va, None, train=False)
    with torch.no_grad():
        for i in range(len(vs)):
            s, vn = vs[i]
            _, pv = model(s[None].to(dev))
            v_hat = 5.0 + 35.0 * float(pv[0].clamp(0, 1))
            v_gt = 5.0 + 35.0 * float(vn)
            rels.append(abs(v_hat - v_gt) / v_gt)
            maes.append(abs(v_hat - v_gt) * 3.6)
    print(json.dumps({
        "n_val": len(rels),
        "rel_median": float(np.median(rels)),
        "mae_kmh_median": float(np.median(maes)),
        "mae_kmh_mean": float(np.mean(maes))}, indent=1))


if __name__ == "__main__":
    main()
