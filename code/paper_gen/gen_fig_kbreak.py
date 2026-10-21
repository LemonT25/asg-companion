"""Fig: per-K breakdown of ours (S3) across edits, from v1 rows."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "/workspace/CF_ASG/src")
from p3_train import SceneSet  # noqa

PN = Path("/workspace/CF_ASG/paper_numbers")
led = json.loads((PN / "p3_edit_eval_v1.json").read_text())
ds = SceneSet("val")
Kof = {}
for spec in led["edit_specs"]:
    i = int(spec["scene"])
    if i not in Kof:
        Kof[i] = int(ds[i][-1])
edits = ("E1", "E2", "E3", "E4")
colors = {1: "#c7d9ea", 2: "#7fa8cc", 3: "#2b6ca3"}
fig, ax = plt.subplots(figsize=(3.5, 1.5))
xt, xl = [], []
x = 0
for e in edits:
    rows = led["rows"]["S3"][e]
    for K in (1, 2, 3):
        v = [r["sisdr"] for r in rows if Kof[int(r["scene"])] == K]
        if not v:
            continue
        med = float(np.median(v))
        shown = min(med, 24.0)
        ax.bar(x, shown, color=colors[K], width=0.8)
        lab = f"{med:.0f}" if med > 24 else f"{med:.1f}"
        # labels always above the zero line so they never meet the axis
        ax.text(x, max(shown, 0.0) + 0.6, lab, ha="center", va="bottom",
                fontsize=5.5)
        xt.append(x)
        xl.append(f"K{K}")
        x += 1
    x += 0.9
ax.set_xticks(xt)
ax.set_xticklabels(xl, fontsize=6)
ax.set_ylim(-5, 27)
for i, e in enumerate(edits):
    ax.text([1.0, 4.9, 8.8, 12.7][i], 25.2, e, fontsize=7,
            ha="center")
ax.set_ylabel("median SI-SDR (dB)", fontsize=7)
ax.tick_params(axis="y", labelsize=7)
ax.axhline(0, color="k", lw=0.5)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("/workspace/CF_ASG/paper/figs/fig_kbreak.pdf")
print("wrote fig_kbreak.pdf; medians:")
for e in edits:
    rows = led["rows"]["S3"][e]
    print(e, {K: round(float(np.median([r["sisdr"] for r in rows
          if Kof[int(r["scene"])] == K] or [float("nan")])), 1)
          for K in (1, 2, 3)})
