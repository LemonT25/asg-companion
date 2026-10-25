"""Fig 2: the E2 ladder, from ledgers. -> paper/figs/fig_ladder.pdf"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PN = Path("/workspace/CF_ASG/paper_numbers")
led = json.loads((PN / "p3_edit_eval_v1.json").read_text())
noop = json.loads((PN / "p3_noop_anchor_v1.json").read_text())
ver = json.loads((PN / "p3_sixsys_verdict_v1.json").read_text())
# separation-family rungs from the replayed A14 result files (same heads
# as Table III: p3_r2_<system>.json), instruction editors from the verdict
import numpy as np
r2 = {k: float(np.median([r["sisdr"] for r in json.loads((PN / f"p3_r2_{k}.json").read_text())["rows"]["E2"]]))
      for k in ("ours", "x1", "x2", "x3", "x4")}

rungs = [
    ("Instruction editors\n(native attempt)",
     max(ver["vs"]["X5"]["E2"]["x_median"],
         ver["vs"]["X6"]["E2"]["x_median"]), "#b8b8b8"),
    ("No edit", noop["median"]["E2"], "#b8b8b8"),
    ("OmniSep + our machinery", r2["x4"], "#a8c8e0"),
    ("FlowSep + our machinery", r2["x2"], "#a8c8e0"),
    ("SoloAudio + our machinery", r2["x3"], "#a8c8e0"),
    ("AudioSep + our machinery", r2["x1"], "#8fb8d8"),
    ("Ours", r2["ours"], "#2b6ca3"),
    ("Oracle stems\n+ our estimator",
     led["edits"]["S4"]["E2"]["sisdr"]["median"], "#69a860"),
]
fig, ax = plt.subplots(figsize=(3.5, 2.35))
ys = range(len(rungs))
vals = [r[1] for r in rungs]
ax.barh(list(ys), vals, color=[r[2] for r in rungs], height=0.62)
ax.set_yticks(list(ys))
ax.set_yticklabels([r[0] for r in rungs], fontsize=7)
ax.set_xlabel("E2 median SI-SDR to reference (dB)", fontsize=7)
ax.tick_params(axis="x", labelsize=7)
ax.axvline(0, color="k", lw=0.6)
for y, v in zip(ys, vals):
    ax.text(v + (0.6 if v >= 0 else -0.6), y, f"{v:.1f}",
            va="center", ha="left" if v >= 0 else "right", fontsize=7)
ax.set_xlim(-47, 9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("/workspace/CF_ASG/paper/figs/fig_ladder.pdf")
print("wrote fig_ladder.pdf")
