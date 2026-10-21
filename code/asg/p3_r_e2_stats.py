"""M3: paired statistics for R-E2 -> appended into p3_r_e2_v0.json."""
import json
import numpy as np
from scipy.stats import wilcoxon

P = "/workspace/CF_ASG/paper_numbers/p3_r_e2_v0.json"
d = json.load(open(P))
rows = d["rows"]
clips = {c["clip"]: c for c in d["clips"]}


def series(sysk):
    return {r["clip"]: r["verr_target_kmh"] for r in rows[sysk]}


src = series("source")
stats = {}
for sysk in ("ours", "x5", "x6"):
    sv = series(sysk)
    common = sorted(set(src) & set(sv))
    diffs = [sv[c] - src[c] for c in common]
    stats[f"{sysk}_vs_noedit"] = {
        "n": len(diffs),
        "median_diff_kmh": float(np.median(diffs)),
        "improves_n": int(sum(x < 0 for x in diffs)),
        "wilcoxon_p": float(wilcoxon(diffs).pvalue)}
    for direc, keep in (("up", lambda c: clips[c]["factor"] > 1),
                        ("down", lambda c: clips[c]["factor"] < 1)):
        dd = [sv[c] - src[c] for c in common if keep(c)]
        if len(dd) >= 8:
            stats[f"{sysk}_vs_noedit_{direc}"] = {
                "n": len(dd), "median_diff_kmh": float(np.median(dd)),
                "wilcoxon_p": float(wilcoxon(dd).pvalue)}
ov, x6v = series("ours"), series("x6")
common = sorted(set(ov) & set(x6v))
diffs = [ov[c] - x6v[c] for c in common]
stats["ours_vs_x6"] = {"n": len(diffs),
                       "median_diff_kmh": float(np.median(diffs)),
                       "wilcoxon_p": float(wilcoxon(diffs).pvalue)}
d["stats"] = stats
open(P, "w").write(json.dumps(d, indent=1))
print(json.dumps(stats, indent=1))
