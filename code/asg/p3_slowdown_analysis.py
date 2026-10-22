"""A13.2: why do slow-down edits fail while speed-ups succeed?
Correlates per-clip slow-down verifier error with candidate causes,
and probes the verifier's own response to synthetically re-sped
unedited clips.  -> paper_numbers/p3_slowdown_analysis_v0.json
and paper/figs/fig_slowdown.pdf
"""
import datetime
import json
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, "/workspace/CF_ASG/src")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PN = "/workspace/CF_ASG/paper_numbers"


def main():
    d = json.load(open(f"{PN}/p3_r_e2_v1.json"))
    clips = {c["clip"]: c for c in d["clips"]}
    src = {r["clip"]: r for r in d["rows"]["source"]}
    ours = {r["clip"]: r for r in d["rows"]["ours"]}
    rows = []
    for cid, r in ours.items():
        c = clips[cid]
        rows.append({
            "clip": cid, "up": c["factor"] > 1,
            "v_true": c["v_true"] * 3.6,
            "v_prime": c["v_prime"] * 3.6,
            "verr_edit": r["verr_target_kmh"],
            "verr_src": src[cid]["verr_target_kmh"],
            "editor_err": r.get("editor_err_kmh",
                                r.get("v_hat_err_kmh")),
        })
    up = [r for r in rows if r["up"]]
    dn = [r for r in rows if not r["up"]]

    def corr(rs, key):
        xs = [r[key] for r in rs if r[key] is not None]
        ys = [r["verr_edit"] for r in rs if r[key] is not None]
        if len(xs) < 8:
            return None
        rho, p = spearmanr(xs, ys)
        return {"rho": float(rho), "p": float(p), "n": len(xs)}

    out = {"meta": {"date": str(datetime.date.today()),
                    "script": "src/p3_slowdown_analysis.py",
                    "ledger": "p3_r_e2_v1.json"},
           "medians": {
               "up_edit": float(np.median([r["verr_edit"] for r in up])),
               "dn_edit": float(np.median([r["verr_edit"] for r in dn])),
               "up_src": float(np.median([r["verr_src"] for r in up])),
               "dn_src": float(np.median([r["verr_src"] for r in dn]))},
           "correlations_dn": {
               "v_true": corr(dn, "v_true"),
               "editor_err": corr(dn, "editor_err"),
               "verr_src": corr(dn, "verr_src")},
           "correlations_up": {
               "v_true": corr(up, "v_true"),
               "editor_err": corr(up, "editor_err")}}

    # verifier response probe: re-speed UNEDITED clips by known factors
    # with GT v (no estimation involved) -> does the verifier track?
    from p3_train import Est2, theta_denorm_t, SEG  # noqa: E402
    from p3_edit_eval import inverse_passby, render_node  # noqa: E402
    import soundfile as sf
    import torchaudio
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ver = Est2().to(dev)
    ck = torch.load("/workspace/CF_ASG/p3/ckpt/est2_vs13_ver3.pt",
                    map_location=dev, weights_only=False)
    ver.load_state_dict(ck["model"])
    ver.eval()
    probe = {"up": [], "dn": []}
    for cid, c in list(clips.items())[:30]:
        x, sr = sf.read(c["wav"], dtype="float32")
        if x.ndim > 1:
            x = x.mean(1)
        x = torchaudio.functional.resample(
            torch.tensor(x), sr, 16000).numpy()
        cc = int(c["cpa"] * 16000)
        half = SEG // 2
        cc = max(half, min(len(x) - half, cc))
        seg = (x[cc - half:cc + half] /
               (np.abs(x[cc - half:cc + half]).max() + 1e-9) * 0.3
               ).astype(np.float64)
        for fac, tag in ((1.3, "up"), (0.7, "dn")):
            vt = c["v_true"]
            dry = inverse_passby(seg, vt)          # GT-v inversion
            ed = render_node(dry, "passby", vt * fac, 0.0, 0.0)
            with torch.no_grad():
                _, pv = ver(torch.tensor(ed[None, :SEG],
                                         dtype=torch.float32,
                                         device=dev))
            vh = float(theta_denorm_t(torch.tensor([0]),
                                      pv[0][None].cpu().clamp(0, 1)))
            probe[tag].append(abs(vh - vt * fac) * 3.6)
    out["gt_v_render_probe"] = {
        "note": ("edits executed with GROUND-TRUTH v (no estimator): "
                 "isolates render+verifier from estimation error"),
        "up_median_kmh": float(np.median(probe["up"])),
        "dn_median_kmh": float(np.median(probe["dn"])),
        "n": len(probe["up"])}

    open(f"{PN}/p3_slowdown_analysis_v0.json", "w").write(
        json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "meta"},
                     indent=1))

    # figure: per-clip scatter
    fig, ax = plt.subplots(figsize=(3.5, 1.6))
    for rs, col, lab, mk in ((up, "#2b6ca3", "speed-up", "o"),
                             (dn, "#c04a3a", "slow-down", "s")):
        ax.scatter([r["verr_src"] for r in rs],
                   [r["verr_edit"] for r in rs], s=12, c=col,
                   label=lab, marker=mk, alpha=0.75, linewidths=0)
    lim = 60
    ax.plot([0, lim], [0, lim], "k--", lw=0.6)
    ax.set_xlabel("verifier error, unedited (km/h)", fontsize=7)
    ax.set_ylabel("verifier error, edited", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("/workspace/CF_ASG/paper/figs/fig_slowdown.pdf")
    print("wrote fig_slowdown.pdf")


if __name__ == "__main__":
    main()
