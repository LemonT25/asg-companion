"""A14.5: companion page from the saved waveforms (p3/r2_out, p3/r2_ref,
p3/rset_out).  The listening test is built by p3_r2_listening.py.

Companion page  -> p3/companion/index.html (+ wav + png)
  * index.html: for each edit E1..E4 and chain C1..C4, N_EX scenes:
    original, the edit in words (listening-test wording), ours, ground
    truth; real pass-by speed edits (recording, ours).
  * all_systems.html: the same examples with mixture, reference and all
    seven systems (ZETA, Audio-Omni on the real clips too).
Scene choice: audibility filters of p3_r2_listening.py, then the
scenes with the highest SI-SDR of ours (favourable examples, stated on
the page). The statistics of (paper_numbers/p3_r2_listening_stats.json): every
audible source (within 15 dB of the loudest) must be active at least
40% of the clip and have its class prompt in the CLAP top 4; sources
more than 20 dB down are ignored; the edited nodes must be audible;
pass-by nodes must be vehicle-like; room edits need |delta alpha| >= 0.10
and a transient source. Scenes are then ranked by level balance, and
scenes where the instruction systems were run come first.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, "/workspace/CF_ASG/src")
import p3_r2_common as C  # noqa: E402
from p3_r2_common import SceneSet, load_scene, q_of  # noqa: E402

ROOT = C.ROOT
OUT = ROOT / "p3" / "companion"
SYS = [("ours", "Ours"), ("x1", "AudioSep"), ("x2", "FlowSep"),
       ("x3", "SoloAudio"), ("x4", "OmniSep"), ("x5", "ZETA"),
       ("x6", "Audio-Omni")]
N_EX = 3
GAIN_MIN = 4.0     # dB: our output must be at least this much closer to the truth than the unedited mixture
OURS_MIN = 5.0     # dB: and at least this close to the truth
CHANGE_MAX = 40.0  # dB: an output within this of the mixture is a real change (bit-exact copies reach > 100 dB)
STATS = C.PN / "p3_r2_listening_stats.json"
EDIT_NAMES = {"E1": "E1 remove a source", "E2": "E2 change pass-by speed",
              "E3": "E3 move a static source", "E4": "E4 change room absorption",
              "C1": "C1 = E1 + E3", "C2": "C2 = E2 + E3", "C3": "C3 = E1 + E4",
              "C4": "C4 = E1 + E2"}


def si_sdr(a, b):
    a = a - a.mean(); b = b - b.mean()
    s = (a @ b) / (b @ b + 1e-12) * b
    return float(10 * np.log10((s @ s) / ((a - s) @ (a - s) + 1e-12)))


def mel_png(wav, path, sr=16000):
    import librosa
    S = librosa.feature.melspectrogram(y=wav.astype(np.float32), sr=sr,
                                       n_fft=1024, hop_length=256, n_mels=80)
    fig, ax = plt.subplots(figsize=(3.2, 1.6), dpi=80)
    ax.imshow(10 * np.log10(S + 1e-8), origin="lower", aspect="auto",
              cmap="magma", vmin=-60, vmax=20)
    ax.set_xticks([]); ax.set_yticks([])
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(path); plt.close(fig)


def describe(sc, spec, e):
    cl = [q_of(c) for c in sc["classes"]]
    v = {k: round(C.theta_of(sc, k), 1) for k in range(sc["K"])}
    scene = ", ".join(f"{cl[k]} ({'pass-by %s m/s' % v[k] if C.traj_name(sc, k) == 'passby' else 'static'})"
                      for k in range(sc["K"]))
    room = f"room absorption {sc['alpha']:.2f}" if sc["env"] > 0.5 else "open air"
    if e == "E1":
        d = f"remove the {cl[spec['E1_slot']]}"
    elif e == "E2":
        d = f"{cl[spec['E2_slot']]} passes by at {spec['E2_newv']:.0f} m/s instead of {v[spec['E2_slot']]} m/s"
    elif e == "E3":
        d = f"{cl[spec['E3_slot']]} moved {spec['E3_factor']:g}x farther" if spec["E3_factor"] > 1 else f"{cl[spec['E3_slot']]} moved 2x closer"
    elif e == "E4":
        d = f"room absorption {sc['alpha']:.2f} -> {spec['E4_newa']:.2f}"
    elif e == "C1":
        d = f"remove the {cl[spec['C1_rm']]}; {cl[spec['C1_slot']]} moved {spec['C1_factor']:g}x"
    elif e == "C2":
        d = f"{cl[spec['C2_v_slot']]} at {spec['C2_newv']:.0f} m/s; {cl[spec['C2_r_slot']]} moved {spec['C2_factor']:g}x"
    elif e == "C3":
        d = f"remove the {cl[spec['C3_rm']]}; room absorption -> {spec['C3_newa']:.2f}"
    else:
        d = f"remove the {cl[spec['C4_rm']]}; {cl[spec['C4_v_slot']]} at {spec['C4_newv']:.0f} m/s"
    return f"Scene: {scene}; {room}. Edit: {d}."


def audibility(st, spec, e):
    """Score (higher = better example) or None when the scene is not a
    clean listening example for edit e. Rules as in p3_r2_listening."""
    from p3_r2_listening import LEVEL_DB, QUIET_DB, ACTIVE_MIN, ACTIVE_MAX, CLAP_RANK_MAX, VEHICLE, TRANSIENT
    from p3_r2_listening import fused, exposes_tail, EXCLUDE, LOW_FREQ
    src = st["src"]
    if spec["scene"] in EXCLUDE:
        return None
    if fused(src) and e not in ("E4", "C3"):
        return None
    dom = [k for k, r in enumerate(src) if r["rel"] >= -LEVEL_DB]
    if any(-LEVEL_DB > r["rel"] >= -QUIET_DB for r in src):
        return None
    for k in dom:
        r = src[k]
        if r["clap_rank"] > CLAP_RANK_MAX or r["active"] < ACTIVE_MIN:
            if not (e in ("E4", "C3") and r["cls"] in TRANSIENT and r["clap_rank"] <= CLAP_RANK_MAX):
                return None
        if r["traj"] == "passby" and r["cls"] not in VEHICLE:
            return None
    slots = {"E1": ["E1_slot"], "E2": ["E2_slot"], "E3": ["E3_slot"], "E4": [],
             "C1": ["C1_rm", "C1_slot"], "C2": ["C2_v_slot", "C2_r_slot"],
             "C3": ["C3_rm"], "C4": ["C4_rm", "C4_v_slot"]}[e]
    if any(spec[sl] not in dom for sl in slots):
        return None
    for sl in ("E2_slot", "C2_v_slot", "C4_v_slot"):
        if sl in slots and src[spec[sl]]["active"] < 0.9:
            return None
    if e == "E2" and spec["E2_newv"] > 40:
        return None
    if e in ("E4", "C3"):
        newa = spec["E4_newa"] if e == "E4" else spec["C3_newa"]
        if abs(newa - st["alpha"]) < 0.099:
            return None
        kept = [k for k in dom if not (e == "C3" and k == spec["C3_rm"])]
        if all(src[k]["cls"] in LOW_FREQ for k in kept):
            return None
        ok, frames = exposes_tail(spec["scene"], e, "mix" if e == "E4" else "E1_ref")
        if not ok:
            return None
    if e in ("E1", "E3", "C1", "C3") and len(dom) < 2:
        return None
    # a single-source distance edit is a pure gain in this renderer (ours
    # equals the truth exactly); show two-source scenes where the moved
    # source is the loudest, so the other source has to stay put
    if e == "E3" and src[spec["E3_slot"]]["rel"] < 0:
        return None
    if e in ("E4", "C3"):
        return float(frames)
    return -float(np.std([src[k]["rel"] for k in dom]))


def chain_words(st, sp, e):
    """Plain-English statement of a two-edit chain, same vocabulary as the
    listening test."""
    from p3_r2_listening import scene_words
    cl, _, orig_en, _ = scene_words(st)
    v = [r["v"] for r in st["src"]]

    def move(k, f):
        return (f"the {cl[k]} is moved {f:g} times farther away" if f > 1
                else f"the {cl[k]} is moved 2 times closer")

    def speed(k, v1):
        return (f"the {cl[k]} passes by at {v1:.0f} m/s instead of {v[k]:.0f} m/s, that is "
                f"{'faster' if v1 > v[k] else 'slower'}")

    def room(a1):
        return (f"the room becomes {'more' if a1 > st['alpha'] else 'less'} absorbing "
                f"(absorption {st['alpha']:.2f} to {a1:.2f})")
    if e == "C1":
        d = f"The {cl[sp['C1_rm']]} is removed, and {move(sp['C1_slot'], sp['C1_factor'])}."
    elif e == "C2":
        d = f"{speed(sp['C2_v_slot'], sp['C2_newv'])[0].upper()}{speed(sp['C2_v_slot'], sp['C2_newv'])[1:]}, and {move(sp['C2_r_slot'], sp['C2_factor'])}."
    elif e == "C3":
        d = f"The {cl[sp['C3_rm']]} is removed, and {room(sp['C3_newa'])}."
    else:
        d = f"The {cl[sp['C4_rm']]} is removed, and {speed(sp['C4_v_slot'], sp['C4_newv'])}."
    return orig_en, d


HEAD = ("<!doctype html><html><head><meta charset='utf-8'><title>{title}</title>"
        "<style>body{{font-family:sans-serif;max-width:1100px;margin:auto;padding:12px;line-height:1.4}}"
        "h2{{margin-top:36px;border-bottom:1px solid #ddd}}.ex{{margin:18px 0 28px}}"
        ".ex p{{margin:4px 0}}.row{{display:flex;gap:18px;flex-wrap:wrap}}.cell{{text-align:center}}"
        ".cell b{{display:block;margin-bottom:4px}}.cell img{{display:block;width:260px}}audio{{width:260px}}"
        "table{{border-collapse:collapse;margin-bottom:24px}}td,th{{border:1px solid #ccc;padding:4px;text-align:center;font-size:13px}}"
        "table audio{{width:150px}}table img{{display:block;width:150px}}.d{{font-size:13px;color:#333}}"
        ".edit{{font-weight:bold}}.note{{color:#555;font-size:14px}}</style></head><body>")


def cell(name, label):
    return (f"<div class='cell'><b>{label}</b><img src='png/{name}.png'>"
            f"<audio controls preload='none' src='wav/{name}.wav'></audio></div>")


def main():
    from p3_r2_listening import words
    ds = SceneSet("val")
    led = json.loads((C.PN / "p3_r2_ours.json").read_text())
    specs = {s["scene"]: s for s in led["specs"]}
    present = {k: (C.R2_OUT / k).exists() for k, _ in SYS}
    stats = {int(k): v for k, v in json.loads(STATS.read_text()).items()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "wav").mkdir(exist_ok=True); (OUT / "png").mkdir(exist_ok=True)

    def export(p, name):
        shutil.copy(p, OUT / "wav" / f"{name}.wav")
        w, _ = sf.read(str(p))
        if w.ndim > 1:
            w = w.mean(1)
        mel_png(w, OUT / "png" / f"{name}.png")
        return name

    # ---------------- choose scenes and export files once
    chosen, examples = {}, {}
    for e in C.EDITS:
        cands = []
        for i, spec in specs.items():
            p = C.R2_REF / f"{i:05d}_{e}_ref.wav"
            if not p.exists():
                continue
            need = ["ours"] if e == "E4" else [k for k, _ in SYS[:5] if present[k]]
            if not all((C.R2_OUT / k / f"{i:05d}_{e}.wav").exists() for k in need):
                continue
            sc_ = audibility(stats[i], spec, e)
            if sc_ is None:
                continue
            # the edit must be audible and our result must be a real edit:
            # anchor = SI-SDR of the unedited mixture to the ground truth
            # (doing nothing), ours = SI-SDR of our output, change = SI-SDR
            # of our output to the mixture (very high = output is the
            # mixture, the edit was a no-op). Keep gain = ours - anchor
            # >= GAIN_MIN, ours >= OURS_MIN, change <= CHANGE_MAX; rank by gain.
            m = sf.read(str(C.R2_REF / f"{i:05d}_mix.wav"))[0]
            g = sf.read(str(p))[0]
            o = sf.read(str(C.R2_OUT / "ours" / f"{i:05d}_{e}.wav"))[0]
            anchor, ours, change = si_sdr(m, g), si_sdr(o, g), si_sdr(o, m)
            if ours - anchor < GAIN_MIN or ours < OURS_MIN or change > CHANGE_MAX:
                continue
            cands.append((ours - anchor, i, anchor, ours, change))
        cands.sort(reverse=True)
        print(e, "candidates", [(i, round(gn, 1), round(a, 1), round(o_, 1)) for gn, i, a, o_, _ in cands[:6]])
        chosen[e] = [t[1] for t in cands[:N_EX]]
        examples[e] = []
        for i in chosen[e]:
            st, sp = stats[i], specs[i]
            if e in ("E1", "E2", "E3", "E4"):
                wd = words(st, sp, e)
                orig_en, edit_en = wd["original_en"], wd["edit_en"]
            else:
                orig_en, edit_en = chain_words(st, sp, e)
            files = {"mix": export(C.R2_REF / f"{i:05d}_mix.wav", f"{i:05d}_{e}_mix"),
                     "ref": export(C.R2_REF / f"{i:05d}_{e}_ref.wav", f"{i:05d}_{e}_ref")}
            for k, _ in SYS:
                p = C.R2_OUT / k / f"{i:05d}_{e}.wav"
                if present[k] and p.exists():
                    files[k] = export(p, f"{i:05d}_{e}_{k}")
            examples[e].append({"scene": i, "orig": orig_en, "edit": edit_en,
                                "spec": describe(load_scene(ds, i), sp, e), "files": files})
    # real pass-by edits
    real = []
    re2 = ROOT / "p3" / "rset_out" / "e2"
    man = ROOT / "p3" / "rset_out" / "e2_manifest.json"
    if re2.exists() and man.exists():
        for cc in json.loads(man.read_text())["clips"][:6]:
            j = cc["clip"]
            files = {}
            for k in ("source", "ours", "x5", "x6"):
                p = re2 / k / f"clip_{j:03d}.wav"
                if p.exists():
                    files[k] = export(p, f"real_{j:03d}_{k}")
            if "source" in files and "ours" in files:
                real.append({"text": f"{cc['car']}: the recording was made at {cc['v_true_kmh']:.0f} km/h. "
                                     f"The vehicle passes by at {cc['v_prime_kmh']:.0f} km/h instead.",
                             "files": files})

    # ---------------- page 1: original, edit, ours, reference
    h = [HEAD.format(title="Intervenable Audio Scene Graphs: audio examples"),
         "<h1>Intervenable Audio Scene Graphs: audio examples</h1>",
         "<p class='note'>Each example gives the original recording, the edit in words, our result, and the"
         " ground truth (the edited scene rendered in closed form by the same physical renderer that made the"
         " original). All clips are 4 s at 16 kHz. Spectrograms are mel scale.</p>",
         "<p class='note'>How the examples were chosen: a scene must first pass fixed audibility rules (every named"
         " source is within 15 dB of the loudest and active for at least 40% of the clip; room edits must change the"
         " level in the gaps by more than 3 dB). Second, the edit must matter and our result must be a real edit: our"
         " output must be at least 4 dB closer to the ground truth (SI-SDR) than the unedited mixture, and at least"
         " 5 dB from the mixture itself. Among the scenes that pass, the three shown per edit are the ones with the"
         " largest gain over the unedited mixture. They are therefore favourable examples, not typical ones. The"
         " medians in the paper and the listening-test trials come from the full cohort and from the audibility"
         " rules alone. An edit with no passing scene says so.</p>",
         "<p class='note'>A second page compares <a href='all_systems.html'>all seven systems</a> on the same examples.</p>"]
    for e in C.EDITS:
        h.append(f"<h2>{EDIT_NAMES[e]}</h2>")
        if not examples[e]:
            h.append("<p class='note'>No scene passes the rules above for this edit: our result never gets within 5 dB of the"
                     " ground truth while also improving on the unedited mixture by 4 dB. The paper reports the medians.</p>")
        for ex in examples[e]:
            f = ex["files"]
            h.append(f"<div class='ex'><p>{ex['orig']}</p><p class='edit'>Edit: {ex['edit']}</p><div class='row'>"
                     + cell(f["mix"], "Original") + cell(f["ours"], "Ours") + cell(f["ref"], "Ground truth")
                     + "</div></div>")
    if real:
        h.append("<h2>Real recordings: pass-by speed edits (held-out vehicles)</h2>")
        h.append("<p class='note'>No ground truth exists for a changed speed of a real recording; the paper"
                 " scores these with an independent speed verifier.</p>")
        for ex in real:
            f = ex["files"]
            h.append(f"<div class='ex'><p class='edit'>{ex['text']}</p><div class='row'>"
                     + cell(f["source"], "Recording") + cell(f["ours"], "Ours") + "</div></div>")
    h.append("</body></html>")
    (OUT / "index.html").write_text("\n".join(h))

    # ---------------- page 2: all systems
    g = [HEAD.format(title="All systems: audio examples"),
         "<h1>All systems on the same examples</h1>",
         "<p class='note'><a href='index.html'>Back to the main page.</a> Mixture = the original recording."
         " Reference = the closed-form ground truth. Separation-family systems (AudioSep, FlowSep, SoloAudio,"
         " OmniSep) receive the same estimation heads, inverses and edit identities as ours; instruction"
         " systems (ZETA, Audio-Omni) receive a text instruction (one combined instruction for chains).</p>"]
    for e in C.EDITS:
        g.append(f"<h2>{EDIT_NAMES[e]}</h2>")
        cols = ["Mixture", "Reference"] + [n for k, n in SYS if present[k]]
        g.append("<table><tr><th>scene</th>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>")
        for ex in examples[e]:
            f = ex["files"]
            cells = []
            for tag in ["mix", "ref"] + [k for k, _ in SYS if present[k]]:
                if tag in f:
                    cells.append(f"<td><img src='png/{f[tag]}.png'><audio controls preload='none' src='wav/{f[tag]}.wav'></audio></td>")
                else:
                    cells.append("<td>n/a</td>")
            g.append(f"<tr><td class='d' style='max-width:220px;text-align:left'>{ex['spec']}</td>" + "".join(cells) + "</tr>")
        g.append("</table>")
    if real:
        g.append("<h2>Real recordings: pass-by speed edits</h2>")
        rsys = [("source", "Recording"), ("ours", "Ours"), ("x5", "ZETA"), ("x6", "Audio-Omni")]
        g.append("<table><tr><th>clip</th>" + "".join(f"<th>{n}</th>" for _, n in rsys) + "</tr>")
        for ex in real:
            f = ex["files"]
            cells = [(f"<td><img src='png/{f[k]}.png'><audio controls preload='none' src='wav/{f[k]}.wav'></audio></td>"
                      if k in f else "<td>n/a</td>") for k, _ in rsys]
            g.append(f"<tr><td class='d'>{ex['text']}</td>" + "".join(cells) + "</tr>")
        g.append("</table>")
    g.append("</body></html>")
    (OUT / "all_systems.html").write_text("\n".join(g))
    print("companion page:", OUT / "index.html", "scenes", {e: chosen[e] for e in chosen})


if __name__ == "__main__":
    main()
