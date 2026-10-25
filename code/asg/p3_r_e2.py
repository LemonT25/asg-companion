"""A11 protocol R: R-E2 speed edit on real VS13 held-out-vehicle clips.

60 4-s CPA crops from the EDITOR's held-out cars (est2_vs13.pt
val_cars).  Commanded speed v' = true v * factor, factor alternating
{0.7, 1.3} over the clip index (recorded per clip).

Systems:
  source : the unedited crop (do-nothing baseline / verifier validation)
  ours   : v_hat = editor net (p3/ckpt/est2_vs13.pt, real-domain
           fine-tuned Est2); dry = inverse_passby(crop, v_hat);
           edited = render_node(dry, "passby", v', alpha=0, env=0)
           (semi-analytic physics inversion + re-render, open air)
  x5     : ZETA (ICML'24) prompt edit, source "a car passing by",
           target "a car passing by at {v'*3.6:.0f} km/h" (P1-locked
           repo defaults via p3_x5_zeta.run_zeta)
  x6     : Audio-Omni (arXiv'26) instruction edit, "make the car
           pass by at {v'*3.6:.0f} km/h" (p3_x6_audioomni wiring)
  x1-x4 separation family: N/A-by-design on R-E2 (recorded in ledger).

Metrics per edited clip:
  verifier speed error |v_verifier(edited) - v'| -- INDEPENDENT
    verifier net p3/ckpt/est2_vs13_verifier.pt (disjoint split seed 1),
    validated on the unedited clips against measured speeds first;
  CLAP text-audio adherence to the target caption;
  locality LSD + MRSTFT to the source (LPAPS substituted by LSD+MRSTFT,
    documented: no trained 16-kHz general-audio LPAPS available);
  FAD-CLAP of each system's 60 edited clips vs p3/rset_ref (200 real
    crops).

Stages (each in its venv; partials under p3/rset_out/):
  audiocf  : --stage ours   (writes manifest + source + ours wavs)
  audiocf  : --stage x5
  audioomni: --stage x6
  audiocf  : --stage verify (verifier + LSD + MRSTFT, all systems)
  aumetrics: --stage clap   (CLAP adherence + FAD-CLAP, all systems)
  any      : --stage aggregate -> paper_numbers/p3_r_e2_v0.json
"""
import argparse
import datetime
import json
import os
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, "/workspace/CF_ASG/src")
from p3_r_common import (ROUT, PN, CK, RREF, SR, SEG, peak03,  # noqa
                         si_sdr, lsd, agg, load_vs13_items)

E2 = ROUT / "e2"
PREFIX = "e2"                 # file prefix under ROUT; "e2b" for the
CFG = {"editor_ckpt": "est2_vs13.pt",          # A12.3 v1 rerun
       "verifier_ckpt": "est2_vs13_verifier.pt",
       "cars": "editor_val"}
FACTORS = (0.7, 1.3)
EDIT_SYSTEMS = ("ours", "x5", "x6")
ALL_SYSTEMS = ("source",) + EDIT_SYSTEMS
SRC_PROMPT = "a car passing by"


def part(name):
    return ROUT / f"{PREFIX}_{name}.json"


def tgt_caption(v_prime):
    return f"a car passing by at {v_prime * 3.6:.0f} km/h"


def x6_instruction(v_prime):
    return f"make the car pass by at {v_prime * 3.6:.0f} km/h"


def build_manifest(n=60):
    import torch
    if CFG["cars"] == "split3_eval":     # A12.3: 3 eval cars unseen by
        split = json.loads((CK / "vs13_split3.json").read_text())
        val_cars = list(split["eval_cars"])   # editor AND verifier
    else:
        ck = torch.load(CK / "est2_vs13.pt", map_location="cpu",
                        weights_only=False)
        val_cars = list(ck["val_cars"])
    held = [it for it in load_vs13_items() if it["car"] in val_cars]
    perm = np.random.default_rng(4000).permutation(len(held))
    sel = [held[int(k)] for k in perm[:n]]
    man = []
    for j, it in enumerate(sel):
        f = FACTORS[j % 2]
        man.append({"clip": j, "wav": it["wav"], "car": it["car"],
                    "v_true": it["v"], "v_true_kmh": it["v_kmh"],
                    "cpa": it["cpa"], "factor": f,
                    "v_prime": it["v"] * f,
                    "v_prime_kmh": it["v_kmh"] * f})
    return {"editor_val_cars": val_cars, "clips": man}


def load_crop(item):
    import soundfile as sf
    import torch
    import torchaudio
    x, sr = sf.read(item["wav"], dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    if sr != SR:
        x = torchaudio.functional.resample(torch.tensor(x), sr, SR).numpy()
    x = x.astype(np.float64)
    c = int(item["cpa"] * SR)
    half = SEG // 2
    c = max(half, min(len(x) - half, c))
    return peak03(x[c - half:c + half])


def wav_path(sysname, j):
    return E2 / sysname / f"clip_{j:03d}.wav"


def read_wav(p):
    import soundfile as sf
    x, sr = sf.read(str(p), dtype="float64")
    assert sr == SR
    return x[:SEG]


def write_wav(p, x):
    import soundfile as sf
    p.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(p), np.asarray(x, np.float32), SR, subtype="FLOAT")


def ls_fit(w, ref):
    a = float(np.dot(w, ref) / (np.dot(w, w) + 1e-12))
    return a * w


# ---------------- stages ----------------
def stage_ours(n):
    import torch
    torch.set_num_threads(8)
    dev = "cuda"
    from p3_train import Est2
    from p3_edit_eval import inverse_passby, render_node
    man = build_manifest(n)
    part("manifest").write_text(json.dumps(man, indent=1))
    ed = Est2().to(dev)
    ed.load_state_dict(torch.load(CK / CFG["editor_ckpt"],
                                  map_location=dev,
                                  weights_only=False)["model"])
    ed.eval()
    rows = []
    for it in man["clips"]:
        j = it["clip"]
        crop = load_crop(it)
        write_wav(wav_path("source", j), crop)
        with torch.no_grad():
            _, pv = ed(torch.tensor(crop, dtype=torch.float32
                                    )[None].to(dev))
        v_hat = 5.0 + 35.0 * float(pv[0].clamp(0, 1))
        dry = inverse_passby(crop, v_hat)
        edited = render_node(dry, "passby", it["v_prime"], 0.0, env=0.0)
        write_wav(wav_path("ours", j), edited)
        rows.append({"clip": j, "v_hat_editor": v_hat,
                     "editor_err_true_ms": abs(v_hat - it["v_true"])})
        print(f"ours {j + 1}/{len(man['clips'])} v_hat={v_hat:.1f} "
              f"v_true={it['v_true']:.1f} v'={it['v_prime']:.1f}",
              flush=True)
    part("ours").write_text(json.dumps(
        {"meta": {"date": str(datetime.date.today()),
                  "editor_ckpt": "p3/ckpt/" + CFG["editor_ckpt"],
                  "cars_mode": CFG["cars"]},
         "rows": rows}, indent=1))
    print("editor |v_hat - v_true| median (km/h):",
          3.6 * float(np.median([r["editor_err_true_ms"]
                                 for r in rows])))


def stage_x5(n):
    import torch
    torch.set_num_threads(8)
    from p3_x5_zeta import run_zeta, ZETA_CODE
    import soundfile as sf
    man = json.loads(part("manifest").read_text())
    rows, fails = [], []
    tmp_rel = "p3rE2_tmp"
    pcm_dir = E2 / "source_pcm16"     # ZETA's loader is wave-module
    pcm_dir.mkdir(parents=True, exist_ok=True)  # based: no FLOAT wavs
    try:
        for it in man["clips"][:n]:
            j = it["clip"]
            tar_p = tgt_caption(it["v_prime"])
            if wav_path("x5", j).exists():   # resume: keep prior output
                rows.append({"clip": j, "source_prompt": SRC_PROMPT,
                             "target_prompt": tar_p, "resumed": True})
                continue
            crop0 = read_wav(wav_path("source", j))
            src = str(pcm_dir / f"clip_{j:03d}.wav")
            sf.write(src, np.asarray(crop0, np.float32), SR,
                     subtype="PCM_16")
            res_rel = f"{tmp_rel}/c{j}"
            try:
                ep, _ = run_zeta(src, SRC_PROMPT, tar_p, 4200 + j,
                                 res_rel)
            except Exception as ex:
                ep = None
                fails.append({"clip": j,
                              "error": f"{type(ex).__name__}: {ex}"})
            if ep is None:
                if not fails or fails[-1]["clip"] != j:
                    fails.append({"clip": j,
                                  "error": "run_zeta returned None "
                                           "(see log for stderr)"})
                print(f"x5 FAIL clip {j}", flush=True)
                continue
            import torchaudio
            w, sr = torchaudio.load(str(ep))
            w16 = torchaudio.functional.resample(
                w.mean(0), sr, SR).numpy().astype(np.float64)[:SEG]
            if len(w16) < SEG:
                w16 = np.pad(w16, (0, SEG - len(w16)))
            crop = read_wav(wav_path("source", j))
            write_wav(wav_path("x5", j), ls_fit(w16, crop))
            rows.append({"clip": j, "source_prompt": SRC_PROMPT,
                         "target_prompt": tar_p})
            shutil.rmtree(ZETA_CODE / res_rel, ignore_errors=True)
            print(f"x5 {j + 1}/{len(man['clips'][:n])}", flush=True)
    finally:
        shutil.rmtree(ZETA_CODE / tmp_rel, ignore_errors=True)
    part("x5").write_text(json.dumps(
        {"meta": {"date": str(datetime.date.today()),
                  "system": "X5 ZETA (ICML'24)",
                  "prompts": {"source": SRC_PROMPT,
                              "target": "a car passing by at "
                                        "{v'*3.6:.0f} km/h"},
                  "ls_fit": "edited clip LS scale-fitted to the source "
                            "crop (output gain arbitrary, disclosed)",
                  "input_note": "ZETA consumes a 16-bit PCM copy of "
                                "the source crop (repo loader cannot "
                                "read FLOAT wavs)",
                  "n_ok": len(rows), "n_fail": len(fails)},
         "rows": rows, "fails": fails}, indent=1))


def stage_x6(n):
    import torch
    torch.set_num_threads(8)
    from p3_x6_audioomni import (AudioOmni, CFG, CKPT,  # noqa: E402
                                 edit_instruction)
    man = json.loads(part("manifest").read_text())
    model = AudioOmni(CFG, CKPT, device="cuda")
    rows, fails = [], []
    for it in man["clips"][:n]:
        j = it["clip"]
        instr = x6_instruction(it["v_prime"])
        try:
            w16 = edit_instruction(model, str(wav_path("source", j)),
                                   instr, 100, 7.0, seed=300000 + j)
            crop = read_wav(wav_path("source", j))
            write_wav(wav_path("x6", j), ls_fit(w16, crop))
            rows.append({"clip": j, "instruction": instr})
        except Exception as ex:
            fails.append({"clip": j,
                          "error": f"{type(ex).__name__}: {ex}",
                          "trace": traceback.format_exc()[-600:]})
            print(f"x6 FAIL clip {j}: {ex}", flush=True)
        print(f"x6 {j + 1}/{len(man['clips'][:n])}", flush=True)
    part("x6").write_text(json.dumps(
        {"meta": {"date": str(datetime.date.today()),
                  "system": "X6 Audio-Omni (arXiv'26)",
                  "instruction": "make the car pass by at "
                                 "{v'*3.6:.0f} km/h",
                  "steps": 100, "cfg_scale": 7.0,
                  "ls_fit": "edited clip LS scale-fitted to the source "
                            "crop (repo output peak-normalized, "
                            "disclosed)",
                  "n_ok": len(rows), "n_fail": len(fails)},
         "rows": rows, "fails": fails}, indent=1))


def stage_verify(n):
    import torch
    torch.set_num_threads(8)
    dev = "cuda"
    sys.path.insert(0, "/workspace/CF_PhysAudio/src")
    from p3_train import Est2
    from p2_train import mrstft_loss

    def mrstft_np(a, b):
        ta = torch.tensor(a, dtype=torch.float32)[None]
        tb = torch.tensor(b, dtype=torch.float32)[None]
        L = min(ta.shape[-1], tb.shape[-1])
        return float(mrstft_loss(ta[:, :L], tb[:, :L]))

    ver = Est2().to(dev)
    vck = torch.load(CK / CFG["verifier_ckpt"], map_location=dev,
                     weights_only=False)
    ver.load_state_dict(vck["model"])
    ver.eval()
    eck = torch.load(CK / CFG["editor_ckpt"], map_location="cpu",
                     weights_only=False)
    if "train_cars" in vck:            # A12.3 split3 ckpts
        ver_cars = list(vck.get("eval_cars", []))
        ed_cars = list(eck.get("eval_cars", []))
        ver_train = list(vck["train_cars"])
        ed_train = list(eck.get("train_cars", []))
        overlap = sorted(set(ver_train) & set(ed_train))
    else:
        ver_cars = list(vck.get("val_cars", []))
        ed_cars = list(eck["val_cars"])
        ver_train = None
        ed_train = None
        overlap = sorted(set(ver_cars) & set(ed_cars))

    def vhat(x):
        with torch.no_grad():
            _, pv = ver(torch.tensor(peak03(x), dtype=torch.float32
                                     )[None].to(dev))
        return 5.0 + 35.0 * float(pv[0].clamp(0, 1))

    man = json.loads(part("manifest").read_text())
    out = {s: [] for s in ALL_SYSTEMS}
    for it in man["clips"][:n]:
        j = it["clip"]
        crop = read_wav(wav_path("source", j))
        for s in ALL_SYSTEMS:
            p = wav_path(s, j)
            if not p.exists():
                continue
            x = read_wav(p)
            v = vhat(x)
            row = {"clip": j, "v_verifier_ms": v,
                   "verr_target_ms": abs(v - it["v_prime"]),
                   "verr_target_kmh": 3.6 * abs(v - it["v_prime"])}
            if s == "source":
                row["verr_true_ms"] = abs(v - it["v_true"])
                row["verr_true_kmh"] = 3.6 * abs(v - it["v_true"])
                row["lsd_src"], row["mrstft_src"] = 0.0, 0.0
            else:
                row["lsd_src"] = lsd(x, crop)
                row["mrstft_src"] = mrstft_np(x, crop)
            out[s].append(row)
    part("verify").write_text(json.dumps(
        {"meta": {"date": str(datetime.date.today()),
                  "verifier_ckpt": "p3/ckpt/" + CFG["verifier_ckpt"],
                  "verifier_val_cars": ver_cars,
                  "editor_val_cars": ed_cars,
                  "verifier_train_cars": ver_train,
                  "editor_train_cars": ed_train,
                  "train_cars_overlap" if ver_train is not None
                  else "val_cars_overlap": overlap,
                  "verifier_input": "peak-norm 0.3 (VS13 eval "
                                    "convention)",
                  "locality": "LSD + MRSTFT vs source (LPAPS "
                              "substitute, documented)"},
         "rows": out}, indent=1))
    for s in ALL_SYSTEMS:
        if out[s]:
            print(s, "verr_target_kmh median",
                  round(float(np.median([r["verr_target_kmh"]
                                         for r in out[s]])), 2))
    if out["source"]:
        print("verifier validation |v-v_true| kmh median",
              round(float(np.median([r["verr_true_kmh"]
                                     for r in out["source"]])), 2))


def stage_clap(n):
    os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
    os.environ.setdefault("TORCH_HOME", "/workspace/hf_cache/torch")
    import glob
    import librosa
    import torch
    from scipy import linalg
    from transformers import ClapModel, ClapProcessor
    torch.set_num_threads(8)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    clap = ClapModel.from_pretrained("laion/clap-htsat-unfused"
                                     ).to(dev).eval()
    proc = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    CSR = 48000

    def aemb(path):
        y, _ = librosa.load(path, sr=CSR, mono=True)
        with torch.no_grad():
            inp = proc(audio=[y], sampling_rate=CSR, return_tensors="pt")
            e = clap.get_audio_features(**{k: v.to(dev)
                                           for k, v in inp.items()})
            if not torch.is_tensor(e):
                e = e.pooler_output
        return e.squeeze(0).cpu().numpy()

    def temb(text):
        with torch.no_grad():
            inp = proc(text=[text], return_tensors="pt", padding=True)
            e = clap.get_text_features(**{k: v.to(dev)
                                          for k, v in inp.items()})
            if not torch.is_tensor(e):
                e = e.pooler_output
        return e.squeeze(0).cpu().numpy()

    def fad(ea, eb):
        mu1, s1 = ea.mean(0), np.cov(ea, rowvar=False)
        mu2, s2 = eb.mean(0), np.cov(eb, rowvar=False)
        cm, _ = linalg.sqrtm(s1.dot(s2), disp=False)
        if np.iscomplexobj(cm):
            cm = cm.real
        d = mu1 - mu2
        return float(d.dot(d) + np.trace(s1) + np.trace(s2)
                     - 2 * np.trace(cm))

    man = json.loads(part("manifest").read_text())
    refs = sorted(glob.glob(str(RREF / "ref_*.wav")))
    print(f"embedding {len(refs)} reference wavs", flush=True)
    ref_emb = np.stack([aemb(p) for p in refs])
    rows = {s: [] for s in ALL_SYSTEMS}
    embs = {s: [] for s in ALL_SYSTEMS}
    for it in man["clips"][:n]:
        j = it["clip"]
        te = temb(tgt_caption(it["v_prime"]))
        te = te / (np.linalg.norm(te) + 1e-9)
        for s in ALL_SYSTEMS:
            p = wav_path(s, j)
            if not p.exists():
                continue
            ae = aemb(str(p))
            embs[s].append(ae)
            an = ae / (np.linalg.norm(ae) + 1e-9)
            rows[s].append({"clip": j,
                            "clap_target": float(np.dot(an, te))})
        print(f"clap clip {j + 1}/{len(man['clips'][:n])}", flush=True)
    fads = {s: (fad(np.stack(embs[s]), ref_emb)
                if len(embs[s]) >= 2 else None) for s in ALL_SYSTEMS}
    part("clap").write_text(json.dumps(
        {"meta": {"date": str(datetime.date.today()),
                  "clap": "laion/clap-htsat-unfused",
                  "caption": "a car passing by at {v'*3.6:.0f} km/h",
                  "fad_ref": f"p3/rset_ref ({len(refs)} real crops)"},
         "rows": rows, "fad_clap": fads}, indent=1))
    print(json.dumps(fads, indent=1))


def stage_aggregate(out_name):
    man = json.loads(part("manifest").read_text())
    ours = json.loads(part("ours").read_text())
    ver = json.loads(part("verify").read_text())
    clap = json.loads(part("clap").read_text())
    parts = {}
    for s in ("x5", "x6"):
        p = part(s)
        parts[s] = json.loads(p.read_text()) if p.exists() else None

    led = {"meta": {
        "date": str(datetime.date.today()),
        "script": "src/p3_r_e2.py",
        "protocol": ("PROTOCOL_P3.md amendment A11, R-E2 (weak GT, "
                     "headline)" +
                     ("; A12.3 redo: three-way vehicle split, editor "
                      "and verifier fine-tuned on disjoint 5-car sets, "
                      "verifier synthetic-pretrained FROM SCRATCH "
                      "(fresh seed, no weight sharing with the editor "
                      "at any stage), eval clips from 3 cars unseen by "
                      "both" if CFG["cars"] == "split3_eval" else "")),
        "config": dict(CFG),
        "n_clips": len(man["clips"]),
        "eval_cars" if CFG["cars"] == "split3_eval"
        else "editor_val_cars": man["editor_val_cars"],
        "verifier": ver["meta"],
        "factors": "v' = v_true * {0.7, 1.3}, alternating by clip "
                   "index (recorded per clip)",
        "separation_family": "X1-X4 N/A-by-design on R-E2 (no "
                             "parametric re-render path; documented)",
        "lpaps_note": "LPAPS substituted by LSD + MRSTFT (no trained "
                      "16-kHz general-audio LPAPS available; both "
                      "cover the locality axis; documented choice)",
        "external_fails": {s: (parts[s]["fails"] if parts[s] else
                               "stage missing") for s in ("x5", "x6")}},
        "clips": man["clips"], "editor_rows": ours["rows"],
        "rows": {}, "aggregates": {}}

    # verifier vehicle exposure.  Legacy (v0): the verifier trained on
    # all cars except ITS OWN val cars, so an eval car outside that set
    # was seen during verifier fine-tuning.  A12.3 (v1): the verifier
    # trained ONLY on its 5 verifier cars, so seen == car in that list
    # (empty on the 3 eval cars by construction).
    car_of = {c["clip"]: c["car"] for c in man["clips"]}
    ver_train = ver["meta"].get("verifier_train_cars")
    if ver_train is not None:
        seen = {j: (car_of[j] in set(ver_train)) for j in car_of}
        note = ("A12.3 split3: veh_seen_by_verifier = clip's car in "
                "the verifier's 5 fine-tune cars (eval cars are "
                "disjoint from editor AND verifier fine-tune sets by "
                "construction)")
    else:
        ver_val = set(ver["meta"].get("verifier_val_cars", []))
        seen = {j: (car_of[j] not in ver_val) for j in car_of}
        note = ("veh_seen_by_verifier = clip's car was in the "
                "verifier's fine-tune training set (cars not in "
                "verifier val_cars); flagged per row and split in "
                "the validation aggregate")
    led["meta"]["verifier_exposure"] = {
        "note": note,
        "seen_cars": sorted({car_of[j] for j in car_of if seen[j]}),
        "unseen_cars": sorted({car_of[j] for j in car_of
                               if not seen[j]})}

    vrows = {s: {r["clip"]: r for r in ver["rows"].get(s, [])}
             for s in ALL_SYSTEMS}
    crows = {s: {r["clip"]: r for r in clap["rows"].get(s, [])}
             for s in ALL_SYSTEMS}
    for s in ALL_SYSTEMS:
        merged = []
        for j in sorted(vrows[s]):
            row = dict(vrows[s][j])
            row["veh_seen_by_verifier"] = seen.get(j)
            if j in crows[s]:
                row["clap_target"] = crows[s][j]["clap_target"]
            merged.append(row)
        led["rows"][s] = merged
        led["aggregates"][s] = {
            "n": len(merged),
            "verr_target_kmh": agg(merged, "verr_target_kmh"),
            "verr_target_ms": agg(merged, "verr_target_ms"),
            "clap_target": agg(merged, "clap_target"),
            "lsd_src": agg(merged, "lsd_src"),
            "mrstft_src": agg(merged, "mrstft_src"),
            "fad_clap_vs_ref": clap["fad_clap"].get(s)}
    src_rows = led["rows"]["source"]
    led["aggregates"]["verifier_validation"] = {
        "desc": "verifier on UNEDITED clips vs measured speed "
                "(validation gate for the weak-GT metric)",
        "verr_true_kmh": agg(src_rows, "verr_true_kmh"),
        "verr_true_kmh_veh_seen": agg(
            [r for r in src_rows if r["veh_seen_by_verifier"]],
            "verr_true_kmh"),
        "verr_true_kmh_veh_unseen": agg(
            [r for r in src_rows if not r["veh_seen_by_verifier"]],
            "verr_true_kmh")}
    led["aggregates"]["editor_on_source"] = {
        "desc": "editor |v_hat - v_true| on the same clips (km/h)",
        "err_kmh": agg([{"e": 3.6 * r["editor_err_true_ms"]}
                        for r in ours["rows"]], "e")}

    # paired statistics (pattern of src/p3_r_e2_stats.py, M3)
    from scipy.stats import wilcoxon
    clip_of = {c["clip"]: c for c in man["clips"]}

    def series(sysk):
        return {r["clip"]: r["verr_target_kmh"]
                for r in led["rows"].get(sysk, [])}

    stats = {}
    src_s = series("source")
    for sysk in EDIT_SYSTEMS:
        sv = series(sysk)
        common = sorted(set(src_s) & set(sv))
        if len(common) < 8:
            continue
        diffs = [sv[c] - src_s[c] for c in common]
        stats[f"{sysk}_vs_noedit"] = {
            "n": len(diffs),
            "median_diff_kmh": float(np.median(diffs)),
            "improves_n": int(sum(x < 0 for x in diffs)),
            "wilcoxon_p": float(wilcoxon(diffs).pvalue)}
        for direc, keep in (("up", lambda c: clip_of[c]["factor"] > 1),
                            ("down", lambda c: clip_of[c]["factor"] < 1)):
            dd = [sv[c] - src_s[c] for c in common if keep(c)]
            if len(dd) >= 8:
                stats[f"{sysk}_vs_noedit_{direc}"] = {
                    "n": len(dd),
                    "median_diff_kmh": float(np.median(dd)),
                    "wilcoxon_p": float(wilcoxon(dd).pvalue)}
    ov = series("ours")
    for xk in ("x5", "x6"):
        xv = series(xk)
        common = sorted(set(ov) & set(xv))
        if len(common) >= 8:
            diffs = [ov[c] - xv[c] for c in common]
            stats[f"ours_vs_{xk}"] = {
                "n": len(common),
                "median_diff_kmh": float(np.median(diffs)),
                "wilcoxon_p": float(wilcoxon(diffs).pvalue)}
    led["stats"] = stats
    (PN / out_name).write_text(json.dumps(led, indent=1))
    print(json.dumps(stats, indent=1))
    for s in ALL_SYSTEMS:
        a = led["aggregates"][s]
        print(s, "n=", a["n"],
              "verr_kmh=", a["verr_target_kmh"]["median"]
              if a["verr_target_kmh"] else None,
              "clap=", round(a["clap_target"]["median"], 3)
              if a["clap_target"] else None,
              "lsd=", round(a["lsd_src"]["median"], 2)
              if a["lsd_src"] else None,
              "fad=", round(a["fad_clap_vs_ref"], 2)
              if a["fad_clap_vs_ref"] is not None else None)


def main():
    global E2, PREFIX, CFG
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=("ours", "x5", "x6", "verify", "clap",
                             "aggregate"))
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", default="p3_r_e2_v0.json")
    ap.add_argument("--v1", action="store_true",
                    help="A12.3 redo: three-way vehicle split "
                         "(vs13_split3.json), editor est2_vs13_ed3.pt, "
                         "scratch-based verifier est2_vs13_ver3.pt, "
                         "clips from the 3 eval cars; wavs under "
                         "p3/rset_out/e2b, parts e2b_*.json")
    args = ap.parse_args()
    if args.v1:
        E2 = ROUT / "e2b"
        PREFIX = "e2b"
        CFG = {"editor_ckpt": "est2_vs13_ed3.pt",
               "verifier_ckpt": "est2_vs13_ver3.pt",
               "cars": "split3_eval"}
    E2.mkdir(parents=True, exist_ok=True)
    if args.stage == "ours":
        stage_ours(args.n)
    elif args.stage == "x5":
        stage_x5(args.n)
    elif args.stage == "x6":
        stage_x6(args.n)
    elif args.stage == "verify":
        stage_verify(args.n)
    elif args.stage == "clap":
        stage_clap(args.n)
    else:
        stage_aggregate(args.out)


if __name__ == "__main__":
    main()
