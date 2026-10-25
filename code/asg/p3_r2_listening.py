"""A14.4 (v2): listening-test package, ten trials, agreement question.

Replaces the v1 MUSHRA-style package (p3_r2_companion.py) after the
author's pilot on 2026-10-18 found most v1 trials unusable: removed or
kept sources 30 to 60 dB below the mixture, pass-by sources active for
one second, absorption steps of 0.05, moved sources in single-source
scenes (a pure gain change), and class labels that did not match what
one hears.

Design
  Each trial shows the ORIGINAL recording openly, states in words what
  the edit must produce, and asks for each clip how much the listener
  agrees that the clip is the original with exactly that change.
  Hidden conditions: the ground-truth edit (closed-form render, "gt")
  and the unedited original ("orig"); systems: ours, AudioSep (x1),
  SoloAudio (x3), Audio-Omni (x6, where it was run). E4 has no external
  system (the room path is stem independent).
  Trials: E1 x3, E2 x2, E3 x2, E4 x3 = 10.

Scene choice (deterministic, data driven)
  Every source in a chosen scene must be within LEVEL_DB of the loudest
  one, be active for at least ACTIVE_MIN of the clip (E4: transient
  sources, active at most ACTIVE_MAX), and pass a CLAP label check (its
  class prompt is the top-1 or top-2 match among all 23 class prompts).
  E2: vehicle-like class, speed ratio >= 2.5, new speed <= 40 m/s.
  E3: two-source room scene, factor 4 (farther) or 0.5 (closer).
  E4: |delta alpha| >= 0.15.
  Among the survivors, ranking prefers balanced levels (E1, E3),
  larger speed ratio (E2), larger |delta alpha| (E4), and scenes where
  Audio-Omni was run.
-> p3/listening_test_v2/{index.html, config.js, config.json, stimuli/,
   README.txt, analyze.py}, paper_numbers/p3_r2_listening_v2.json
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, "/workspace/CF_ASG/src")
import p3_r2_common as C  # noqa: E402
from p3_r2_common import SceneSet, load_scene, q_of  # noqa: E402
from p3_r2_metrics import Clap, prompt  # noqa: E402
from p3_scenes import BROAD  # noqa: E402

ROOT = Path("/workspace/CF_ASG")
LT = ROOT / "p3" / "listening_test_v2"
SR = 16000
LEVEL_DB = 15.0      # a source within this of the loudest is "dominant"
QUIET_DB = 20.0      # a source this far below the loudest is ignored (E2, E4)
ACTIVE_MIN = 0.4
ACTIVE_MAX = 0.6
CLAP_RANK_MAX = 3
CREST_IMPULSIVE = 8.0   # two dominant impulsive sources fuse perceptually (pilot: bell + hammer)
LOW_FREQ = {"Bass_drum", "Thump_and_thud"}   # a scene whose audible sources are all low-frequency cannot expose the tail (pilot: bass drum)
GAP_MIN = 80       # E4: at least 0.8 s of 10 ms frames inside gaps where the edit moves the level by > 3 dB
LEAD_MAX = 0.5     # E4: first audible onset within 0.5 s (pilot: 1.1 s of silence then a late tail was not heard)
# scenes rejected in the author's pilot, with the reason
EXCLUDE = {1433: "glass and alarm cannot be told apart by ear"}
N_TRIALS = {"E3": 2, "E2": 2, "E4": 3, "E1": 3}   # scarce edits first
SYS = ["ours", "x1", "x3", "x6"]
VEHICLE = {"Siren", "Vehicle_horn_and_car_horn_and_honking", "Alarm", "Engine",
           "Engine_starting"}
TRANSIENT = {"Knock", "Clapping", "Slam", "Drum", "Snare_drum", "Bass_drum",
             "Glass", "Gunshot_and_gunfire", "Thump_and_thud", "Hammer", "Tap",
             "Door", "Sliding_door"}
EN = {"tap": "tapping sound", "thump": "thump", "glass": "breaking glass"}
ZH = {"siren": "警笛声", "car horn": "汽车喇叭声", "church bell": "教堂钟声",
      "bell": "铃声", "doorbell": "门铃声", "cowbell": "牛铃声",
      "bicycle bell": "自行车铃声", "alarm": "警报声", "engine": "引擎声",
      "knock": "敲击声", "clapping": "掌声", "slam": "摔门声", "drum": "鼓声",
      "snare drum": "小军鼓声", "bass drum": "大鼓声", "glass": "玻璃声",
      "gunshot": "枪声", "thump": "重击声", "hammer": "锤击声", "tap": "轻敲声",
      "door": "门声", "engine starting": "发动机启动声", "sliding door": "滑门声"}


def dbfs(x):
    return float(10 * np.log10(np.mean(x ** 2) + 1e-12))


def crest(x):
    return float(np.max(np.abs(x)) / (np.sqrt(np.mean(x ** 2)) + 1e-9))


def fused(src):
    """True when two or more dominant sources are all impulsive."""
    dom = [r for r in src if r["rel"] >= -LEVEL_DB]
    return len(dom) >= 2 and all(r.get("crest", 0) > CREST_IMPULSIVE for r in dom)


def envelope_db(x, hop=0.01):
    n = int(hop * SR)
    m = len(x) // n
    return 10 * np.log10((x[:m * n].reshape(m, n) ** 2).mean(1) + 1e-12)


def gap_score(base_wav, edited_wav):
    """Measured tail exposure of a room edit: (frames, lead). frames = 10 ms
    frames before 3.3 s where the unedited level sits in a gap (20 dB below
    the peak), the louder of the two clips is still audible (within 40 dB of
    the peak) and the edit moves the level by more than 3 dB. lead = time of
    the first audible onset (s). Replaces the earlier class-based rule
    (pilot: bass drum, and a gunshot after 1.1 s of silence, were not heard)."""
    eo = envelope_db(sf.read(str(base_wav))[0])
    eg = envelope_db(sf.read(str(edited_wav))[0])
    pk = max(eo.max(), eg.max())
    t = np.arange(len(eo)) / 100.0
    gap = (eo < pk - 20) & (np.maximum(eo, eg) > pk - 40) & (t < 3.3)
    frames = int((gap & (np.abs(eg - eo) > 3)).sum())
    lead = float(np.argmax(eo > pk - 40)) / 100.0
    return frames, lead


def exposes_tail(i, e="E4", base="mix"):
    """True when the measured tail exposure of scene i for room edit e
    passes GAP_MIN and LEAD_MAX (base: the clip the edit is compared to)."""
    b = C.R2_REF / f"{i:05d}_{base}.wav"
    r = C.R2_REF / f"{i:05d}_{e}_ref.wav"
    if not (b.exists() and r.exists()):
        return False, 0
    frames, lead = gap_score(b, r)
    return frames >= GAP_MIN and lead <= LEAD_MAX, frames


def article(c):
    return ("an " if c[0] in "aeiou" else "a ") + c


def active_fraction(x, thr_db=-35.0):
    f = SR // 20
    n = len(x) // f
    r = 10 * np.log10(np.array([np.mean(x[i * f:(i + 1) * f] ** 2) for i in range(n)]) + 1e-12)
    return float(np.mean(r > r.max() + thr_db))


def scan(ds, specs, clap):
    """Per-scene source statistics and CLAP label rank."""
    T = np.stack([clap.text(prompt(c)) for c in BROAD])
    out = {}
    for i in sorted(specs):
        sc = load_scene(ds, i)
        lv = [dbfs(sc["gts"][k]) for k in range(sc["K"])]
        rows = []
        for k in range(sc["K"]):
            e = clap.audio(sc["gts"][k])
            sims = T @ e
            rank = int(np.argsort(-sims).tolist().index(BROAD.index(sc["classes"][k])))
            rows.append({"cls": sc["classes"][k], "level": lv[k],
                         "rel": lv[k] - max(lv), "active": active_fraction(sc["gts"][k]),
                         "crest": crest(sc["gts"][k]),
                         "clap_rank": rank, "traj": C.traj_name(sc, k),
                         "v": C.theta_of(sc, k)})
        out[i] = {"K": sc["K"], "alpha": sc["alpha"], "room": bool(sc["env"] > 0.5),
                  "src": rows}
    return out


def ok_source(r, active_min=ACTIVE_MIN):
    # a pass-by source must be a vehicle-like class (a bell moving at
    # 20 m/s is physically valid but confuses listeners)
    return (r["rel"] >= -LEVEL_DB and r["clap_rank"] <= CLAP_RANK_MAX
            and r["active"] >= active_min
            and (r["traj"] != "passby" or r["cls"] in VEHICLE))


def dominant(src):
    """Sources that matter perceptually; the rest must be very quiet."""
    dom = [r for r in src if r["rel"] >= -LEVEL_DB]
    quiet = [r for r in src if r["rel"] < -LEVEL_DB]
    if any(r["rel"] >= -QUIET_DB for r in quiet):
        return None
    return dom


def has_x6(i, e):
    return (C.R2_OUT / "x6" / f"{i:05d}_{e}.wav").exists()


def candidates(stats, specs, e):
    cands = []
    for i, st in stats.items():
        sp = specs[i]
        src = st["src"]
        if i in EXCLUDE:
            continue
        if fused(src) and e != "E4":     # for E4 the identity of the sources does not matter
            continue
        if e == "E1":
            if st["K"] != 2 or "E1_slot" not in sp:
                continue
            if not all(ok_source(r) for r in src):
                continue
            score = -abs(src[0]["rel"] - src[1]["rel"]) + 3 * has_x6(i, e)
        elif e == "E2":
            if "E2_slot" not in sp:
                continue
            r = src[sp["E2_slot"]]
            v0, v1 = r["v"], sp["E2_newv"]
            ratio = max(v1 / v0, v0 / v1)
            dom = dominant(src)
            if (dom is None or r["cls"] not in VEHICLE or r["traj"] != "passby" or ratio < 2.5
                    or v1 > 40 or not ok_source(r, 0.9) or not all(ok_source(q) for q in dom)):
                continue
            score = ratio + 3 * has_x6(i, e)
        elif e == "E3":
            if st["K"] != 2 or not st["room"] or "E3_factor" not in sp:
                continue
            if not all(ok_source(r) for r in src):
                continue
            big = sp["E3_factor"] in (4.0, 0.5)
            score = 20 * big - abs(src[0]["rel"] - src[1]["rel"]) + 3 * has_x6(i, e) - 10 * st["alpha"]
        else:
            if "E4_newa" not in sp or abs(sp["E4_newa"] - st["alpha"]) < 0.099:
                continue
            dom = dominant(src)
            if dom is None or not all(r["clap_rank"] <= CLAP_RANK_MAX for r in dom):
                continue
            if all(r["cls"] in LOW_FREQ for r in dom):
                continue
            # the tail must be exposed and audibly changed (measured on the renders)
            ok, frames = exposes_tail(i)
            if not ok:
                continue
            score = frames
        if not (C.R2_REF / f"{i:05d}_{e}_ref.wav").exists():
            continue
        if not (C.R2_OUT / "ours" / f"{i:05d}_{e}.wav").exists():
            continue
        cands.append((score, i))
    cands.sort(reverse=True)
    return cands


def pick(cands, n, used, want_factor=None, specs=None, stats=None, rm_key=None):
    """Greedy over ranked candidates: distinct scenes, distinct removed
    class (E1), optional E3 factor alternation."""
    chosen, seen = [], set()
    for _, i in cands:
        if i in used:
            continue
        if rm_key is not None:
            c = stats[i]["src"][specs[i][rm_key]]["cls"]
            if c in seen:
                continue
            seen.add(c)
        if want_factor is not None:
            f = specs[i]["E3_factor"]
            if f != want_factor[len(chosen) % len(want_factor)]:
                continue
        chosen.append(i)
        used.add(i)
        if len(chosen) == n:
            break
    return chosen


def scene_words(st):
    """Class words (EN, ZH) and the sentence that describes the original."""
    cl = [q_of(r["cls"]) for r in st["src"]]
    zh = [ZH.get(c, c) for c in cl]
    cl = [EN.get(c, c) for c in cl]     # plain English for the listener ("tap" could be read as a water tap)
    # name only the sources a listener can hear
    heard = [k for k, r in enumerate(st["src"]) if r["rel"] >= -LEVEL_DB]
    room_en = (f"in a room (absorption {st['alpha']:.2f})" if st["room"] else "in the open air")
    room_zh = (f"在一个房间里（吸声系数 {st['alpha']:.2f}）" if st["room"] else "在室外")
    scene_en = " and ".join(article(cl[k]) + (" passing by" if st["src"][k]["traj"] == "passby" else "")
                            for k in heard)
    scene_zh = "和".join(zh[k] + ("（经过）" if st["src"][k]["traj"] == "passby" else "")
                        for k in heard)
    orig_en = f"The original recording contains {scene_en}, {room_en}."
    orig_zh = f"原录音里有{scene_zh}，{room_zh}。"
    return cl, zh, orig_en, orig_zh


def words(st, sp, e):
    cl, zh, orig_en, orig_zh = scene_words(st)
    if e == "E1":
        a = sp["E1_slot"]
        others = [k for k in range(len(cl)) if k != a and st["src"][k]["rel"] >= -LEVEL_DB]
        stay_en = " and ".join(f"the {cl[k]}" for k in others)
        edit_en = f"The {cl[a]} is removed."
        edit_zh = f"去掉{zh[a]}。"
        if others:
            edit_en += f" {stay_en[0].upper()}{stay_en[1:]} must stay exactly as in the original."
            edit_zh += "和".join(zh[k] for k in others) + "必须和原录音完全一样。"
    elif e == "E2":
        a = sp["E2_slot"]
        v0, v1 = st["src"][a]["v"], sp["E2_newv"]
        fast = v1 > v0
        edit_en = (f"The {cl[a]} passes by at {v1:.0f} m/s instead of {v0:.0f} m/s, that is "
                   f"{'faster' if fast else 'slower'}. The pitch drop as it passes should be "
                   f"{'larger and quicker' if fast else 'smaller and more gradual'}, and the "
                   f"rise and fall of loudness should be {'shorter' if fast else 'longer'}.")
        edit_zh = (f"{zh[a]}以 {v1:.0f} m/s 而不是 {v0:.0f} m/s 经过，也就是变{'快' if fast else '慢'}了。"
                   f"经过时的音高下落应当{'更大更急' if fast else '更小更缓'}，"
                   f"响度的起落应当{'更短' if fast else '更长'}。")
    elif e == "E3":
        a = sp["E3_slot"]
        others = [k for k in range(len(cl)) if k != a and st["src"][k]["rel"] >= -LEVEL_DB]
        far = sp["E3_factor"] > 1
        ftxt = {4.0: ("4 times farther away", "4 倍远"), 2.0: ("2 times farther away", "2 倍远"),
                0.5: ("2 times closer", "2 倍近")}[sp["E3_factor"]]
        # the renderer scales the whole node by the distance ratio (direct
        # and room part alike), so the cue is level only. The distributed
        # listening package of 2026-10-18 also said "with relatively more
        # room sound"; its exact text is kept in p3_r2_listening_v2.json.
        edit_en = (f"The {cl[a]} is moved {ftxt[0]}: it should "
                   f"sound {'quieter and more distant' if far else 'louder and closer'}.")
        edit_zh = (f"{zh[a]}被移到{ftxt[1]}的位置：它应当听起来"
                   f"{'更轻更远' if far else '更响更近'}。")
        if others:
            stay_en = " and ".join(f"the {cl[k]}" for k in others)
            edit_en += f" {stay_en[0].upper()}{stay_en[1:]} must stay exactly as in the original."
            edit_zh += "和".join(zh[k] for k in others) + "必须和原录音完全一样。"
    else:
        a0, a1 = st["alpha"], sp["E4_newa"]
        more = a1 > a0
        edit_en = (f"The room becomes {'more' if more else 'less'} absorbing (absorption {a0:.2f} to {a1:.2f}): "
                   f"the reverberation tail after each sound becomes {'shorter' if more else 'longer'}. "
                   f"The sounds themselves stay the same.")
        edit_zh = (f"房间变得{'更吸声' if more else '更不吸声'}（吸声系数 {a0:.2f} 变为 {a1:.2f}）："
                   f"每个声音之后的混响尾巴应当变{'短' if more else '长'}。声音本身不变。")
    return {"original_en": orig_en, "original_zh": orig_zh,
            "edit_en": edit_en, "edit_zh": edit_zh}


def main():
    ds = SceneSet("val")
    led = json.loads((C.PN / "p3_r2_ours.json").read_text())
    specs = {s["scene"]: s for s in led["specs"]}
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    cache = C.PN / "p3_r2_listening_stats.json"
    if cache.exists():
        stats = {int(k): v for k, v in json.loads(cache.read_text()).items()}
    else:
        stats = scan(ds, specs, Clap(dev))
        cache.write_text(json.dumps(stats))
    used = set()
    chosen = {}
    for e, n in N_TRIALS.items():
        cands = candidates(stats, specs, e)
        chosen[e] = pick(cands, n, used, want_factor=(4.0, 0.5) if e == "E3" else None,
                         specs=specs, stats=stats, rm_key="E1_slot" if e == "E1" else None)
        if e == "E3" and len(chosen[e]) < n:   # fall back to any factor
            chosen[e] += pick(cands, n - len(chosen[e]), used)
        print(e, "candidates", len(cands), "chosen", chosen[e])
        if len(chosen[e]) < n:
            print("  WARNING: fewer trials than planned for", e)
    # ---------------- package
    if LT.exists():
        shutil.rmtree(LT)
    (LT / "stimuli").mkdir(parents=True)
    trials, ledger = [], []
    for e in ("E1", "E2", "E3", "E4"):
        for i in chosen[e]:
            st, sp = stats[i], specs[i]
            conds = {"gt": C.R2_REF / f"{i:05d}_{e}_ref.wav",
                     "orig": C.R2_REF / f"{i:05d}_mix.wav"}
            for k in (["ours"] if e == "E4" else SYS):
                p = C.R2_OUT / k / f"{i:05d}_{e}.wav"
                if p.exists():
                    conds[k] = p
            tid = f"{e}_{i:05d}"
            for name, p in conds.items():
                shutil.copy(p, LT / "stimuli" / f"{tid}_{name}.wav")
            w = words(st, sp, e)
            trials.append({"id": tid, "edit": e, "scene": i, **w,
                           "original": f"stimuli/{tid}_orig.wav",
                           "conditions": {n: f"stimuli/{tid}_{n}.wav" for n in conds}})
            ref, _ = sf.read(str(conds["gt"]))
            ledger.append({"id": tid, "edit": e, "scene": i, "spec": {k: v for k, v in sp.items() if not k.startswith("C")},
                           "alpha": st["alpha"], "room": st["room"], "sources": st["src"],
                           "gt_level_dbfs": dbfs(ref), "conditions": sorted(conds),
                           "text": w})
    cfg = {"title": "Audio edit listening test / 音频编辑听音测试",
           "intro_en": ("Each trial plays an ORIGINAL recording and describes one change that should be made to it. "
                        "Listen to the original first, then to every clip. For each clip, rate how much you agree "
                        "that it is the original with exactly this change and nothing else (0 = not at all, "
                        "100 = completely). Use headphones. Replay as often as you like."),
           "intro_zh": ("每一题先播放原录音，并说明应该对它做的一个改动。先听原录音，再听每个片段。"
                        "对每个片段打分：你在多大程度上认同它就是原录音做了这一改动而且没有其他变化"
                        "（0 完全不认同，100 完全认同）。请戴耳机，可反复播放。"),
           "trials": trials}
    (LT / "config.json").write_text(json.dumps(cfg, indent=1, ensure_ascii=False))
    (LT / "config.js").write_text("window.LT_CONFIG = " + json.dumps(cfg, ensure_ascii=False) + ";\n")
    # config is embedded so the page works even when opened without
    # config.js next to it (a preview in a browser, a cloud viewer)
    (LT / "index.html").write_text(HTML.replace(
        "__CONFIG__", "window.LT_CONFIG = " + json.dumps(cfg, ensure_ascii=False) + ";"))
    (LT / "README.txt").write_text(README)
    (LT / "analyze.py").write_text(ANALYZE)
    (C.PN / "p3_r2_listening_v2.json").write_text(json.dumps(
        {"meta": {"script": "src/p3_r2_listening.py", "design": "agreement 0..100 per clip, original shown, "
                  "hidden gt and orig", "filters": {"level_db": LEVEL_DB, "active_min": ACTIVE_MIN,
                  "active_max_transient": ACTIVE_MAX, "clap_rank_max": CLAP_RANK_MAX,
                  "quiet_db": QUIET_DB, "crest_impulsive": CREST_IMPULSIVE,
                  "low_freq_excluded_for_e4": sorted(LOW_FREQ), "e4_gap_min_frames": GAP_MIN, "e4_lead_max_s": LEAD_MAX,
                  "pilot_excluded": EXCLUDE}},
         "trials": ledger}, indent=1, ensure_ascii=False))
    print("package:", LT, "trials", len(trials))
    for t in ledger:
        print(t["id"], t["text"]["edit_en"])
        for r in t["sources"]:
            print(f"    {q_of(r['cls']):12s} rel {r['rel']:6.1f} dB active {r['active']*100:3.0f}% clap_rank {r['clap_rank']}")


HTML = r"""<!doctype html><html><head><meta charset='utf-8'><title>Listening test</title>
<style>body{font-family:sans-serif;max-width:900px;margin:auto;padding:16px;line-height:1.4}
.c{display:flex;align-items:center;gap:12px;margin:8px 0}.c input[type=range]{flex:1}
.box{background:#f4f4f4;padding:10px;border-radius:6px;margin:10px 0}
button{padding:8px 16px;font-size:15px}#done{display:none}.zh{color:#444}</style></head><body>
<h1 id='title'></h1><p id='intro_en'></p><p id='intro_zh' class='zh'></p>
<div id='trial'><h2 id='tname'></h2>
<div class='box'><b>Original / 原录音</b> <audio id='orig' controls></audio><p id='o_en'></p><p id='o_zh' class='zh'></p><p id='audio_err' style='color:#a00'></p></div>
<div class='box'><b>The change / 改动</b><p id='e_en'></p><p id='e_zh' class='zh'></p></div>
<p><b>How much do you agree that each clip is the original with exactly this change? / 你多大程度认同每个片段就是原录音做了这一改动？</b> (0 = not at all 完全不认同, 100 = completely 完全认同)</p>
<div id='conds'></div>
<button id='next'>Next / 下一题</button></div>
<div id='done'><h2>Thank you. / 谢谢。</h2><p>Download your answers and send the file back. / 请下载结果文件并发回。</p>
<button id='dl'>Download results / 下载结果</button></div>
<script>__CONFIG__</script>
<script>
let cfg, t = 0, res = {listener: '', answers: []};
const shuffle = a => {for (let i = a.length - 1; i > 0; i--) {const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]];} return a;};
const $ = id => document.getElementById(id);
function start(){ try { cfg = window.LT_CONFIG; $('title').textContent = cfg.title; $('intro_en').textContent = cfg.intro_en; $('intro_zh').textContent = cfg.intro_zh;
 res.listener = prompt('Listener ID (any name or number) / 听者编号') || 'anon'; res.started = new Date().toISOString();
 res.order = shuffle(cfg.trials.map((_, i) => i)); show(); } catch (err) {
 $('intro_en').textContent = 'The page could not start (' + err + '). Please unzip the whole folder and open index.html from it. / 页面无法启动，请先解压整个文件夹，再从解压后的文件夹打开 index.html。'; } }
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start); else start();
function show(){ if (t >= cfg.trials.length) {$('trial').style.display='none'; $('done').style.display='block'; return;}
 const tr = cfg.trials[res.order[t]]; $('tname').textContent = 'Trial ' + (t+1) + ' of ' + cfg.trials.length;
 $('orig').src = tr.original; $('o_en').textContent = tr.original_en; $('o_zh').textContent = tr.original_zh;
 $('audio_err').textContent = ''; $('orig').onerror = () => {$('audio_err').textContent = 'Audio file not found: keep the stimuli folder next to index.html. / 找不到音频，请保持 stimuli 文件夹和 index.html 在同一目录。';};
 $('e_en').textContent = tr.edit_en; $('e_zh').textContent = tr.edit_zh;
 const d = $('conds'); d.innerHTML = '';
 shuffle(Object.keys(tr.conditions)).forEach((k, n) => {d.innerHTML += `<div class='c'><span>Clip ${String.fromCharCode(65+n)}</span><audio controls src='${tr.conditions[k]}'></audio><input type='range' min='0' max='100' value='50' data-k='${k}'><span class='v'>50</span></div>`;});
 d.querySelectorAll('input').forEach(i => i.oninput = e => e.target.nextElementSibling.textContent = e.target.value);
 window.scrollTo(0, 0); }
$('next').onclick = () => {const tr = cfg.trials[res.order[t]]; const a = {trial: tr.id, edit: tr.edit, ratings: {}};
 document.querySelectorAll('#conds input').forEach(i => a.ratings[i.dataset.k] = +i.value); res.answers.push(a); t++; show();};
$('dl').onclick = () => {res.finished = new Date().toISOString(); const b = new Blob([JSON.stringify(res, null, 1)], {type: 'application/json'});
 const a = document.createElement('a'); a.href = URL.createObjectURL(b); a.download = 'listening_' + res.listener + '.json'; a.click();};
</script></body></html>
"""

README = """Audio edit listening test (ASG paper, TASLP revision), 10 trials, about 15 minutes.

1. Unzip. Keep index.html, config.js and the stimuli folder together.
2. Open index.html in Chrome, Edge, Firefox or Safari (double click). Use headphones.
3. Enter any listener ID (a name or a number).
4. Each trial plays the ORIGINAL recording and states one change that should be made to it.
   Listen to the original, then to every clip. For each clip rate how much you agree that it
   is the original with exactly this change and nothing else (0 = not at all, 100 = completely).
   Clips can be replayed freely. Some clips are the untouched original or the exact intended
   result; rate them by the same rule.
5. Press Next after each trial. At the end press Download results and send listening_<ID>.json back.

音频编辑听音测试，10 题，约 15 分钟。
1. 解压后保持 index.html、config.js 和 stimuli 文件夹在同一目录。
2. 双击 index.html 用浏览器打开，请戴耳机。
3. 输入任意听者编号。
4. 每一题先播放原录音，并说明应该做的一个改动。先听原录音，再听每个片段，
   对每个片段打分：你多大程度认同它就是原录音做了这一改动而且没有别的变化
   （0 完全不认同，100 完全认同）。可反复播放。有的片段就是原录音本身或正确结果，按同一标准打分即可。
5. 每题点 Next。做完点 Download results，把 listening_<编号>.json 发回。
"""

ANALYZE = r'''"""Aggregate result files: python analyze.py results/*.json
Per edit and condition: median agreement over listeners (each listener's
mean over trials), paired Wilcoxon against ours and against the hidden
ground truth (gt). Screening: a listener is dropped when the hidden
ground truth is rated below the untouched original in more than 30% of
trials (they could not follow the instruction)."""
import json, sys
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon

R = defaultdict(lambda: defaultdict(list))
kept = 0
subs = {}
for f in sys.argv[1:]:
    d = json.load(open(f))
    k = d["listener"].strip().lower()
    if k in subs and subs[k]["started"] <= d["started"]:
        print("duplicate listener id", d["listener"], "later submission ignored (first attempt counts)")
        continue
    if k in subs:
        print("duplicate listener id", d["listener"], "later submission ignored (first attempt counts)")
    subs[k] = d
for d in subs.values():
    bad = [a["ratings"].get("gt", 100) < a["ratings"].get("orig", 0) for a in d["answers"]]
    if np.mean(bad) > 0.3:
        print("excluded", d["listener"]); continue
    kept += 1
    for a in d["answers"]:
        for c, v in a["ratings"].items():
            R[(a["edit"], c)][d["listener"]].append(v)
print("listeners kept", kept)
for e in sorted({e for e, _ in R}):
    print("==", e)
    for c in ("gt", "orig", "ours", "x1", "x3", "x6"):
        if (e, c) not in R: continue
        x = [np.mean(v) for v in R[(e, c)].values()]
        line = f"{c:5s} median {np.median(x):5.1f} (n={len(x)})"
        for ref in ("ours", "gt"):
            if ref != c and (e, ref) in R and len(x) >= 5:
                y = [np.mean(R[(e, ref)][l]) for l in R[(e, c)]]
                line += f"  vs {ref} p={wilcoxon(x, y).pvalue:.3g}" if any(np.array(x) != np.array(y)) else f"  vs {ref} p=1"
        print(line)
'''


if __name__ == "__main__":
    main()
