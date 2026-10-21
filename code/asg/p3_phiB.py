"""A13 physics-mismatch wrapper: patches the WORLD's renderer to
Phi_B, leaves the system's Phi_A machinery untouched (by-value
imports), then runs the requested tool.

Phi_B: c_s 343->350; air 0.0052 f^1.5 -> 0.0078 f^1.7 dB/m;
pass-by d0 2.0->3.0 m; room 30x20x8 -> 24x16x6, src (14,9,1.5),
mic (11,7,1.6).
Usage:
  p3_phiB.py scenes  --n 250 --seed 2         (render mmB cohort)
  p3_phiB.py evalref -- <p3_edit_eval args>   (Phi_B refs, Phi_A system)
  p3_phiB.py noop    -- <p3_noop_anchor args>
"""
import sys

sys.path.insert(0, "/workspace/audiocf_render")
sys.path.insert(0, "/workspace/CF_PhysAudio/src")
sys.path.insert(0, "/workspace/CF_ASG/src")

import numpy as np  # noqa: E402
import core  # noqa: E402
import p2_common  # noqa: E402


def apply_phiB():
    core.C_SOUND = 350.0
    core.air_absorption_db_per_m = lambda f: 0.0078 * f ** 1.7
    p2_common.ROOM = (24.0, 16.0, 6.0)
    p2_common.RIR_SRC = (14.0, 9.0, 1.5)
    p2_common.RIR_MIC = (11.0, 7.0, 1.6)
    p2_common._RIR_CACHE = {}
    orig_pos = core.passby_positions

    def pos_d0(n, sr, v, d0=2.0, t_close=None):
        return orig_pos(n, sr, v=v, d0=3.0, t_close=t_close)
    core.passby_positions = pos_d0

    def phi_velocity_B(dry, v):
        n = len(dry)
        pos = pos_d0(n, p2_common.SR, v=v, t_close=n / p2_common.SR / 2)
        return core.render_moving_source(dry, p2_common.SR, pos)
    p2_common.phi_velocity = phi_velocity_B


def main():
    mode = sys.argv[1]
    rest = sys.argv[2:]
    if rest and rest[0] == "--":
        rest = rest[1:]
    if mode == "scenes":
        apply_phiB()
        import p3_scenes
        sys.argv = ["p3_scenes.py", "--split", "mmB"] + rest
        p3_scenes.main()
    elif mode == "evalref":
        # build Phi_B ref renderer, restore Phi_A, run eval with hook
        import copy
        origs = (core.C_SOUND, core.air_absorption_db_per_m,
                 p2_common.ROOM, p2_common.RIR_SRC, p2_common.RIR_MIC,
                 core.passby_positions, p2_common.phi_velocity,
                 dict(p2_common._RIR_CACHE))
        apply_phiB()
        pvB = p2_common.phi_velocity
        pdB = p2_common.phi_distance      # distance uses core globals
        prB_cache = {}

        def phi_reverb_B(dry, alpha):
            from scipy.signal import fftconvolve
            from room import image_source_rir
            if alpha not in prB_cache:
                prB_cache[alpha] = image_source_rir(
                    (24.0, 16.0, 6.0), (14.0, 9.0, 1.5), (11.0, 7.0, 1.6),
                    p2_common.SR, alpha=alpha, max_order=22, rir_len_s=2.5)
            y = fftconvolve(dry, prB_cache[alpha])
            return y[:len(dry) + int(2.5 * p2_common.SR)]
        coreB_C = core.C_SOUND
        airB = core.air_absorption_db_per_m
        posB = core.passby_positions

        def render_node_B(d, traj, theta, alpha, env=1.0):
            # re-apply Phi_B globals around the call (system runs Phi_A)
            sc, sa, sp = core.C_SOUND, core.air_absorption_db_per_m, \
                core.passby_positions
            core.C_SOUND, core.air_absorption_db_per_m, \
                core.passby_positions = coreB_C, airB, posB
            try:
                w = pvB(d, theta) if traj == "passby" else pdB(d, theta)
                SEG = 64000
                if env > 0.5:
                    w = phi_reverb_B(w[:SEG], alpha)[:SEG]
                return w[:SEG]
            finally:
                core.C_SOUND, core.air_absorption_db_per_m, \
                    core.passby_positions = sc, sa, sp
        # restore Phi_A for everything else
        (core.C_SOUND, core.air_absorption_db_per_m, p2_common.ROOM,
         p2_common.RIR_SRC, p2_common.RIR_MIC, core.passby_positions,
         p2_common.phi_velocity) = origs[:7]
        p2_common._RIR_CACHE = origs[7]
        import p3_edit_eval
        p3_edit_eval.REF_RENDER = render_node_B
        sys.argv = ["p3_edit_eval.py"] + rest
        p3_edit_eval.main()
    elif mode == "noop":
        # Phi_B anchors: same ref renderer as evalref
        sys.argv = ["p3_phiB.py", "evalref"]  # reuse builder below
        raise SystemExit("use noop2")
    else:
        raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
