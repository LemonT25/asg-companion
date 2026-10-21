"""A13: no-edit anchors for the Phi_B cohort (anchors involve only the
world's physics, so full Phi_B patching is correct here)."""
import sys
sys.path.insert(0, "/workspace/CF_ASG/src")
import p3_phiB  # noqa: E402
p3_phiB.apply_phiB()
import importlib  # noqa: E402
import p2_common  # noqa: E402
importlib.reload(p2_common)  # rebind under patched core
p3_phiB.apply_phiB()          # re-apply p2_common-level patches
import p3_edit_eval  # noqa: E402
importlib.reload(p3_edit_eval)
import p3_noop_anchor  # noqa: E402
importlib.reload(p3_noop_anchor)
sys.argv = ["p3_noop_anchor.py", "--ledger", "p3_edit_eval_mmB.json",
            "--split", "mmB", "--out", "p3_noop_anchor_mmB.json"]
p3_noop_anchor.main()
