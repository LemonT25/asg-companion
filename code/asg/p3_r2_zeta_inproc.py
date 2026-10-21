"""In-process ZETA (DDPM-inversion editing, ICML'24) for the A14 replay.

Replicates AudioEditingCode/code/main_run.py (mode 'ours', eta 1,
numerical_fix, no cutoff points) but loads the AudioLDM2 pipeline ONCE
instead of once per edit. Same settings as the main ledger
(num_diffusion_steps 200, tstart 100, cfg_src 3, cfg_tar 12, seed
rule 42 + spec_index). Runs under /workspace/venvs/zeta.
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

ZETA_CODE = Path("/workspace/CF_AQA/AudioEditingCode/code")
sys.path.insert(0, str(ZETA_CODE))
os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
os.environ.setdefault("WANDB_MODE", "disabled")
from torch import inference_mode  # noqa: E402
from ddm_inversion.inversion_utils import (inversion_forward_process,  # noqa
                                           inversion_reverse_process)
from models import load_model  # noqa: E402
from utils import set_reproducability, load_audio  # noqa: E402

CFG = {"num_diffusion_steps": 200, "tstart": 100, "cfg_src": [3.0],
       "cfg_tar": [12.0], "fix_alpha": 0.1}


class ZetaInProc:
    def __init__(self, model_id="cvssp/audioldm2", device="cuda:0"):
        self.device = device
        self.model_id = model_id
        torch.cuda.set_device(0)
        self.ldm = load_model(model_id, device, CFG["num_diffusion_steps"],
                              token=None)

    def edit(self, wav_path, src_prompt, tar_prompt, seed, out_path):
        """Edit wav_path from src_prompt to tar_prompt; writes out_path
        (16 kHz mono wav); returns True on success."""
        cwd = os.getcwd()
        os.chdir(ZETA_CODE)          # repo-relative resources
        try:
            set_reproducability(seed, extreme=False)
            tstart = torch.tensor([CFG["tstart"]], dtype=torch.int)
            skip = CFG["num_diffusion_steps"] - tstart
            x0, sr, duration = load_audio(str(wav_path), self.ldm.get_fn_STFT(),
                                          device=self.device, stft=True,
                                          model_sr=self.ldm.get_sr())
            torch.cuda.empty_cache()
            with inference_mode():
                w0 = self.ldm.vae_encode(x0)
                wt, zs, wts, extra = inversion_forward_process(
                    self.ldm, w0, etas=1.0, prompts=[src_prompt],
                    cfg_scales=CFG["cfg_src"], prog_bar=False,
                    num_inference_steps=CFG["num_diffusion_steps"],
                    cutoff_points=None, numerical_fix=True, duration=duration)
                w0, _ = inversion_reverse_process(
                    self.ldm, xT=wts, tstart=tstart, fix_alpha=CFG["fix_alpha"],
                    etas=1.0, prompts=[tar_prompt], neg_prompts=[""],
                    cfg_scales=CFG["cfg_tar"], prog_bar=False,
                    zs=zs[:int(CFG["num_diffusion_steps"] - int(skip.min()))],
                    cutoff_points=None, duration=duration, extra_info=extra)
                x0_dec = self.ldm.vae_decode(w0)
                if x0_dec.dim() < 4:
                    x0_dec = x0_dec[None, :, :, :]
                with torch.no_grad():
                    audio = self.ldm.decode_to_mel(x0_dec)
            torchaudio.save(str(out_path), audio.detach().cpu(), sample_rate=sr)
            return True
        except Exception as ex:   # noqa: BLE001
            print(f"ZETA FAIL {type(ex).__name__}: {ex}", flush=True)
            return False
        finally:
            os.chdir(cwd)
