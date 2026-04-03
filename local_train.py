"""
local_train.py
==============
Pre-configured launcher for RTX 4060 8 GB VRAM.
Calls train.py with settings tuned for your GPU.

Usage
-----
  python local_train.py                        # fresh start
  python local_train.py --resume               # auto-resume from latest.pth
  python local_train.py --resume --fast        # 128px mode (faster iteration)
  python local_train.py --epochs 50            # override epoch count
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# ── RTX 4060 8 GB – recommended settings ──────────────────────────────────
#
#  256×256, batch=1 : ~2.8 GB VRAM   ← safe, standard quality
#  256×256, batch=2 : ~5.2 GB VRAM   ← good speed/quality trade-off
#  256×256, batch=4 : ~9.8 GB VRAM   ← OOM on 8 GB, don't use
#  128×128, batch=4 : ~3.1 GB VRAM   ← fast prototyping / smoke-test
#
# With batch=1 on RTX 4060 expect ~0.15 s/iter → ~3–5 min/epoch
# (depends on dataset size)
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    data_root      = "./dataset",
    checkpoint_dir = "./checkpoints",
    sample_dir     = "./samples",
    image_size     = 256,
    batch_size     = 2,       # safe for 8 GB at 256px
    epochs         = 200,
    lr             = 2e-4,
    beta1          = 0.5,
    lambda_cyc     = 10,
    lambda_id      = 5,
    ngf            = 64,
    ndf            = 64,
    n_res          = 9,
    buffer_size    = 50,
    num_workers    = 0,       # 0 required on Windows (avoids DataLoader multiprocessing freeze)
    save_every     = 1,
    sample_every   = 200,
)

FAST_OVERRIDES = dict(
    image_size  = 128,
    batch_size  = 4,
    n_res       = 6,
    ngf         = 32,
    ndf         = 32,
)


def main():
    p = argparse.ArgumentParser(
        description="RTX 4060 local CycleGAN launcher"
    )
    p.add_argument("--resume", action="store_true",
                   help="Auto-resume from checkpoints/latest.pth if it exists")
    p.add_argument("--fast", action="store_true",
                   help="128px fast-iteration mode (lower quality, quicker epochs)")
    p.add_argument("--data_root",  default=DEFAULTS["data_root"])
    p.add_argument("--epochs",     type=int, default=DEFAULTS["epochs"])
    p.add_argument("--batch_size", type=int, default=None,
                   help="Override batch size (default: 2 for 256px, 4 for 128px)")
    args = p.parse_args()

    cfg = dict(DEFAULTS)
    if args.fast:
        cfg.update(FAST_OVERRIDES)
        print("[local_train] Fast mode: 128×128, ngf=32, batch=4")
    if args.batch_size:
        cfg["batch_size"] = args.batch_size
    cfg["data_root"] = args.data_root
    cfg["epochs"]    = args.epochs

    # Directories
    Path(cfg["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["sample_dir"]).mkdir(parents=True, exist_ok=True)

    # Resume flag
    resume_flag = []
    latest = os.path.join(cfg["checkpoint_dir"], "latest.pth")
    if args.resume and os.path.isfile(latest):
        resume_flag = ["--resume", latest]
        print(f"[local_train] Resuming from {latest}")
    elif args.resume:
        print(f"[local_train] No checkpoint found at {latest} – starting fresh.")

    # Print GPU info
    try:
        import torch
        if torch.cuda.is_available():
            dev = torch.cuda.get_device_properties(0)
            print(f"[local_train] GPU  : {dev.name}")
            print(f"[local_train] VRAM : {dev.total_memory / 1e9:.1f} GB")
            # Rough VRAM estimate
            vram_est = cfg["batch_size"] * (2.8 if cfg["image_size"] == 256 else 0.9)
            print(f"[local_train] Est. VRAM usage : ~{vram_est:.1f} GB")
        else:
            print("[local_train] WARNING: No CUDA GPU detected – training on CPU will be very slow.")
    except ImportError:
        pass

    print(f"\n[local_train] Config:")
    for k, v in cfg.items():
        print(f"  {k:>16} = {v}")
    print()

    # Build command
    cmd = [sys.executable, "train.py"] + resume_flag
    for k, v in cfg.items():
        cmd += [f"--{k}", str(v)]

    print("[local_train] Running:", " ".join(cmd[:6]), "…\n")

    # Enable TF32 for Ampere+ GPUs (RTX 30/40 series) – free ~10% speedup
    env = os.environ.copy()
    env["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "1"

    try:
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        print("\n[local_train] Training interrupted. Latest checkpoint saved.")


if __name__ == "__main__":
    main()
