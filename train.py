"""
CycleGAN Training Script
=========================
Trains two generators (G_AB: face→sketch, G_BA: sketch→face) and two
discriminators (D_A: real vs fake face, D_B: real vs fake sketch).

Losses
------
  GAN loss      : LSGAN (MSE) as in the original paper
  Cycle loss    : L1, weighted by λ_cyc  (default 10)
  Identity loss : L1, weighted by λ_id   (default 5)

Usage
-----
  python train.py --data_root ./dataset --epochs 200 --batch_size 1

Colab tip
---------
  Mount your Google Drive and point --checkpoint_dir to a Drive path so
  weights are preserved across runtime restarts.
"""

import argparse
import itertools
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
# AMP: torch.cuda.amp is deprecated in PyTorch 2.x, use torch.amp
from torch.optim.lr_scheduler import LambdaLR
from torchvision.utils import save_image

from models import Generator, Discriminator, weights_init
from dataset import build_dataloaders, ImageBuffer

# RTX 30/40-series optimisations
# TF32 gives ~1.5× speedup on Ampere/Ada GPUs with negligible quality loss
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32        = True
torch.backends.cudnn.benchmark         = True   # auto-tune convolution kernels


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tensor_to_01(t: torch.Tensor) -> torch.Tensor:
    """Convert [-1,1] tensor to [0,1] for saving."""
    return (t * 0.5 + 0.5).clamp(0, 1)


def lambda_rule(epoch, n_epochs, n_epochs_decay):
    """Learning rate schedule: constant for first half, linear decay."""
    if epoch < n_epochs:
        return 1.0
    return max(0.0, 1.0 - (epoch - n_epochs) / max(n_epochs_decay, 1))


def save_checkpoint(state: dict, path: str):
    torch.save(state, path)
    print(f"  ✓ Checkpoint saved → {path}")


def load_checkpoint(path: str, device: torch.device) -> dict:
    ckpt = torch.load(path, map_location=device)
    print(f"  ✓ Loaded checkpoint  ← {path}  (epoch {ckpt['epoch']})")
    return ckpt


# ---------------------------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="CycleGAN – Face ↔ Sketch")

    # Paths
    p.add_argument("--data_root",      default="./dataset",
                   help="Root folder with trainA/trainB/testA/testB sub-dirs")
    p.add_argument("--checkpoint_dir", default="./checkpoints",
                   help="Where to save model weights")
    p.add_argument("--sample_dir",     default="./samples",
                   help="Where to save training samples")
    p.add_argument("--resume",         default=None,
                   help="Path to checkpoint to resume from")

    # Training hyper-parameters
    p.add_argument("--image_size",  type=int,   default=256)
    p.add_argument("--batch_size",  type=int,   default=1)
    p.add_argument("--epochs",      type=int,   default=200,
                   help="Total epochs (first half constant LR, second half decay)")
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--beta1",       type=float, default=0.5)
    p.add_argument("--lambda_cyc",  type=float, default=10.0,
                   help="Cycle-consistency loss weight")
    p.add_argument("--lambda_id",   type=float, default=5.0,
                   help="Identity loss weight")
    p.add_argument("--buffer_size", type=int,   default=50,
                   help="Size of image history buffer")
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--ngf",         type=int,   default=64,
                   help="Generator base filters")
    p.add_argument("--ndf",         type=int,   default=64,
                   help="Discriminator base filters")
    p.add_argument("--n_res",       type=int,   default=9,
                   help="Number of residual blocks in generator")
    p.add_argument("--save_every",  type=int,   default=1,
                   help="Save checkpoint every N epochs")
    p.add_argument("--sample_every",type=int,   default=100,
                   help="Save sample images every N iterations")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"  CycleGAN  –  Face ↔ Sketch  |  device: {device}")
    print(f"{'='*55}\n")

    # ---- Directories ----
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.sample_dir).mkdir(parents=True, exist_ok=True)

    # ---- Data ----
    train_loader, test_loader = build_dataloaders(
        args.data_root, args.image_size, args.batch_size, args.num_workers
    )
    if train_loader is None:
        raise RuntimeError(
            f"No training data found under {args.data_root}. "
            "Expected sub-dirs: trainA/ and trainB/"
        )

    # ---- Models ----
    G_AB = Generator(ngf=args.ngf, n_res_blocks=args.n_res).to(device)  # face → sketch
    G_BA = Generator(ngf=args.ngf, n_res_blocks=args.n_res).to(device)  # sketch → face
    D_A  = Discriminator(ndf=args.ndf).to(device)   # discriminates real faces
    D_B  = Discriminator(ndf=args.ndf).to(device)   # discriminates real sketches

    G_AB.apply(weights_init)
    G_BA.apply(weights_init)
    D_A.apply(weights_init)
    D_B.apply(weights_init)

    # ---- Losses ----
    criterion_GAN   = nn.MSELoss()          # LSGAN
    criterion_cycle = nn.L1Loss()
    criterion_id    = nn.L1Loss()

    # ---- Optimisers ----
    n_epochs_decay = args.epochs // 2
    n_epochs_const = args.epochs - n_epochs_decay

    opt_G = torch.optim.Adam(
        itertools.chain(G_AB.parameters(), G_BA.parameters()),
        lr=args.lr, betas=(args.beta1, 0.999)
    )
    opt_D_A = torch.optim.Adam(D_A.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    opt_D_B = torch.optim.Adam(D_B.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    sched_G   = LambdaLR(opt_G,   lr_lambda=lambda e: lambda_rule(e, n_epochs_const, n_epochs_decay))
    sched_D_A = LambdaLR(opt_D_A, lr_lambda=lambda e: lambda_rule(e, n_epochs_const, n_epochs_decay))
    sched_D_B = LambdaLR(opt_D_B, lr_lambda=lambda e: lambda_rule(e, n_epochs_const, n_epochs_decay))

    # ---- Mixed-precision (AMP) – RTX 4060 optimised ----
    # Reduces VRAM ~40% and speeds up training ~1.4x on Ada Lovelace GPUs
    use_amp = (device.type == "cuda")
    scaler_G   = torch.amp.GradScaler(device="cuda", enabled=use_amp)
    scaler_D_A = torch.amp.GradScaler(device="cuda", enabled=use_amp)
    scaler_D_B = torch.amp.GradScaler(device="cuda", enabled=use_amp)
    if use_amp:
        print("[Train] AMP mixed-precision enabled (RTX 4060 optimised) ✓")

    # ---- Image Buffers ----
    buffer_A = ImageBuffer(args.buffer_size)
    buffer_B = ImageBuffer(args.buffer_size)

    # ---- Resume ----
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = load_checkpoint(args.resume, device)
        G_AB.load_state_dict(ckpt["G_AB"])
        G_BA.load_state_dict(ckpt["G_BA"])
        D_A.load_state_dict(ckpt["D_A"])
        D_B.load_state_dict(ckpt["D_B"])
        opt_G.load_state_dict(ckpt["opt_G"])
        opt_D_A.load_state_dict(ckpt["opt_D_A"])
        opt_D_B.load_state_dict(ckpt["opt_D_B"])
        start_epoch = ckpt["epoch"] + 1
        # Fast-forward LR schedulers
        for _ in range(start_epoch):
            sched_G.step(); sched_D_A.step(); sched_D_B.step()

    # ---- Target tensors (real/fake labels for LSGAN) ----
    def make_target(pred, is_real):
        val = 1.0 if is_real else 0.0
        return torch.full_like(pred, val, device=device)

    # ---- Training ----
    print(f"Starting training from epoch {start_epoch + 1}/{args.epochs}\n")
    total_iters = len(train_loader)

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        running = {"G": 0.0, "D_A": 0.0, "D_B": 0.0}

        G_AB.train(); G_BA.train(); D_A.train(); D_B.train()

        for i, batch in enumerate(train_loader):
            real_A = batch["A"].to(device)   # real face
            real_B = batch["B"].to(device)   # real sketch

            # ============================================================
            #   (1) Update Generators  G_AB  and  G_BA
            # ============================================================
            # Freeze discriminators while updating generators
            for p in D_A.parameters(): p.requires_grad_(False)
            for p in D_B.parameters(): p.requires_grad_(False)

            opt_G.zero_grad()

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                # --- Identity losses ---
                id_B = G_AB(real_B)
                loss_id_B = criterion_id(id_B, real_B) * args.lambda_id
                id_A = G_BA(real_A)
                loss_id_A = criterion_id(id_A, real_A) * args.lambda_id

                # --- GAN losses ---
                fake_B = G_AB(real_A)
                pred_fake_B = D_B(fake_B)
                loss_GAN_AB = criterion_GAN(pred_fake_B, make_target(pred_fake_B, True))
                fake_A = G_BA(real_B)
                pred_fake_A = D_A(fake_A)
                loss_GAN_BA = criterion_GAN(pred_fake_A, make_target(pred_fake_A, True))

                # --- Cycle losses ---
                rec_A = G_BA(fake_B)
                loss_cyc_A = criterion_cycle(rec_A, real_A) * args.lambda_cyc
                rec_B = G_AB(fake_A)
                loss_cyc_B = criterion_cycle(rec_B, real_B) * args.lambda_cyc

                loss_G = (loss_GAN_AB + loss_GAN_BA
                          + loss_cyc_A + loss_cyc_B
                          + loss_id_A  + loss_id_B)

            scaler_G.scale(loss_G).backward()
            scaler_G.step(opt_G)
            scaler_G.update()

            # ============================================================
            #   (2) Update Discriminator D_A  (real faces)
            # ============================================================
            for p in D_A.parameters(): p.requires_grad_(True)
            opt_D_A.zero_grad()

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                pred_real_A  = D_A(real_A)
                loss_real_A  = criterion_GAN(pred_real_A, make_target(pred_real_A, True))
                fake_A_buf   = buffer_A.push_and_pop(fake_A.detach())
                pred_fake_A2 = D_A(fake_A_buf)
                loss_fake_A  = criterion_GAN(pred_fake_A2, make_target(pred_fake_A2, False))
                loss_D_A = (loss_real_A + loss_fake_A) * 0.5

            scaler_D_A.scale(loss_D_A).backward()
            scaler_D_A.step(opt_D_A)
            scaler_D_A.update()

            # ============================================================
            #   (3) Update Discriminator D_B  (real sketches)
            # ============================================================
            for p in D_B.parameters(): p.requires_grad_(True)
            opt_D_B.zero_grad()

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                pred_real_B  = D_B(real_B)
                loss_real_B  = criterion_GAN(pred_real_B, make_target(pred_real_B, True))
                fake_B_buf   = buffer_B.push_and_pop(fake_B.detach())
                pred_fake_B2 = D_B(fake_B_buf)
                loss_fake_B  = criterion_GAN(pred_fake_B2, make_target(pred_fake_B2, False))
                loss_D_B = (loss_real_B + loss_fake_B) * 0.5

            scaler_D_B.scale(loss_D_B).backward()
            scaler_D_B.step(opt_D_B)
            scaler_D_B.update()

            # ---- Logging ----
            running["G"]   += loss_G.item()
            running["D_A"] += loss_D_A.item()
            running["D_B"] += loss_D_B.item()

            if (i + 1) % args.sample_every == 0:
                G_AB.eval(); G_BA.eval()
                with torch.no_grad():
                    sample_fake_B = G_AB(real_A)
                    sample_fake_A = G_BA(real_B)
                grid = torch.cat([
                    tensor_to_01(real_A),
                    tensor_to_01(sample_fake_B),
                    tensor_to_01(real_B),
                    tensor_to_01(sample_fake_A),
                ], dim=0)
                save_image(grid,
                           f"{args.sample_dir}/epoch{epoch+1:03d}_iter{i+1:05d}.png",
                           nrow=args.batch_size)
                G_AB.train(); G_BA.train()

            print(f"\r  Epoch [{epoch+1:3d}/{args.epochs}]  "
                  f"[{i+1:4d}/{total_iters}]  "
                  f"G: {loss_G.item():.3f}  "
                  f"D_A: {loss_D_A.item():.3f}  "
                  f"D_B: {loss_D_B.item():.3f}",
                  end="", flush=True)

        # ---- End of epoch ----
        elapsed = time.time() - epoch_start
        avg_G   = running["G"]   / total_iters
        avg_D_A = running["D_A"] / total_iters
        avg_D_B = running["D_B"] / total_iters
        print(f"\n  → Epoch {epoch+1:3d} done in {elapsed:.0f}s  "
              f"| avg G={avg_G:.4f}  D_A={avg_D_A:.4f}  D_B={avg_D_B:.4f}")

        sched_G.step(); sched_D_A.step(); sched_D_B.step()

        # ---- Checkpoint ----
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(
                args.checkpoint_dir, f"cyclegan_epoch{epoch+1:03d}.pth"
            )
            save_checkpoint({
                "epoch":   epoch,
                "G_AB":    G_AB.state_dict(),
                "G_BA":    G_BA.state_dict(),
                "D_A":     D_A.state_dict(),
                "D_B":     D_B.state_dict(),
                "opt_G":   opt_G.state_dict(),
                "opt_D_A": opt_D_A.state_dict(),
                "opt_D_B": opt_D_B.state_dict(),
            }, ckpt_path)
            # Also keep a "latest" alias for easy resuming
            latest = os.path.join(args.checkpoint_dir, "latest.pth")
            save_checkpoint({
                "epoch":   epoch,
                "G_AB":    G_AB.state_dict(),
                "G_BA":    G_BA.state_dict(),
                "D_A":     D_A.state_dict(),
                "D_B":     D_B.state_dict(),
                "opt_G":   opt_G.state_dict(),
                "opt_D_A": opt_D_A.state_dict(),
                "opt_D_B": opt_D_B.state_dict(),
            }, latest)

    print("\n\nTraining complete!")


if __name__ == "__main__":
    main()