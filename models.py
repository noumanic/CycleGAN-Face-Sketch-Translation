"""
CycleGAN Model Architecture
============================
Generator  : ResNet-based (9 residual blocks for 256x256 images)
Discriminator: PatchGAN (70x70 receptive field)

Reference: "Unpaired Image-to-Image Translation using Cycle-Consistent
           Adversarial Networks" – Zhu et al., 2017
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building Blocks
# ---------------------------------------------------------------------------

class ConvNormReLU(nn.Module):
    """Conv → InstanceNorm → ReLU (or LeakyReLU)."""

    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1,
                 norm=True, activation="relu", reflection_pad=False):
        super().__init__()
        layers = []

        if reflection_pad:
            layers.append(nn.ReflectionPad2d(padding))
            padding = 0

        layers.append(nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=not norm))

        if norm:
            layers.append(nn.InstanceNorm2d(out_ch))

        if activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif activation == "leaky":
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        # activation == "none" → no activation added

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    """ResNet residual block with InstanceNorm and reflection padding."""

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)   # skip connection


# ---------------------------------------------------------------------------
# Generator  (ResNet-9)
# ---------------------------------------------------------------------------

class Generator(nn.Module):
    """
    ResNet-based Generator.
    Architecture:
        c7s1-64 → d128 → d256 → R256×9 → u128 → u64 → c7s1-3
    """

    def __init__(self, in_channels=3, out_channels=3,
                 ngf=64, n_res_blocks=9):
        super().__init__()

        # ---- Encoder ----
        encoder = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, 7, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
            # Downsampling ×2
            nn.Conv2d(ngf,     ngf * 2, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf * 2, ngf * 4, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(inplace=True),
        ]

        # ---- Residual Blocks ----
        res_blocks = [ResidualBlock(ngf * 4) for _ in range(n_res_blocks)]

        # ---- Decoder ----
        decoder = [
            # Upsampling ×2
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, stride=2,
                               padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 2, ngf, 3, stride=2,
                               padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
            # Output layer
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_channels, 7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*encoder, *res_blocks, *decoder)

    def forward(self, x):
        return self.model(x)


# ---------------------------------------------------------------------------
# Discriminator  (PatchGAN 70×70)
# ---------------------------------------------------------------------------

class Discriminator(nn.Module):
    """
    PatchGAN discriminator.
    Architecture:
        C64 → C128 → C256 → C512 → output patch
    No InstanceNorm on the first layer (as per original paper).
    """

    def __init__(self, in_channels=3, ndf=64):
        super().__init__()

        self.model = nn.Sequential(
            # Layer 1 – no norm
            nn.Conv2d(in_channels, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 2
            nn.Conv2d(ndf,     ndf * 2, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 3
            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 4
            nn.Conv2d(ndf * 4, ndf * 8, 4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # Output  → 1-channel patch map
            nn.Conv2d(ndf * 8, 1, 4, stride=1, padding=1),
        )

    def forward(self, x):
        return self.model(x)


# ---------------------------------------------------------------------------
# Weight Initialisation
# ---------------------------------------------------------------------------

def weights_init(m):
    """Gaussian initialisation (mean=0, std=0.02) for Conv / BatchNorm."""
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif "InstanceNorm2d" in classname and m.weight is not None:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    G = Generator().to(device)
    D = Discriminator().to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    x = torch.randn(1, 3, 256, 256).to(device)
    fake = G(x)
    score = D(fake)

    print(f"Generator   input : {x.shape}     output : {fake.shape}")
    print(f"Discriminator output : {score.shape}")

    total_G = sum(p.numel() for p in G.parameters()) / 1e6
    total_D = sum(p.numel() for p in D.parameters()) / 1e6
    print(f"Generator params   : {total_G:.2f} M")
    print(f"Discriminator params: {total_D:.2f} M")
