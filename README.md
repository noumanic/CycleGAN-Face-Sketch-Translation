# CycleGAN — Person Face ↔ Sketch Translation

Unpaired image-to-image translation between real face photographs and pencil sketches,
implemented from scratch in PyTorch following the original paper:

> **Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks**
> Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros — ICCV 2017

Dataset: [Person Face Sketches — Kaggle](https://www.kaggle.com/datasets/almightyj/person-face-sketches)

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Architecture Overview](#architecture-overview)
3. [Generator — ResNet-9](#generator--resnet-9)
4. [Discriminator — PatchGAN-70](#discriminator--patchgan-70)
5. [Loss Functions](#loss-functions)
6. [Quick Start](#quick-start)
7. [Training on RTX 4060](#training-on-rtx-4060)
8. [Inference & CLI](#inference--cli)
9. [Web Interface](#web-interface)
10. [API Reference](#api-reference)
11. [Hyperparameter Guide](#hyperparameter-guide)
12. [Architecture Diagrams](#architecture-diagrams)

---

## Project Structure

```
cyclegan_face_sketch/
│
├── models.py              Generator (ResNet-9) + Discriminator (PatchGAN)
├── dataset.py             Dataset loader, ImageBuffer, transforms
├── train.py               Training loop — AMP, TF32, checkpointing
├── inference.py           Domain auto-detection + translation
├── app.py                 Flask web application
├── predict.py             CLI inference tool (batch + single image)
│
├── local_train.py         RTX 4060 pre-configured launcher
├── setup_dataset.py       Auto-organise downloaded dataset
├── train_local.bat        Windows one-click launcher
├── train_local.sh         Linux/macOS one-click launcher
│
├── templates/
│   └── index.html         Browser UI (upload tab + camera tab)
│
├── architecture/          Mermaid diagrams (.mmd)
│   ├── 01_system_overview.mmd
│   ├── 02_training_loop.mmd
│   ├── 03_generator_arch.mmd
│   ├── 04_discriminator_arch.mmd
│   ├── 05_inference_pipeline.mmd
│   └── 06_file_dependency.mmd
│
├── requirements.txt
└── README.md
```

---

## Architecture Overview

```
Domain A (Faces)    Domain B (Sketches)
      |                     |
      v                     v
  +-------+            +-------+
  |  D_A  |            |  D_B  |
  |PatchGAN|           |PatchGAN|
  +-------+            +-------+
      ^                     ^
      |    +---------+      |
  fake_A <--+  G_BA   +<-- real_B
            | Sketch  |
            |  ->Face |
            +---------+

  real_A --> +---------+ --> fake_B
             |  G_AB   |
             | Face -> |
             | Sketch  |
             +---------+
                  |
                  v
  Cycle:  G_BA(fake_B) ≈ real_A
  Cycle:  G_AB(fake_A) ≈ real_B
```

Two generators and two discriminators are trained simultaneously:

| Network | Role | Parameters |
|---------|------|-----------|
| G_AB | Face → Sketch translation | 11.37 M |
| G_BA | Sketch → Face translation | 11.37 M |
| D_A | Discriminates real vs fake faces | 2.76 M |
| D_B | Discriminates real vs fake sketches | 2.76 M |
| **Total** | | **28.26 M** |

---

## Generator — ResNet-9

Input: `3 x 256 x 256`

### Encoder (downsampling)

| Layer | Operation | Output Shape |
|-------|-----------|-------------|
| 1 | ReflectionPad2d(3) | 3 x 262 x 262 |
| 2 | Conv2d(3->64, k=7, s=1) + InstanceNorm + ReLU | 64 x 256 x 256 |
| 3 | Conv2d(64->128, k=3, s=2) + InstanceNorm + ReLU | 128 x 128 x 128 |
| 4 | Conv2d(128->256, k=3, s=2) + InstanceNorm + ReLU | 256 x 64 x 64 |

### Residual Blocks (x9)

Each block:
```
x  ->  ReflPad(1) -> Conv(256->256, k=3) -> IN -> ReLU
    -> ReflPad(1) -> Conv(256->256, k=3) -> IN
    -> x + F(x)                              (skip connection)
```

Feature maps remain at `256 x 64 x 64` throughout all 9 blocks.

### Decoder (upsampling)

| Layer | Operation | Output Shape |
|-------|-----------|-------------|
| 1 | ConvTranspose2d(256->128, k=3, s=2) + IN + ReLU | 128 x 128 x 128 |
| 2 | ConvTranspose2d(128->64, k=3, s=2) + IN + ReLU | 64 x 256 x 256 |
| 3 | ReflectionPad2d(3) | 64 x 262 x 262 |
| 4 | Conv2d(64->3, k=7) + Tanh | 3 x 256 x 256 |

Output: `3 x 256 x 256` in range `[-1, 1]`

---

## Discriminator — PatchGAN-70

Rather than scoring a whole image, PatchGAN produces a `1 x 30 x 30` map
where each value scores one `70 x 70` overlapping patch of the input.

| Layer | Operation | Output Shape | Receptive Field |
|-------|-----------|-------------|----------------|
| 1 | Conv2d(3->64, k=4, s=2) + LeakyReLU(0.2) | 64 x 128 x 128 | 4 x 4 |
| 2 | Conv2d(64->128, k=4, s=2) + IN + LeakyReLU | 128 x 64 x 64 | 10 x 10 |
| 3 | Conv2d(128->256, k=4, s=2) + IN + LeakyReLU | 256 x 32 x 32 | 22 x 22 |
| 4 | Conv2d(256->512, k=4, s=1) + IN + LeakyReLU | 512 x 31 x 31 | 46 x 46 |
| 5 | Conv2d(512->1, k=4, s=1) | **1 x 30 x 30** | **70 x 70** |

Note: Layer 1 has no InstanceNorm, as specified in the original paper.

---

## Loss Functions

### GAN Loss — LSGAN (Least Squares)

Using MSE instead of BCE for more stable gradients:

```
L_GAN(G,D,X,Y) = 0.5 * E[(D(y)-1)^2] + 0.5 * E[(D(G(x))-0)^2]
```

### Cycle-Consistency Loss — L1, lambda=10

```
L_cyc(G,F) = E[||F(G(x)) - x||_1] + E[||G(F(y)) - y||_1]
```

Ensures that translating A->B->A reconstructs the original A.

### Identity Loss — L1, lambda=5

```
L_id = E[||G_AB(y) - y||_1] + E[||G_BA(x) - x||_1]
```

Encourages generators to preserve colour/structure when input is already in target domain.

### Total Generator Loss

```
L_G = L_GAN_AB + L_GAN_BA
    + 10*L_cyc_A + 10*L_cyc_B
    + 5*L_id_A   + 5*L_id_B
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Organise your dataset

```bash
# Auto-detects photos/ and sketches/ sub-folders
python setup_dataset.py --src "path/to/downloaded/dataset"

# Output:
#   dataset/trainA/   <- faces   (90%)
#   dataset/trainB/   <- sketches (90%)
#   dataset/testA/    <- faces   (10%)
#   dataset/testB/    <- sketches (10%)
```

### 3. Train

```bash
# Windows  (double-click or run in terminal)
train_local.bat

# Linux / macOS
chmod +x train_local.sh && ./train_local.sh

# Or directly
python local_train.py
python local_train.py --resume        # continue from last checkpoint
python local_train.py --fast          # 128px quick test mode
```

### 4. Run the web interface

```bash
# Windows
set CHECKPOINT_PATH=checkpoints\latest.pth && python app.py

# Linux / macOS
CHECKPOINT_PATH=checkpoints/latest.pth python app.py
```

Open `http://localhost:5000`

---

## Training on RTX 4060

The training script is specifically optimised for the RTX 4060 8 GB:

| Optimisation | Effect |
|---|---|
| `allow_tf32 = True` (matmul + cudnn) | ~1.5x faster on Ada Lovelace |
| `cudnn.benchmark = True` | Auto-tunes convolution algorithms |
| AMP `autocast` + `GradScaler` x3 | ~40% less VRAM, ~1.4x faster |

### VRAM budget at 256x256

| batch_size | Estimated VRAM | Status |
|---|---|---|
| 1 | ~2.8 GB | Very safe |
| **2** | **~5.2 GB** | **Default (recommended)** |
| 4 | ~9.8 GB | OOM on 8 GB — avoid |

### Expected training time

With batch=2 on RTX 4060: roughly **3-5 minutes per epoch**.
Full 200-epoch run: **10-16 hours** (split across sessions with `--resume`).

---

## Inference & CLI

### Single image

```bash
python predict.py --input face.jpg

# Force direction
python predict.py --input sketch.png --direction sketch2face

# Side-by-side comparison
python predict.py --input face.jpg --compare
```

### Batch folder

```bash
python predict.py --input ./testA --output ./results --direction face2sketch
```

### Auto-detection logic

```
is_sketch = (mean_HSV_saturation < 0.12) AND (grey_channel_std > 0.20)
```

Sketches have low colour saturation and high greyscale contrast.
No secondary model is needed for detection.

---

## Web Interface

The Flask UI at `http://localhost:5000` provides:

**Upload Tab** — drag-and-drop any image, choose direction (auto/force),
translate instantly, download the result.

**Camera Tab** — live WebRTC webcam feed, click "Capture & Translate"
to snapshot and translate a frame in real time.

---

## API Reference

### POST /translate

```bash
curl -X POST http://localhost:5000/translate \
     -F "file=@face.jpg" \
     -F "direction=auto" \
     --output translated.png
# Response header:  X-Translation-Direction: face -> sketch
```

### POST /translate_base64

```bash
curl -X POST http://localhost:5000/translate_base64 \
     -H "Content-Type: application/json" \
     -d '{"image":"data:image/jpeg;base64,...","direction":"auto"}'
# Response: {"image":"data:image/png;base64,...",
#            "direction":"face -> sketch",
#            "detected_domain":"face"}
```

### GET /health

```json
{ "status": "ok", "models_ready": true, "checkpoint": "checkpoints/latest.pth" }
```

---

## Hyperparameter Guide

| Parameter | Default | When to change |
|-----------|---------|----------------|
| `--lambda_cyc` | 10 | Raise if mode collapse; lower if over-constrained |
| `--lambda_id` | 5 | Set to 0 if colours look washed out |
| `--lr` | 2e-4 | Lower to 1e-4 for more stable training |
| `--batch_size` | 2 | Keep at 2 for RTX 4060 8 GB |
| `--epochs` | 200 | 100 for quick results; 200 for full quality |
| `--n_res` | 9 | Use 6 for 128px or faster iteration |
| `--ngf / --ndf` | 64 | Lower to 32 if OOM; raise to 128 for quality |
| `--buffer_size` | 50 | Set to 0 to disable history pool |

---

## Architecture Diagrams

Six Mermaid `.mmd` files in `architecture/`. Render with:

```bash
# Install CLI
npm install -g @mermaid-js/mermaid-cli

# Render all to PNG
for f in architecture/*.mmd; do
    mmdc -i "$f" -o "${f%.mmd}.png" -w 1600
done
```

Or paste any `.mmd` file into [mermaid.live](https://mermaid.live) to view instantly.

| File | Contents |
|------|----------|
| `01_system_overview.mmd` | Full system: data to models to losses to Flask |
| `02_training_loop.mmd` | Step-by-step training with all loss computations |
| `03_generator_arch.mmd` | ResNet-9 layer-by-layer with tensor shapes |
| `04_discriminator_arch.mmd` | PatchGAN-70 with receptive field analysis |
| `05_inference_pipeline.mmd` | Flask routes to domain detection to response |
| `06_file_dependency.mmd` | How all project files connect |

---

## References

- Zhu et al. (2017) — [CycleGAN paper](https://arxiv.org/abs/1703.10593)
- [Official PyTorch implementation](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- [Person Face Sketches dataset](https://www.kaggle.com/datasets/almightyj/person-face-sketches)
