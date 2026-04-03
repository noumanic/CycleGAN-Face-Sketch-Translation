"""
Inference Utilities
====================
Provides:
  • load_generators()    – load G_AB and G_BA from a checkpoint
  • detect_domain()      – auto-detect whether an image is a face or sketch
  • translate_image()    – run the appropriate generator on a PIL image
"""

import io
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T

from models import Generator


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

_G_AB = None   # face → sketch  (module-level singletons for Flask reuse)
_G_BA = None
_device = None


def load_generators(checkpoint_path: str, device: str = None, ngf: int = 64, n_res: int = 9):
    """
    Load both generators from a CycleGAN checkpoint.

    Args:
        checkpoint_path : path to .pth checkpoint saved by train.py
        device          : 'cuda', 'cpu', or None (auto-detect)
        ngf             : generator filter count (must match training config)
        n_res           : number of residual blocks (must match training config)
    """
    global _G_AB, _G_BA, _device

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    _device = torch.device(device)

    ckpt = torch.load(checkpoint_path, map_location=_device)

    _G_AB = Generator(ngf=ngf, n_res_blocks=n_res).to(_device)
    _G_BA = Generator(ngf=ngf, n_res_blocks=n_res).to(_device)

    _G_AB.load_state_dict(ckpt["G_AB"])
    _G_BA.load_state_dict(ckpt["G_BA"])

    _G_AB.eval()
    _G_BA.eval()

    print(f"[Inference] Generators loaded from: {checkpoint_path}")
    print(f"[Inference] Device: {_device}  |  Epoch: {ckpt.get('epoch', '?')}")
    return _G_AB, _G_BA


# ---------------------------------------------------------------------------
# Domain Detection
# ---------------------------------------------------------------------------

def detect_domain(image: Image.Image) -> str:
    """
    Heuristically decide whether `image` belongs to domain A (real face)
    or domain B (sketch).

    Strategy
    --------
    Sketches are near-greyscale with high contrast and low colour saturation.
    We check two signals:
      1. Colour saturation: sketches have very low saturation.
      2. Edge density: sketches have dense, sharp edges.

    Returns: "face" or "sketch"
    """
    rgb = image.convert("RGB")
    arr = np.array(rgb, dtype=np.float32) / 255.0

    # Signal 1 – saturation (HSV)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    cmax = arr.max(axis=-1)
    cmin = arr.min(axis=-1)
    diff = cmax - cmin
    # Saturation = diff / cmax  (avoid div-by-zero)
    sat = np.where(cmax > 0, diff / (cmax + 1e-8), 0.0)
    mean_sat = sat.mean()

    # Signal 2 – grey channel variance
    grey = 0.299 * r + 0.587 * g + 0.114 * b
    grey_std = grey.std()

    # Thresholds (tuned empirically; adjust if needed)
    # Sketches: mean_sat < 0.10  AND  grey_std > 0.25
    is_sketch = (mean_sat < 0.12) and (grey_std > 0.20)
    return "sketch" if is_sketch else "face"


# ---------------------------------------------------------------------------
# Transform  (same as test-time transform in dataset.py)
# ---------------------------------------------------------------------------

_transform = T.Compose([
    T.Resize(256, Image.BICUBIC),
    T.CenterCrop(256),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    t = (t.squeeze(0) * 0.5 + 0.5).clamp(0, 1)
    arr = (t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

def translate_image(image: Image.Image,
                    direction: str = "auto") -> tuple[Image.Image, str]:
    """
    Translate a PIL image using the loaded generators.

    Args:
        image     : input PIL image (any size)
        direction : "auto"     – detect domain automatically
                    "face2sketch" – force face→sketch translation
                    "sketch2face" – force sketch→face translation

    Returns:
        (output_image, detected_direction)
    """
    if _G_AB is None or _G_BA is None:
        raise RuntimeError(
            "Generators not loaded. Call load_generators() first."
        )

    image = image.convert("RGB")

    if direction == "auto":
        domain = detect_domain(image)
        direction = "face2sketch" if domain == "face" else "sketch2face"

    tensor = _transform(image).unsqueeze(0).to(_device)

    with torch.no_grad():
        if direction == "face2sketch":
            output_tensor = _G_AB(tensor)   # G_AB: face → sketch
        else:
            output_tensor = _G_BA(tensor)   # G_BA: sketch → face

    output_image = _tensor_to_pil(output_tensor)
    return output_image, direction


# ---------------------------------------------------------------------------
# PIL ↔ bytes helpers  (used by Flask routes)
# ---------------------------------------------------------------------------

def pil_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


def bytes_to_pil(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")
