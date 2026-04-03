"""
Dataset & Utilities
====================
Handles loading of the Person Face Sketches dataset from Kaggle.

Expected directory layout after downloading:
    dataset/
        trainA/   ← real face photos
        trainB/   ← sketch images
        testA/
        testB/

The Kaggle dataset may use slightly different folder names; adjust
DOMAIN_A / DOMAIN_B at the bottom of this file if needed.
"""

import os
import random
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_transforms(image_size: int = 256, train: bool = True):
    """Return a transform pipeline.

    During training we apply random horizontal flip + jitter.
    At test time we only resize + centre-crop.
    """
    if train:
        return T.Compose([
            T.Resize(int(image_size * 1.12), Image.BICUBIC),
            T.RandomCrop(image_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    else:
        return T.Compose([
            T.Resize(image_size, Image.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def _list_images(folder: str):
    folder = Path(folder)
    paths = sorted([
        str(p) for p in folder.rglob("*")
        if p.suffix.lower() in IMG_EXTENSIONS
    ])
    return paths


class FaceSketchDataset(Dataset):
    """
    Unpaired dataset that randomly samples one image from domain A
    and one from domain B for each __getitem__ call, as required by
    CycleGAN's unpaired training regime.

    Args:
        root_a  : folder containing domain-A images (real faces)
        root_b  : folder containing domain-B images (sketches)
        transform: torchvision transform (same applied to both domains)
    """

    def __init__(self, root_a: str, root_b: str, transform=None):
        self.files_a = _list_images(root_a)
        self.files_b = _list_images(root_b)
        self.transform = transform

        if len(self.files_a) == 0:
            raise FileNotFoundError(f"No images found in {root_a}")
        if len(self.files_b) == 0:
            raise FileNotFoundError(f"No images found in {root_b}")

        print(f"[Dataset] Domain A (faces):   {len(self.files_a):,} images")
        print(f"[Dataset] Domain B (sketches): {len(self.files_b):,} images")

    def __len__(self):
        # Use the larger domain so we see all images
        return max(len(self.files_a), len(self.files_b))

    def __getitem__(self, idx):
        path_a = self.files_a[idx % len(self.files_a)]
        path_b = self.files_b[random.randint(0, len(self.files_b) - 1)]

        img_a = Image.open(path_a).convert("RGB")
        img_b = Image.open(path_b).convert("RGB")

        if self.transform:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)

        return {"A": img_a, "B": img_b}


# ---------------------------------------------------------------------------
# Image Buffer  (history pool to stabilise discriminator training)
# ---------------------------------------------------------------------------

class ImageBuffer:
    """
    Implements the image history buffer from Section 3 of the CycleGAN paper.
    Stores up to `max_size` previously generated images; when the buffer is
    full, with probability 0.5 a random stored image is returned instead of
    the freshly generated one (and the buffer is updated accordingly).
    """

    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, images: torch.Tensor) -> torch.Tensor:
        if self.max_size == 0:
            return images

        return_images = []
        for image in images.unbind(0):          # iterate over batch dim
            image = image.unsqueeze(0)
            if len(self.data) < self.max_size:
                self.data.append(image)
                return_images.append(image)
            else:
                if random.random() > 0.5:
                    idx = random.randint(0, self.max_size - 1)
                    stored = self.data[idx].clone()
                    self.data[idx] = image
                    return_images.append(stored)
                else:
                    return_images.append(image)

        return torch.cat(return_images, dim=0)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_dataloaders(root: str,
                      image_size: int = 256,
                      batch_size: int = 1,
                      num_workers: int = 4):
    """
    Build train and (optional) test DataLoaders.

    root layout:
        root/
            trainA/  trainB/
            testA/   testB/   (optional)
    """
    train_loader = test_loader = None

    train_a = os.path.join(root, "trainA")
    train_b = os.path.join(root, "trainB")
    if os.path.isdir(train_a) and os.path.isdir(train_b):
        train_ds = FaceSketchDataset(
            train_a, train_b,
            transform=get_transforms(image_size, train=True)
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True
        )

    test_a = os.path.join(root, "testA")
    test_b = os.path.join(root, "testB")
    if os.path.isdir(test_a) and os.path.isdir(test_b):
        test_ds = FaceSketchDataset(
            test_a, test_b,
            transform=get_transforms(image_size, train=False)
        )
        test_loader = DataLoader(
            test_ds, batch_size=1, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )

    return train_loader, test_loader