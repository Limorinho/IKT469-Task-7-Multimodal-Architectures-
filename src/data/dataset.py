"""MM-IMDB dataset loader with multi-label genre classification."""

import os
from typing import Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

GENRES = [
    "Action", "Adventure", "Animation", "Biography", "Comedy", "Crime",
    "Documentary", "Drama", "Family", "Fantasy", "Film-Noir", "History",
    "Horror", "Music", "Musical", "Mystery", "News", "Romance", "Sci-Fi",
    "Short", "Sport", "Thriller", "War", "Western",
]
NUM_GENRES = len(GENRES)
GENRE_TO_IDX = {g: i for i, g in enumerate(GENRES)}

IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]


def get_image_transform(augment: bool = False) -> transforms.Compose:
    if augment:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
    ])


def genre_to_label(genre_str: str) -> torch.Tensor:
    label = torch.zeros(NUM_GENRES)
    for part in genre_str.split(" - "):
        idx = GENRE_TO_IDX.get(part.strip())
        if idx is not None:
            label[idx] = 1.0
    return label


class MMIMDBDataset(Dataset):
    """MM-IMDB dataset: returns (image_tensor, text_str, label_tensor) per sample."""

    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        split: str,
        transform: Optional[transforms.Compose] = None,
        labeled: bool = True,
    ):
        df = pd.read_csv(csv_path)
        self.data = df[df["split"] == split].reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform or get_image_transform(augment=False)
        self.labeled = labeled

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_dir, row["image_path"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        text = f"{row['title']}. {row['plot outline']}"

        if self.labeled:
            label = genre_to_label(row["genre"])
            return image, text, label
        return image, text


class MMIMDBUnlabeledDataset(Dataset):
    """Wraps MMIMDBDataset to return samples without labels (for semi-supervised)."""

    def __init__(self, csv_path: str, image_dir: str, split: str = "dev",
                 transform: Optional[transforms.Compose] = None):
        df = pd.read_csv(csv_path)
        self.data = df[df["split"] == split].reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform or get_image_transform(augment=False)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_dir, row["image_path"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        text = f"{row['title']}. {row['plot outline']}"
        return image, text
