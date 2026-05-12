import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoTokenizer

GENRES = [
    "Action", "Adventure", "Animation", "Biography", "Comedy", "Crime",
    "Documentary", "Drama", "Family", "Fantasy", "Film-Noir", "History",
    "Horror", "Music", "Musical", "Mystery", "News", "Romance", "Sci-Fi",
    "Short", "Sport", "Thriller", "War", "Western",
]
GENRE_TO_IDX = {g: i for i, g in enumerate(GENRES)}
NUM_GENRES = len(GENRES)

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class MMIMDBDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: Path,
        tokenizer,
        max_length: int = 512,
        indices: np.ndarray | None = None,
    ):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.subset = np.asarray(indices, dtype=np.int64) if indices is not None else None

    def __len__(self) -> int:
        return len(self.subset) if self.subset is not None else len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        i = int(self.subset[idx]) if self.subset is not None else idx
        row = self.df.iloc[i]

        image = Image.open(self.images_dir / row["image_path"]).convert("RGB")
        image = IMAGE_TRANSFORM(image)

        encoding = self.tokenizer(
            row["plot outline"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        label = torch.zeros(len(GENRES))
        for genre in row["genre"].split(" - "):
            genre = genre.strip()
            if genre in GENRE_TO_IDX:
                label[GENRE_TO_IDX[genre]] = 1.0

        return {
            "image": image,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": label,
        }


def get_loaders(
    dataset_root: Path,
    tokenizer_name: str = "bert-base-uncased",
    batch_size: int = 32,
    num_workers: int = 0,
    label_fraction: float = 1.0,
    seed: int = 0,
) -> dict[str, DataLoader]:
    csv_path = dataset_root / "tinymmimdb" / "data.csv"
    images_dir = dataset_root / "tinymmimdb" / "images"

    df = pd.read_csv(csv_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    if label_fraction < 1.0:
        rng = np.random.default_rng(seed)
        n = len(train_df)
        idx = np.sort(rng.choice(n, size=int(round(n * label_fraction)), replace=False))
        train_dataset = MMIMDBDataset(train_df, images_dir, tokenizer, indices=idx)
    else:
        train_dataset = MMIMDBDataset(train_df, images_dir, tokenizer)

    loaders: dict[str, DataLoader] = {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "dev": DataLoader(
            MMIMDBDataset(df[df["split"] == "dev"].reset_index(drop=True), images_dir, tokenizer),
            batch_size=batch_size, num_workers=num_workers,
        ),
        "test": DataLoader(
            MMIMDBDataset(df[df["split"] == "test"].reset_index(drop=True), images_dir, tokenizer),
            batch_size=batch_size, num_workers=num_workers,
        ),
    }
    return loaders
