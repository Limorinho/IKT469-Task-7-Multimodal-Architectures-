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

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class MMIMDBDataset(Dataset):
    def __init__(self, df: pd.DataFrame, images_dir: Path, tokenizer, max_length: int = 512):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

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
    num_workers: int = 4,
):
    csv_path = dataset_root / "tinymmimdb" / "data.csv"
    images_dir = dataset_root / "tinymmimdb" / "images"

    df = pd.read_csv(csv_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    splits = {
        "train": df[df["split"] == "train"],
        "val": df[df["split"] == "dev"],
        "test": df[df["split"] == "test"],
    }

    loaders = {}
    for split, split_df in splits.items():
        dataset = MMIMDBDataset(split_df, images_dir, tokenizer)
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
        )

    return loaders
