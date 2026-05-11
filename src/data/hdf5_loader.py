import h5py
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

TEXT_DIM = 300
IMAGE_DIM = 4096
NUM_GENRES = 23
GENRES = [
    "Action", "Adventure", "Animation", "Biography", "Comedy", "Crime",
    "Documentary", "Drama", "Family", "Fantasy", "Film-Noir", "History",
    "Horror", "Music", "Musical", "Mystery", "News", "Romance", "Sci-Fi",
    "Short", "Sport", "Thriller", "War", "Western",
]


def _read_splits(hdf5_path: Path) -> dict[str, tuple[int, int]]:
    with h5py.File(hdf5_path, "r") as f:
        raw = f.attrs["split"]
    out: dict[str, tuple[int, int]] = {}
    for row in raw:
        split = row[0].decode() if isinstance(row[0], bytes) else row[0]
        dataset_name = row[1].decode() if isinstance(row[1], bytes) else row[1]
        if dataset_name != "features":
            continue
        out[split] = (int(row[2]), int(row[3]))
    return out


class MMIMDBDataset(Dataset):
    def __init__(
        self,
        hdf5_path: Path,
        split: str,
        label_fraction: float = 1.0,
        seed: int = 0,
        indices: np.ndarray | None = None,
    ):
        self.hdf5_path = Path(hdf5_path)
        self.split = split

        splits = _read_splits(self.hdf5_path)
        start, end = splits[split]

        with h5py.File(self.hdf5_path, "r") as f:
            self.text = torch.from_numpy(f["features"][start:end].astype(np.float32))
            self.image = torch.from_numpy(f["vgg_features"][start:end].astype(np.float32))
            self.labels = torch.from_numpy(f["genres"][start:end].astype(np.float32))

        if indices is not None:
            self.subset = np.asarray(indices, dtype=np.int64)
        elif label_fraction < 1.0:
            rng = np.random.default_rng(seed)
            n = len(self.text)
            k = int(round(n * label_fraction))
            self.subset = rng.choice(n, size=k, replace=False)
            self.subset.sort()
        else:
            self.subset = None

    def __len__(self) -> int:
        return len(self.subset) if self.subset is not None else len(self.text)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        i = int(self.subset[idx]) if self.subset is not None else idx
        return {
            "text_emb": self.text[i],
            "image_emb": self.image[i],
            "label": self.labels[i],
        }


def get_loaders(
    hdf5_path: Path,
    batch_size: int = 64,
    num_workers: int = 0,
    label_fraction: float = 1.0,
    seed: int = 0,
) -> dict[str, DataLoader]:
    loaders: dict[str, DataLoader] = {}
    for split in ("train", "dev", "test"):
        frac = label_fraction if split == "train" else 1.0
        ds = MMIMDBDataset(hdf5_path, split=split, label_fraction=frac, seed=seed)
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=(split == "train"), num_workers=num_workers
        )
    return loaders
