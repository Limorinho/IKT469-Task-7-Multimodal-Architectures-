import pytest
import torch
from pathlib import Path
from src.data.hdf5_loader import MMIMDBDataset, get_loaders, NUM_GENRES, TEXT_DIM, IMAGE_DIM

HDF5_PATH = Path("dataset/multimodal_imdb.hdf5")


@pytest.mark.skipif(not HDF5_PATH.exists(), reason="HDF5 not available")
def test_dataset_shapes():
    ds = MMIMDBDataset(HDF5_PATH, split="dev")
    item = ds[0]
    assert item["text_emb"].shape == (TEXT_DIM,)
    assert item["image_emb"].shape == (IMAGE_DIM,)
    assert item["label"].shape == (NUM_GENRES,)
    assert item["text_emb"].dtype == torch.float32
    assert item["image_emb"].dtype == torch.float32
    assert item["label"].dtype == torch.float32


@pytest.mark.skipif(not HDF5_PATH.exists(), reason="HDF5 not available")
def test_split_sizes():
    train = MMIMDBDataset(HDF5_PATH, split="train")
    dev = MMIMDBDataset(HDF5_PATH, split="dev")
    test = MMIMDBDataset(HDF5_PATH, split="test")
    assert len(train) == 15552
    assert len(dev) == 18160 - 15552
    assert len(test) == 25959 - 18160


@pytest.mark.skipif(not HDF5_PATH.exists(), reason="HDF5 not available")
def test_label_fraction_subsample():
    ds = MMIMDBDataset(HDF5_PATH, split="train", label_fraction=0.2, seed=0)
    assert len(ds) == pytest.approx(15552 * 0.2, abs=1)


@pytest.mark.skipif(not HDF5_PATH.exists(), reason="HDF5 not available")
def test_get_loaders_returns_three_splits():
    loaders = get_loaders(HDF5_PATH, batch_size=4)
    assert set(loaders.keys()) == {"train", "dev", "test"}
    batch = next(iter(loaders["train"]))
    assert batch["text_emb"].shape == (4, TEXT_DIM)
    assert batch["image_emb"].shape == (4, IMAGE_DIM)
    assert batch["label"].shape == (4, NUM_GENRES)
