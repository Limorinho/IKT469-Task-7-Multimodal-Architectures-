import kagglehub
import shutil
from pathlib import Path


def _dataset_dir() -> Path:
    base = Path(__file__).parent.parent.parent if "__file__" in dir() else Path.cwd()
    return base / "dataset" / "tiny-mm-imdb"


def download():
    dataset_dir = _dataset_dir()
    if dataset_dir.exists():
        print(f"Dataset already exists at {dataset_dir}")
        return

    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    path = kagglehub.dataset_download("gabrieltardochi/tiny-mm-imdb")
    shutil.move(path, str(dataset_dir))

    print(f"Dataset downloaded and moved to {dataset_dir}")


if __name__ == "__main__":
    download()
