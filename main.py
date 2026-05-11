"""MM-IMDB multimodal experiments.

Run a single experiment at a time, e.g.:
    python main.py inspect-data
    python main.py text-only --epochs 20
    python main.py fusion-gmu --epochs 30 --batch-size 128
    python main.py semi-pseudo --label-fraction 0.2 --epochs 20

Each experiment writes results/<name>.json with the full metric history.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

HDF5_PATH = Path("dataset/multimodal_imdb.hdf5")
RESULTS_DIR = Path("results")


def _save(name: str, payload: dict) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{name}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nResults written to {out}")


def _device(arg: str) -> str:
    if arg != "auto":
        return arg
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_inspect_data(args):
    import h5py
    from src.data.hdf5_loader import MMIMDBDataset, GENRES

    with h5py.File(HDF5_PATH, "r") as f:
        print("HDF5 datasets:")
        for k in f.keys():
            d = f[k]
            print(f"  {k}: shape={d.shape} dtype={d.dtype}")
    print("\nLabel distribution (train split):")
    ds = MMIMDBDataset(HDF5_PATH, split="train")
    counts = ds.labels.sum(dim=0).int().tolist()
    for g, c in sorted(zip(GENRES, counts), key=lambda kv: -kv[1]):
        print(f"  {g:12s} {c}")


def _run_supervised(args, fusion: nn.Module, name: str, hidden_dim: int = 128) -> dict:
    from src.combiner.classifier import MultimodalClassifier
    from src.data.hdf5_loader import get_loaders, NUM_GENRES
    from src.training.supervised.train import run
    from src.training.evaluate import evaluate_metrics

    device = _device(args.device)
    torch.manual_seed(args.seed)

    model = MultimodalClassifier(fusion=fusion, num_classes=NUM_GENRES, hidden_dim=hidden_dim).to(device)
    loaders = get_loaders(
        HDF5_PATH,
        batch_size=args.batch_size,
        label_fraction=args.label_fraction,
        seed=args.seed,
    )
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    history = run(model, loaders, optim, criterion, device=device, epochs=args.epochs)
    test = evaluate_metrics(model, loaders["test"], device=device, criterion=criterion)
    payload = {
        "experiment": name,
        "args": vars(args),
        "history": history,
        "test": test,
    }
    _save(name, payload)
    return payload


def cmd_text_only(args):
    from src.fusion.unimodal import TextOnly
    _run_supervised(args, TextOnly(hidden_dim=128), name="text-only")


def cmd_image_only(args):
    from src.fusion.unimodal import ImageOnly
    _run_supervised(args, ImageOnly(hidden_dim=128), name="image-only")


def cmd_fusion_early(args):
    from src.fusion.early import ConcatMLP
    _run_supervised(args, ConcatMLP(hidden_dim=128), name="fusion-early")


def cmd_fusion_late(args):
    from src.fusion.late import LateFusion
    from src.data.hdf5_loader import NUM_GENRES
    _run_supervised(args, LateFusion(hidden_dim=128, num_classes=NUM_GENRES), name="fusion-late")


class _GmuWrapper(nn.Module):
    """Adapts Gmu (image, text) signature to the (text, image) interface."""

    def __init__(self, gmu):
        super().__init__()
        self.gmu = gmu

    def forward(self, text_emb, image_emb):
        return self.gmu(image_emb, text_emb)


def cmd_fusion_gmu(args):
    from src.fusion.gmu import Gmu

    fusion = _GmuWrapper(Gmu(text_dim=300, img_dim=4096, output_dim=128))
    _run_supervised(args, fusion, name="fusion-gmu")


def _run_semi(args, mode: str, name: str) -> dict:
    from src.combiner.classifier import MultimodalClassifier
    from src.data.hdf5_loader import MMIMDBDataset, NUM_GENRES
    from src.fusion.gmu import Gmu
    from src.training.semi_supervised.train import run
    from src.training.evaluate import evaluate_metrics
    from torch.utils.data import DataLoader
    import numpy as np

    device = _device(args.device)
    torch.manual_seed(args.seed)

    fusion = _GmuWrapper(Gmu(text_dim=300, img_dim=4096, output_dim=128))
    model = MultimodalClassifier(fusion=fusion, num_classes=NUM_GENRES, hidden_dim=128).to(device)

    full_train = MMIMDBDataset(HDF5_PATH, split="train", label_fraction=1.0)
    n = len(full_train)
    rng = np.random.default_rng(args.seed)
    labeled_indices = np.sort(rng.choice(n, size=int(n * args.label_fraction), replace=False))
    unlabeled_indices = np.setdiff1d(np.arange(n), labeled_indices, assume_unique=True)

    labeled_loaders = {
        "train": DataLoader(
            MMIMDBDataset(HDF5_PATH, split="train", indices=labeled_indices),
            batch_size=args.batch_size, shuffle=True,
        ),
        "dev": DataLoader(MMIMDBDataset(HDF5_PATH, split="dev"), batch_size=args.batch_size),
        "test": DataLoader(MMIMDBDataset(HDF5_PATH, split="test"), batch_size=args.batch_size),
    }
    unlabeled_loader = DataLoader(
        MMIMDBDataset(HDF5_PATH, split="train", indices=unlabeled_indices),
        batch_size=args.batch_size, shuffle=True,
    )

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    history = run(
        model, labeled_loaders, unlabeled_loader, optim, criterion, device=device,
        epochs=args.epochs, mode=mode,
        pseudo_threshold=args.pseudo_threshold,
        consistency_weight=args.consistency_weight,
        ema_alpha=args.ema_alpha,
    )
    test = evaluate_metrics(model, labeled_loaders["test"], device=device, criterion=criterion)
    payload = {"experiment": name, "args": vars(args), "history": history, "test": test}
    _save(name, payload)
    return payload


def cmd_semi_baseline(args):
    _run_semi(args, mode="baseline", name=f"semi-baseline-{int(args.label_fraction*100)}")


def cmd_semi_pseudo(args):
    _run_semi(args, mode="pseudo", name=f"semi-pseudo-{int(args.label_fraction*100)}")


def cmd_semi_meanteacher(args):
    _run_semi(args, mode="mean_teacher", name=f"semi-meanteacher-{int(args.label_fraction*100)}")


def cmd_selfsup_contrastive(args):
    from src.data.hdf5_loader import get_loaders, NUM_GENRES, TEXT_DIM, IMAGE_DIM
    from src.ssl.contrastive import info_nce, ProjectionHead
    from src.training.evaluate import compute_f1_scores
    import numpy as np

    device = _device(args.device)
    torch.manual_seed(args.seed)

    text_proj = ProjectionHead(TEXT_DIM, hidden_dim=256, output_dim=128).to(device)
    image_proj = ProjectionHead(IMAGE_DIM, hidden_dim=256, output_dim=128).to(device)

    loaders = get_loaders(HDF5_PATH, batch_size=args.batch_size, label_fraction=1.0)
    optim = torch.optim.Adam(
        list(text_proj.parameters()) + list(image_proj.parameters()),
        lr=args.lr, weight_decay=1e-5,
    )

    contrastive_loss_history: list[float] = []
    for epoch in range(args.epochs):
        text_proj.train()
        image_proj.train()
        epoch_loss = 0.0
        n = 0
        for batch in loaders["train"]:
            t = text_proj(batch["text_emb"].to(device))
            i = image_proj(batch["image_emb"].to(device))
            loss = info_nce(t, i, temperature=args.temperature)
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += float(loss.item())
            n += 1
        epoch_loss /= max(n, 1)
        contrastive_loss_history.append(epoch_loss)
        print(f"[contrastive] epoch {epoch+1:3d}/{args.epochs} | nce-loss {epoch_loss:.4f}")

    # Linear probe on the concatenated projections, using args.label_fraction of train labels.
    text_proj.eval()
    image_proj.eval()

    def embed(loader):
        zs, ys = [], []
        with torch.no_grad():
            for batch in loader:
                t = text_proj(batch["text_emb"].to(device))
                i = image_proj(batch["image_emb"].to(device))
                z = torch.cat([t, i], dim=1).cpu().numpy()
                zs.append(z)
                ys.append(batch["label"].numpy())
        return np.concatenate(zs), np.concatenate(ys)

    z_train, y_train = embed(loaders["train"])
    _, _ = embed(loaders["dev"])  # currently unused; available for future early stopping
    z_test, y_test = embed(loaders["test"])

    if args.label_fraction < 1.0:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(z_train), size=int(len(z_train) * args.label_fraction), replace=False)
        z_train, y_train = z_train[idx], y_train[idx]

    probe = nn.Linear(256, NUM_GENRES).to(device)
    probe_optim = torch.optim.Adam(probe.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    zt = torch.tensor(z_train).to(device)
    yt = torch.tensor(y_train).to(device)
    for _ in range(50):
        probe.train()
        probe_optim.zero_grad()
        loss = criterion(probe(zt), yt)
        loss.backward()
        probe_optim.step()
    probe.eval()
    with torch.no_grad():
        preds = (torch.sigmoid(probe(torch.tensor(z_test).to(device))) > 0.5).int().cpu().numpy()
    test_f1 = compute_f1_scores(y_test.astype(int), preds)

    payload = {
        "experiment": "selfsup-contrastive",
        "args": vars(args),
        "history": {"contrastive_loss": contrastive_loss_history},
        "test": {"f1": test_f1},
    }
    _save(f"selfsup-contrastive-{int(args.label_fraction*100)}", payload)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp):
        sp.add_argument("--epochs", type=int, default=20)
        sp.add_argument("--batch-size", type=int, default=128)
        sp.add_argument("--lr", type=float, default=1e-3)
        sp.add_argument("--device", default="auto")
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--label-fraction", type=float, default=1.0)

    sub.add_parser("inspect-data").set_defaults(func=cmd_inspect_data)

    for name, fn in [
        ("text-only", cmd_text_only),
        ("image-only", cmd_image_only),
        ("fusion-early", cmd_fusion_early),
        ("fusion-late", cmd_fusion_late),
        ("fusion-gmu", cmd_fusion_gmu),
    ]:
        sp = sub.add_parser(name)
        add_common(sp)
        sp.set_defaults(func=fn)

    for name, fn in [
        ("semi-baseline", cmd_semi_baseline),
        ("semi-pseudo", cmd_semi_pseudo),
        ("semi-meanteacher", cmd_semi_meanteacher),
    ]:
        sp = sub.add_parser(name)
        add_common(sp)
        sp.set_defaults(label_fraction=0.2)
        sp.add_argument("--pseudo-threshold", type=float, default=0.9)
        sp.add_argument("--consistency-weight", type=float, default=1.0)
        sp.add_argument("--ema-alpha", type=float, default=0.999)
        sp.set_defaults(func=fn)

    sp = sub.add_parser("selfsup-contrastive")
    add_common(sp)
    sp.set_defaults(label_fraction=0.2)
    sp.add_argument("--temperature", type=float, default=0.07)
    sp.set_defaults(func=cmd_selfsup_contrastive)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
