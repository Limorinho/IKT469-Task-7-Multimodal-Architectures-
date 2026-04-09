"""
Multimodal MM-IMDB: early fusion, late fusion, semi-supervised, consistency training.

Usage:
  uv run main.py --mode early_fusion --epochs 10
  uv run main.py --mode late_fusion --epochs 10
  uv run main.py --mode semi_supervised --epochs 10
  uv run main.py --mode consistency --epochs 10
  uv run main.py --mode all --epochs 10
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader

from src.data.dataset import (
    MMIMDBDataset,
    MMIMDBUnlabeledDataset,
    get_image_transform,
)
from src.models.fusion import EarlyFusionModel, LateFusionModel
from src.models.semi_supervised import ConsistencyModel
from src.training.semi_sup_trainer import ConsistencyTrainer, SemiSupervisedTrainer
from src.training.trainer import Trainer

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_ROOT = os.path.join("dataset", "tiny-mm-imdb", "tinymmimdb")
CSV_PATH = os.path.join(DATASET_ROOT, "data.csv")
IMAGE_DIR = os.path.join(DATASET_ROOT, "images")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loaders(batch_size: int = 32, num_workers: int = 4):
    train_ds = MMIMDBDataset(CSV_PATH, IMAGE_DIR, "train",
                             transform=get_image_transform(augment=True))
    val_ds = MMIMDBDataset(CSV_PATH, IMAGE_DIR, "dev")
    test_ds = MMIMDBDataset(CSV_PATH, IMAGE_DIR, "test")
    unlabeled_ds = MMIMDBUnlabeledDataset(CSV_PATH, IMAGE_DIR, "dev")

    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, shuffle=False, **kw)
    unlabeled_loader = DataLoader(unlabeled_ds, shuffle=True, **kw)
    return train_loader, val_loader, test_loader, unlabeled_loader


def print_results(name: str, metrics: dict):
    print(f"\n{'='*50}")
    print(f"  {name} — Test Results")
    print(f"  F1 micro : {metrics['f1_micro']:.4f}")
    print(f"  F1 macro : {metrics['f1_macro']:.4f}")
    print(f"  mAP      : {metrics['mAP']:.4f}")
    print(f"{'='*50}")


def run_early_fusion(device, train_loader, val_loader, test_loader, epochs):
    print("\n>>> Early Fusion Model")
    model = EarlyFusionModel(freeze_backbones=True)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    trainer = Trainer(model, optimizer, device, scheduler)
    trainer.fit(train_loader, val_loader, epochs)
    results = trainer.evaluate(test_loader)
    print_results("Early Fusion", results)
    return model, results


def run_late_fusion(device, train_loader, val_loader, test_loader, epochs):
    print("\n>>> Late Fusion Model")
    model = LateFusionModel(freeze_backbones=True, learned_weights=True)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    trainer = Trainer(model, optimizer, device, scheduler)
    trainer.fit(train_loader, val_loader, epochs)
    results = trainer.evaluate(test_loader)
    print_results("Late Fusion", results)

    # Log per-modality fusion weights
    w = torch.softmax(model.fusion_weights.detach().cpu(), dim=0)
    print(f"  Learned fusion weights: image={w[0]:.3f}, text={w[1]:.3f}")
    return model, results


def run_semi_supervised(device, train_loader, val_loader, test_loader,
                        unlabeled_loader, epochs):
    print("\n>>> Semi-Supervised (Pseudo-Labeling)")
    model = EarlyFusionModel(freeze_backbones=True)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs + 2 * 5
    )
    trainer = SemiSupervisedTrainer(model, optimizer, device, scheduler,
                                    confidence_threshold=0.75)
    warmup = max(1, epochs // 2)
    finetune = max(1, epochs // 4)
    trainer.fit_semi(train_loader, unlabeled_loader, val_loader,
                     warmup_epochs=warmup, finetune_epochs=finetune, pseudo_rounds=2)
    results = trainer.evaluate(test_loader)
    print_results("Semi-Supervised (Pseudo-Labeling)", results)
    return model, results


def run_consistency(device, train_loader, val_loader, test_loader,
                    unlabeled_loader, epochs):
    print("\n>>> Self-Supervised Consistency Training")
    model = ConsistencyModel(freeze_backbones=True, consistency_weight=1.0)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    trainer = ConsistencyTrainer(model, optimizer, device,
                                 alpha=1.0, beta=0.5, scheduler=scheduler)
    trainer.fit(train_loader, val_loader, epochs, unlabeled_loader=unlabeled_loader)
    results = trainer.evaluate(test_loader)
    print_results("Consistency Training", results)
    return model, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="all",
                        choices=["early_fusion", "late_fusion",
                                 "semi_supervised", "consistency", "all"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, unlabeled_loader = make_loaders(
        args.batch_size, args.num_workers
    )

    results = {}

    if args.mode in ("early_fusion", "all"):
        _, r = run_early_fusion(device, train_loader, val_loader, test_loader, args.epochs)
        results["early_fusion"] = r

    if args.mode in ("late_fusion", "all"):
        _, r = run_late_fusion(device, train_loader, val_loader, test_loader, args.epochs)
        results["late_fusion"] = r

    if args.mode in ("semi_supervised", "all"):
        _, r = run_semi_supervised(device, train_loader, val_loader, test_loader,
                                   unlabeled_loader, args.epochs)
        results["semi_supervised"] = r

    if args.mode in ("consistency", "all"):
        _, r = run_consistency(device, train_loader, val_loader, test_loader,
                               unlabeled_loader, args.epochs)
        results["consistency"] = r

    if len(results) > 1:
        print("\n{'='*50}")
        print("  Comparison Summary")
        print(f"  {'Model':<30} {'F1 micro':>10} {'F1 macro':>10} {'mAP':>10}")
        print(f"  {'-'*60}")
        for name, m in results.items():
            print(f"  {name:<30} {m['f1_micro']:>10.4f} {m['f1_macro']:>10.4f} {m['mAP']:>10.4f}")


if __name__ == "__main__":
    main()
