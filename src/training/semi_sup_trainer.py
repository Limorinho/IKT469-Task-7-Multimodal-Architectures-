"""
Semi-supervised and self-supervised trainers.

SemiSupervisedTrainer: pseudo-labeling over multiple rounds.
ConsistencyTrainer: supervised + consistency + contrastive losses.
"""

import time
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

from src.data.dataset import MMIMDBUnlabeledDataset, get_image_transform
from src.models.semi_supervised import ConsistencyModel, PseudoLabelWrapper
from src.training.trainer import Trainer
from src.utils.metrics import compute_metrics


class SemiSupervisedTrainer(Trainer):
    """
    Pseudo-labeling trainer.

    Workflow:
      1. Train on labeled data for `warmup_epochs`.
      2. Generate pseudo-labels on unlabeled data.
      3. Combine confident pseudo-labeled samples with labeled data.
      4. Retrain for `finetune_epochs`.
    """

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 device: torch.device,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 confidence_threshold: float = 0.8):
        super().__init__(model, optimizer, device, scheduler)
        self.wrapper = PseudoLabelWrapper(model, confidence_threshold)

    @torch.no_grad()
    def build_pseudo_dataset(self, unlabeled_loader: DataLoader):
        """
        Run inference on unlabeled_loader, collect high-confidence predictions.
        Returns a TensorDataset of (images, pseudo_labels) for confident samples.
        """
        self.model.eval()
        confident_images, confident_labels, confident_texts = [], [], []

        for images, texts in unlabeled_loader:
            images = images.to(self.device)
            pseudo_labels, mask = self.wrapper.generate_pseudo_labels(images, texts)
            if mask.any():
                confident_images.append(images[mask].cpu())
                confident_labels.append(pseudo_labels[mask].cpu())
                confident_texts.extend([texts[i] for i in range(len(texts)) if mask[i]])

        if not confident_images:
            return None, 0

        imgs = torch.cat(confident_images)
        lbls = torch.cat(confident_labels)
        print(f"  Pseudo-labels: {len(imgs)} confident samples "
              f"({len(imgs) / (len(imgs) + 1) * 100:.1f}% of unlabeled)")
        return (imgs, lbls, confident_texts), len(imgs)

    def fit_semi(
        self,
        train_loader: DataLoader,
        unlabeled_loader: DataLoader,
        val_loader: DataLoader,
        warmup_epochs: int = 5,
        finetune_epochs: int = 5,
        pseudo_rounds: int = 2,
        log_every: int = 1,
    ) -> list[dict]:
        print(f"=== Warmup: {warmup_epochs} epochs on labeled data ===")
        history = self.fit(train_loader, val_loader, warmup_epochs, log_every)

        for round_idx in range(1, pseudo_rounds + 1):
            print(f"\n=== Pseudo-label round {round_idx}/{pseudo_rounds} ===")
            pseudo_data, n_pseudo = self.build_pseudo_dataset(unlabeled_loader)

            if pseudo_data is None or n_pseudo == 0:
                print("  No confident pseudo-labels found, skipping round.")
                continue

            imgs, lbls, texts = pseudo_data
            # Build a combined dataset loader (images + pseudo-labels as tensors)
            pseudo_ds = _PseudoDataset(imgs, lbls, texts)
            combined_ds = ConcatDataset([train_loader.dataset, pseudo_ds])
            combined_loader = DataLoader(
                combined_ds,
                batch_size=train_loader.batch_size,
                shuffle=True,
                num_workers=train_loader.num_workers,
                collate_fn=_collate_fn,
            )

            print(f"  Finetuning for {finetune_epochs} epochs on "
                  f"{len(combined_ds)} samples...")
            round_history = self.fit(combined_loader, val_loader, finetune_epochs, log_every)
            history.extend(round_history)

        return history


class ConsistencyTrainer:
    """
    Self-supervised consistency trainer.

    Loss = supervised_loss + alpha * consistency_loss + beta * contrastive_loss
    """

    def __init__(
        self,
        model: ConsistencyModel,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        alpha: float = 1.0,
        beta: float = 0.5,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.scheduler = scheduler
        self.criterion = nn.BCEWithLogitsLoss()
        self.aug_transform = get_image_transform(augment=True)

    def _augment_batch(self, images: torch.Tensor) -> torch.Tensor:
        """Apply per-image random augmentations to create a second view."""
        # images is already a tensor batch; re-apply augmentation per image
        # We use ColorJitter + RandomHorizontalFlip via torchvision functional
        import torchvision.transforms.functional as TF
        import random
        augmented = []
        for img in images.cpu():
            # Convert to PIL, augment, back to tensor
            pil = TF.to_pil_image(img)
            if random.random() > 0.5:
                pil = TF.hflip(pil)
            pil = TF.to_tensor(pil)
            augmented.append(pil)
        return torch.stack(augmented).to(self.device)

    def train_epoch(self, labeled_loader: DataLoader,
                    unlabeled_loader: Optional[DataLoader] = None) -> dict:
        self.model.train()
        total_loss = 0.0
        all_logits, all_labels = [], []

        unlabeled_iter = iter(unlabeled_loader) if unlabeled_loader else None

        for images, texts, labels in labeled_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Supervised loss
            logits = self.model(images, texts)
            sup_loss = self.criterion(logits, labels)

            # Consistency + contrastive on labeled batch (or unlabeled if available)
            if unlabeled_iter is not None:
                try:
                    u_images, u_texts = next(unlabeled_iter)
                    u_images = u_images.to(self.device)
                except StopIteration:
                    u_images, u_texts = images, texts
            else:
                u_images, u_texts = images, texts

            images_aug2 = self._augment_batch(u_images)
            cons_loss = self.model.consistency_loss(u_images, u_texts, images_aug2)
            cont_loss = self.model.contrastive_loss(u_images, u_texts, images_aug2)

            loss = sup_loss + self.alpha * cons_loss + self.beta * cont_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item() * len(labels)
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.cpu())

        if self.scheduler is not None:
            self.scheduler.step()

        logits_cat = torch.cat(all_logits)
        labels_cat = torch.cat(all_labels)
        metrics = compute_metrics(logits_cat, labels_cat)
        metrics["loss"] = total_loss / len(labeled_loader.dataset)
        return metrics

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict:
        self.model.eval()
        total_loss = 0.0
        all_logits, all_labels = [], []
        criterion = nn.BCEWithLogitsLoss()

        for images, texts, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            logits = self.model(images, texts)
            total_loss += criterion(logits, labels).item() * len(labels)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

        logits_cat = torch.cat(all_logits)
        labels_cat = torch.cat(all_labels)
        metrics = compute_metrics(logits_cat, labels_cat)
        metrics["loss"] = total_loss / len(loader.dataset)
        return metrics

    def fit(self, labeled_loader: DataLoader, val_loader: DataLoader,
            epochs: int, unlabeled_loader: Optional[DataLoader] = None,
            log_every: int = 1) -> list[dict]:
        history = []
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_m = self.train_epoch(labeled_loader, unlabeled_loader)
            val_m = self.evaluate(val_loader)
            elapsed = time.time() - t0
            history.append({"epoch": epoch, "train": train_m, "val": val_m})
            if epoch % log_every == 0:
                print(
                    f"Epoch {epoch:3d}/{epochs} ({elapsed:.1f}s) | "
                    f"train loss={train_m['loss']:.4f} f1={train_m['f1_micro']:.3f} | "
                    f"val loss={val_m['loss']:.4f} f1={val_m['f1_micro']:.3f} "
                    f"mAP={val_m['mAP']:.3f}"
                )
        return history


class _PseudoDataset(torch.utils.data.Dataset):
    """Wraps tensor images/labels + text list as a dataset."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor, texts: list[str]):
        self.images = images
        self.labels = labels
        self.texts = texts

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.texts[idx], self.labels[idx]


def _collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    texts = [b[1] for b in batch]
    labels = torch.stack([b[2] for b in batch])
    return images, texts, labels
