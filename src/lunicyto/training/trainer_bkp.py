"""
Цикл обучения для гибридной модели HybridViTCNN.

Возможности:
- Автоопределение устройства (CUDA / Apple MPS / CPU)
- Автоматическое смешанное точность (AMP) для CUDA
- Label smoothing
- Mixup augmentation
- Cosine Annealing с линейным warmup
- Early stopping
- Сохранение best checkpoint
- TensorBoard логирование
"""
import logging
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from lunicyto.training.metrics import compute_metrics, plot_confusion_matrix, plot_training_curves

logger = logging.getLogger(__name__)


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Mixup augmentation: смешивает пары примеров."""
    if alpha > 0:
        lam = float(torch.distributions.Beta(alpha, alpha).sample())
    else:
        lam = 1.0
    batch_size = x.size(0)
    idx = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(
    criterion: nn.Module,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class WarmupCosineScheduler(torch.optim.lr_scheduler.LambdaLR):
    """Linear warmup → cosine decay."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
    ):
        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(max(1, warmup_epochs))
            progress = float(epoch - warmup_epochs) / float(
                max(1, total_epochs - warmup_epochs)
            )
            return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * progress)).item()))

        super().__init__(optimizer, lr_lambda)


class EarlyStopping:
    """Ранняя остановка по val_loss или val_accuracy."""

    def __init__(self, patience: int = 10, mode: str = "max", min_delta: float = 1e-4):
        assert mode in ("min", "max")
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best: Optional[float] = None

    def __call__(self, value: float) -> bool:
        """Возвращает True если нужно остановить обучение."""
        if self.best is None:
            self.best = value
            return False

        improved = (
            value > self.best + self.min_delta
            if self.mode == "max"
            else value < self.best - self.min_delta
        )
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience


class Trainer:
    """
    Полный цикл обучения и оценки HybridViTCNN на SIPaKMeD.

    Parameters
    ----------
    model: модель PyTorch
    train_loader, val_loader, test_loader: DataLoader'ы
    output_dir: папка для checkpoint'ов и логов
    learning_rate: начальный LR (для backbone и transformer раздельно)
    weight_decay: L2-регуляризация
    epochs: максимальное число эпох
    warmup_epochs: эпохи линейного разогрева LR
    label_smoothing: сглаживание меток (0.1 = 10%)
    mixup_alpha: параметр Mixup (0 = выключить)
    grad_clip: максимальная норма градиента
    early_stopping_patience: число эпох без улучшения до остановки
    class_names: имена классов (для отчётов)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        output_dir: str | Path = "runs/experiment",
        learning_rate: float = 2e-4,
        backbone_lr_scale: float = 0.1,
        weight_decay: float = 1e-2,
        epochs: int = 50,
        warmup_epochs: int = 5,
        label_smoothing: float = 0.1,
        mixup_alpha: float = 0.4,
        grad_clip: float = 1.0,
        early_stopping_patience: int = 10,
        class_names: Optional[list] = None,
    ):
        self.device = _get_device()
        logger.info(f"Устройство: {self.device}")

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.epochs = epochs
        self.mixup_alpha = mixup_alpha
        self.grad_clip = grad_clip
        self.class_names = class_names or [str(i) for i in range(5)]

        # Loss с label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # Раздельные LR: backbone обучается медленнее (fine-tuning)
        backbone_params = list(model.backbone.parameters())
        head_params = [
            p for n, p in model.named_parameters()
            if not n.startswith("backbone")
        ]
        self.optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": learning_rate * backbone_lr_scale},
                {"params": head_params, "lr": learning_rate},
            ],
            weight_decay=weight_decay,
        )

        self.scheduler = WarmupCosineScheduler(
            self.optimizer, warmup_epochs=warmup_epochs, total_epochs=epochs
        )

        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience, mode="max"
        )

        # AMP только для CUDA
        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.output_dir / "tb_logs"))

        # История метрик
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.train_accs: list[float] = []
        self.val_accs: list[float] = []
        self.best_val_acc = 0.0

    def train(self) -> dict:
        """Основной цикл обучения. Возвращает метрики на тестовой выборке."""
        logger.info(f"Начало обучения: {self.epochs} эпох, устройство: {self.device}")
        logger.info(
            f"Train: {len(self.train_loader.dataset)} | "
            f"Val: {len(self.val_loader.dataset)} | "
            f"Test: {len(self.test_loader.dataset)}"
        )

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()

            train_loss, train_acc = self._train_epoch(epoch)
            val_loss, val_acc, val_f1 = self._val_epoch(epoch)

            self.scheduler.step()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)

            elapsed = time.time() - t0
            lr = self.optimizer.param_groups[1]["lr"]
            logger.info(
                f"Epoch {epoch:3d}/{self.epochs} | "
                f"train loss={train_loss:.4f} acc={train_acc:.4f} | "
                f"val loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f} | "
                f"lr={lr:.2e} | {elapsed:.1f}s"
            )

            # TensorBoard
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/val", val_loss, epoch)
            self.writer.add_scalar("Accuracy/train", train_acc, epoch)
            self.writer.add_scalar("Accuracy/val", val_acc, epoch)
            self.writer.add_scalar("F1_macro/val", val_f1, epoch)
            self.writer.add_scalar("LR", lr, epoch)

            # Сохраняем лучший checkpoint
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self._save_checkpoint(epoch, val_acc)

            # Early stopping
            if self.early_stopping(val_acc):
                logger.info(
                    f"Early stopping на эпохе {epoch} (лучший val_acc={self.best_val_acc:.4f})"
                )
                break

        self.writer.close()

        # Финальная оценка на тесте
        test_metrics = self._evaluate(self.test_loader, phase="test")
        self._save_results(test_metrics)

        logger.info("\n=== Результаты на тестовой выборке ===")
        logger.info(f"Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"F1 macro: {test_metrics['f1_macro']:.4f}")
        logger.info(f"\n{test_metrics['report']}")

        return test_metrics

    def _train_epoch(self, epoch: int) -> tuple[float, float]:
        """Одна эпоха обучения с Mixup."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Train {epoch}",
            leave=False,
            dynamic_ncols=True,
        )
        for imgs, labels in pbar:
            imgs = imgs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # Mixup
            if self.mixup_alpha > 0:
                imgs, y_a, y_b, lam = mixup_data(imgs, labels, self.mixup_alpha)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(imgs)
                if self.mixup_alpha > 0:
                    loss = mixup_criterion(self.criterion, logits, y_a, y_b, lam)
                else:
                    loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_size = imgs.size(0)
            total_loss += loss.item() * batch_size
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / total, correct / total

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> tuple[float, float, float]:
        """Одна эпоха валидации."""
        self.model.eval()
        total_loss = 0.0
        all_preds: list[int] = []
        all_labels: list[int] = []

        for imgs, labels in self.val_loader:
            imgs = imgs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(imgs)
                loss = self.criterion(logits, labels)

            total_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        metrics = compute_metrics(all_labels, all_preds, self.class_names)
        n = len(self.val_loader.dataset)
        return total_loss / n, metrics["accuracy"], metrics["f1_macro"]

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader, phase: str = "test") -> dict:
        """Полная оценка на произвольном DataLoader."""
        self.model.eval()
        all_preds: list[int] = []
        all_labels: list[int] = []

        for imgs, labels in tqdm(loader, desc=f"Evaluate ({phase})", leave=False):
            imgs = imgs.to(self.device, non_blocking=True)
            logits = self.model(imgs)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        return compute_metrics(all_labels, all_preds, self.class_names)

    def _save_checkpoint(self, epoch: int, val_acc: float) -> None:
        """Сохраняет checkpoint лучшей модели."""
        ckpt_path = self.output_dir / "best_model.pth"
        torch.save(
            {
                "epoch": epoch,
                "val_acc": val_acc,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            ckpt_path,
        )
        logger.info(f"Checkpoint сохранён: {ckpt_path} (val_acc={val_acc:.4f})")

    def _save_results(self, test_metrics: dict) -> None:
        """Сохраняет confusion matrix и кривые обучения."""
        # Confusion matrix
        cm_fig = plot_confusion_matrix(
            test_metrics["confusion_matrix"],
            self.class_names,
            save_path=self.output_dir / "confusion_matrix.png",
            title="Confusion Matrix (Test)",
        )
        plt_close(cm_fig)

        # Кривые обучения
        if self.train_losses:
            curve_fig = plot_training_curves(
                self.train_losses,
                self.val_losses,
                self.train_accs,
                self.val_accs,
                save_path=self.output_dir / "training_curves.png",
            )
            plt_close(curve_fig)

        # Текстовый отчёт
        report_path = self.output_dir / "test_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"Accuracy: {test_metrics['accuracy']:.4f}\n")
            f.write(f"F1 macro: {test_metrics['f1_macro']:.4f}\n\n")
            f.write(test_metrics["report"])
        logger.info(f"Отчёт сохранён: {report_path}")


def plt_close(fig) -> None:
    import matplotlib.pyplot as plt
    plt.close(fig)
