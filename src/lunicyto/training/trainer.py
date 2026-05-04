import logging
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image as PILImage
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import lunicyto
from lunicyto.training.early_stopping import EarlyStopping
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


def _save_image_grid(
    samples: list[tuple[Path, int, int]],
    class_names: list[str],
    save_path: Path,
    title: str,
    mode: str = "wrong",
    thumb_size: int = 112,
    ncols: int = 8,
) -> None:
    n = len(samples)
    if n == 0:
        return
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.8, nrows * 2.2))
    axes_flat: list = np.array(axes).flatten().tolist() if n > 1 else [axes]

    for i, ax in enumerate(axes_flat):
        ax.axis("off")
        if i >= n:
            continue
        path, true_label, pred_label = samples[i]
        img = PILImage.open(path).convert("RGB").resize((thumb_size, thumb_size))
        ax.imshow(np.array(img))
        true_name = class_names[true_label] if true_label < len(class_names) else str(true_label)
        pred_name = class_names[pred_label] if pred_label < len(class_names) else str(pred_label)
        color = "red" if mode == "wrong" else "green"
        if mode == "wrong":
            caption = f"T: {true_name[:10]}\nP: {pred_name[:10]}"
        else:
            caption = true_name[:14]
        ax.set_title(caption, fontsize=5.5, color=color, pad=2)

    fig.suptitle(title, fontsize=9, y=1.01)
    plt.tight_layout(pad=0.4)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


class WarmupCosineScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
    ):
        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(max(1, warmup_epochs))
            progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        super().__init__(optimizer, lr_lambda)


class Trainer:
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
        class_names: list | None = None,
        use_tta: bool = True,
        test_samples: list | None = None,
    ):
        self.device = _get_device()
        logger.info(f"Device: {self.device}")

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
        self.use_tta = use_tta
        self.test_samples = test_samples  # list[tuple[Path, int, str]] | None

        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        backbone_params = [p for n, p in model.named_parameters() if n.startswith("backbone")]
        head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone")]
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

        self.early_stopping = EarlyStopping(patience=early_stopping_patience, mode="max")

        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.writer = SummaryWriter(log_dir=str(self.output_dir / "tb_logs"))

        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.train_accs: list[float] = []
        self.val_accs: list[float] = []
        self.best_val_acc = 0.0

    def train(self) -> dict:
        logger.info(
            f"Lunicyto v{lunicyto.__version__} | device: {self.device} | epochs: {self.epochs}"
        )
        logger.info(
            f"Train: {len(self.train_loader)} batches | "
            f"Val: {len(self.val_loader)} batches | "
            f"Test: {len(self.test_loader)} batches"
        )

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()

            train_loss, train_acc = self._train_epoch(epoch)
            val_loss, val_acc, val_f1, val_auc = self._val_epoch(epoch)

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
                f"val loss={val_loss:.4f} acc={val_acc:.4f} "
                f"f1={val_f1:.4f} auc={val_auc:.4f} | "
                f"lr={lr:.2e} | {elapsed:.1f}s"
            )

            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/val", val_loss, epoch)
            self.writer.add_scalar("Accuracy/train", train_acc, epoch)
            self.writer.add_scalar("Accuracy/val", val_acc, epoch)
            self.writer.add_scalar("F1_macro/val", val_f1, epoch)
            self.writer.add_scalar("AUC_macro/val", val_auc, epoch)
            self.writer.add_scalar("LR", lr, epoch)

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self._save_checkpoint(epoch, val_acc)

            if self.early_stopping(val_acc):
                logger.info(
                    f"Early stopping on epoch {epoch} (best val_acc={self.best_val_acc:.4f})"
                )
                break

        self.writer.close()

        ckpt_path = self.output_dir / "best_model.pth"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(ckpt["model_state_dict"])
            logger.info(
                f"Restored best checkpoint (epoch {ckpt['epoch']}, "
                f"val_acc={ckpt['val_acc']:.4f}) for final evaluation"
            )

        tta_label = " (с TTA)" if self.use_tta else ""
        logger.info(f"Финальная оценка на тесте{tta_label}...")
        test_metrics = self._evaluate(self.test_loader, phase="test", use_tta=self.use_tta)
        self._save_results(test_metrics)
        if self.test_samples:
            self._save_prediction_samples(self.test_samples)

        logger.info("\nResults:")
        logger.info(f"Accuracy:  {test_metrics['accuracy']:.4f}")
        logger.info(f"F1 macro:  {test_metrics['f1_macro']:.4f}")
        if "auc_roc_macro" in test_metrics:
            logger.info(f"AUC macro: {test_metrics['auc_roc_macro']:.4f}")
        logger.info(f"\n{test_metrics['report']}")

        return test_metrics

    def _train_epoch(self, epoch: int) -> tuple[float, float]:
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
            if self.mixup_alpha > 0:
                # Soft accuracy под Mixup: взвешенная сумма попаданий по обоим лейблам.
                # preds == labels считает только y_a и игнорирует y_b — отсюда
                # искусственно низкая train accuracy при высокой val accuracy.
                correct += (
                    (lam * (preds == y_a).float() + (1 - lam) * (preds == y_b).float()).sum().item()
                )
            else:
                correct += (preds == labels).sum().item()
            total += batch_size

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / total, correct / total

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> tuple[float, float, float, float]:
        self.model.eval()
        total_loss = 0.0
        total = 0
        all_probs: list[np.ndarray] = []
        all_labels: list[int] = []

        for imgs, labels in self.val_loader:
            imgs = imgs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(imgs)
                loss = self.criterion(logits, labels)

            batch_size = imgs.size(0)
            total_loss += loss.item() * batch_size  # сумма по сэмплам
            total += batch_size

            probs = torch.softmax(logits.float(), dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.extend(labels.cpu().tolist())

        y_score = np.concatenate(all_probs, axis=0)
        y_pred = y_score.argmax(axis=1).tolist()
        metrics = compute_metrics(all_labels, y_pred, self.class_names, y_score=y_score)
        auc = metrics.get("auc_roc_macro", 0.0)
        return total_loss / total, metrics["accuracy"], metrics["f1_macro"], auc

    @torch.no_grad()
    def _evaluate(
        self,
        loader: DataLoader,
        phase: str = "test",
        use_tta: bool = False,
    ) -> dict:
        self.model.eval()
        all_probs: list[np.ndarray] = []
        all_labels: list[int] = []

        # dims для каждой TTA-аугментации; [] = оригинал без флипа
        tta_flips: list[list[int]] = [[], [-1], [-2], [-1, -2]] if use_tta else [[]]

        for imgs, labels in tqdm(loader, desc=f"Evaluate ({phase})", leave=False):
            imgs = imgs.to(self.device, non_blocking=True)

            batch_probs: list[torch.Tensor] = []
            for dims in tta_flips:
                aug = torch.flip(imgs, dims=dims) if dims else imgs
                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    logits = self.model(aug)
                batch_probs.append(torch.softmax(logits.float(), dim=1))

            avg_probs = torch.stack(batch_probs).mean(0)
            all_probs.append(avg_probs.cpu().numpy())
            all_labels.extend(labels.tolist())

        y_score = np.concatenate(all_probs, axis=0)
        y_pred = y_score.argmax(axis=1).tolist()
        return compute_metrics(all_labels, y_pred, self.class_names, y_score=y_score)

    def _save_checkpoint(self, epoch: int, val_acc: float) -> None:
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
        cm_fig = plot_confusion_matrix(
            test_metrics["confusion_matrix"],
            self.class_names,
            save_path=self.output_dir / "confusion_matrix.png",
            title="Confusion Matrix (Test)",
        )
        plt.close(cm_fig)

        if self.train_losses:
            curve_fig = plot_training_curves(
                self.train_losses,
                self.val_losses,
                self.train_accs,
                self.val_accs,
                save_path=self.output_dir / "training_curves.png",
            )
            plt.close(curve_fig)

        report_path = self.output_dir / "test_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"Accuracy:  {test_metrics['accuracy']:.4f}\n")
            f.write(f"F1 macro:  {test_metrics['f1_macro']:.4f}\n")
            if "auc_roc_macro" in test_metrics:
                f.write(f"AUC macro: {test_metrics['auc_roc_macro']:.4f}\n")
            f.write("\n--- Sensitivity (Recall) per class ---\n")
            for name, val in zip(
                self.class_names, test_metrics.get("sensitivity_per_class", []), strict=False
            ):
                f.write(f"  {name:35s}: {val:.4f}\n")
            f.write("\n--- Specificity per class ---\n")
            for name, val in zip(
                self.class_names, test_metrics.get("specificity_per_class", []), strict=False
            ):
                f.write(f"  {name:35s}: {val:.4f}\n")
            if "auc_roc_per_class" in test_metrics:
                f.write("\n--- AUC-ROC per class ---\n")
                for name, val in zip(
                    self.class_names, test_metrics["auc_roc_per_class"], strict=False
                ):
                    f.write(f"  {name:35s}: {val:.4f}\n")
            f.write(f"\n{test_metrics['report']}")
        logger.info(f"Отчёт сохранён: {report_path}")

    def _save_prediction_samples(
        self,
        test_samples: list[tuple[Path, int, str]],
        n_correct_per_class: int = 5,
        img_size: int = 224,
    ) -> None:
        """Run inference on raw test images and save visual inspection grids.

        Produces two PNGs inside ``output_dir/prediction_samples/``:

        * ``wrong_predictions.png`` — up to 32 misclassified cells with
          true / predicted class labels.  Ordered by class so errors from
          the same category are grouped together.

        * ``correct_samples.png`` — up to ``n_correct_per_class`` correctly
          classified cells per class (all classes shown side by side).

        The method uses the model state currently loaded (best checkpoint is
        restored by :meth:`train` before calling this).
        """
        save_dir = self.output_dir / "prediction_samples"
        save_dir.mkdir(exist_ok=True)

        # Minimal eval transform — same as SipakmedDataset val/test pipeline
        transform = T.Compose(
            [
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.model.eval()
        correct_by_class: dict[int, list[tuple[Path, int, int]]] = {
            i: [] for i in range(len(self.class_names))
        }
        wrong_by_class: dict[int, list[tuple[Path, int, int]]] = {
            i: [] for i in range(len(self.class_names))
        }

        with torch.no_grad():
            for path, true_label, _group in tqdm(
                test_samples, desc="Saving prediction samples", leave=False
            ):
                img = PILImage.open(path).convert("RGB")
                tensor = transform(img).unsqueeze(0).to(self.device, non_blocking=True)
                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    logits = self.model(tensor)
                pred = int(logits.float().argmax(dim=1).item())
                triple = (path, true_label, pred)
                if pred == true_label:
                    correct_by_class[true_label].append(triple)
                else:
                    wrong_by_class[true_label].append(triple)

        # --- wrong predictions grid (up to 32 samples, grouped by true class) ---
        wrong_all: list[tuple[Path, int, int]] = []
        for cls_idx in range(len(self.class_names)):
            wrong_all.extend(wrong_by_class[cls_idx])
        if wrong_all:
            _save_image_grid(
                wrong_all[:32],
                self.class_names,
                save_dir / "wrong_predictions.png",
                title=f"Wrong predictions  ({len(wrong_all)} errors)",
                mode="wrong",
            )
            logger.info(
                f"Wrong predictions grid ({len(wrong_all)} errors) → "
                f"{save_dir}/wrong_predictions.png"
            )
        else:
            logger.info("No wrong predictions — perfect test accuracy!")

        correct_sample: list[tuple[Path, int, int]] = []
        for cls_idx in range(len(self.class_names)):
            correct_sample.extend(correct_by_class[cls_idx][:n_correct_per_class])
        if correct_sample:
            _save_image_grid(
                correct_sample,
                self.class_names,
                save_dir / "correct_samples.png",
                title=f"Correct predictions ({n_correct_per_class} per class)",
                mode="correct",
                ncols=n_correct_per_class,
            )
            logger.info(f"Correct samples grid → {save_dir}/correct_samples.png")

        n_total = len(test_samples)
        n_wrong = len(wrong_all)
        logger.info(
            f"Sample inspection summary: {n_total - n_wrong}/{n_total} correct "
            f"({(n_total - n_wrong) / n_total * 100:.1f}%)"
        )
        for cls_idx, cls_name in enumerate(self.class_names):
            n_err = len(wrong_by_class[cls_idx])
            n_cls = len(correct_by_class[cls_idx]) + n_err
            logger.info(f"  {cls_name:35s}: {n_err:3d}/{n_cls} wrong")
