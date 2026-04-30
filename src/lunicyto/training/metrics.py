import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str] | None = None,
    y_score: np.ndarray | None = None,
) -> dict:
    """Вычисляет полный набор метрик классификации.

    Args:
        y_true:      истинные метки.
        y_pred:      предсказанные метки.
        class_names: названия классов (для отчёта).
        y_score:     матрица вероятностей (N, C) после softmax.
                     Нужна для AUC-ROC; если None — AUC не считается.

    Returns:
        dict с ключами: accuracy, f1_macro, f1_per_class,
        sensitivity_per_class, specificity_per_class,
        confusion_matrix, report, и опционально
        auc_roc_macro, auc_roc_per_class.
    """
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    n_classes = len(class_names) if class_names else int(y_true_np.max()) + 1
    labels = list(range(n_classes))

    acc = accuracy_score(y_true_np, y_pred_np)
    f1_macro = f1_score(y_true_np, y_pred_np, average="macro", zero_division=0)
    f1_per = f1_score(y_true_np, y_pred_np, average=None, zero_division=0, labels=labels).tolist()

    # Sensitivity = Recall per class  (TP / (TP + FN))
    sensitivity = recall_score(
        y_true_np, y_pred_np, average=None, zero_division=0, labels=labels
    ).tolist()

    # Specificity per class = TN / (TN + FP)  в схеме one-vs-rest
    cm = confusion_matrix(y_true_np, y_pred_np, labels=labels)
    specificity_per_class: list[float] = []
    for i in range(n_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        spec = float(tn) / float(tn + fp) if (tn + fp) > 0 else 0.0
        specificity_per_class.append(spec)

    report = classification_report(y_true_np, y_pred_np, target_names=class_names, zero_division=0)

    result: dict = {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_per_class": f1_per,
        "sensitivity_per_class": sensitivity,
        "specificity_per_class": specificity_per_class,
        "confusion_matrix": cm,
        "report": report,
    }

    # AUC-ROC требует вероятностных оценок
    if y_score is not None:
        # Макро-AUC: сохраняем сразу, до per-class вычислений
        try:
            auc_macro = roc_auc_score(
                y_true_np,
                y_score.astype(np.float64),
                multi_class="ovr",
                average="macro",
                labels=labels,
            )
            result["auc_roc_macro"] = float(auc_macro)
        except ValueError as e:
            logger.debug("AUC macro computation failed: %s", e)

        # Per-class AUC: отдельный try — падение здесь не должно убирать макро-AUC
        try:
            result["auc_roc_per_class"] = [
                float(roc_auc_score((y_true_np == i).astype(int), y_score[:, i].astype(np.float64)))
                for i in range(n_classes)
            ]
        except ValueError as e:
            logger.debug("AUC per-class computation failed: %s", e)

    return result


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    save_path: None | str | Path = None,
    title: str = "Confusion Matrix",
) -> plt.Figure:
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_xlabel("Predicted class", fontsize=12)
    ax.set_ylabel("True class", fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    train_accs: list[float],
    val_accs: list[float],
    save_path: None | str | Path = None,
) -> plt.Figure:
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, "b-", label="Train loss")
    ax1.plot(epochs, val_losses, "r-", label="Val loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_accs, "b-", label="Train accuracy")
    ax2.plot(epochs, val_accs, "r-", label="Val accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
