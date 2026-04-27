"""K-Fold cross-validation для оценки робастности модели.

Зачем нужна ВКР:
  Один train/val/test сплит зависит от случайного seed. K-Fold показывает
  среднее и стандартное отклонение метрик по K разбиениям — это значительно
  убедительнее для рецензента, чем одно число.

Запуск:
  lunicyto cv --config config/train.toml --folds 5
"""

import logging
from pathlib import Path

import numpy as np
import typer
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader

from lunicyto.datasets.sipakmed import (
    CLASS_NAMES,
    SipakmedDataset,
    collect_samples,
    get_transform,
)
from lunicyto.models.baseline import build_baseline_model
from lunicyto.models.hybrid_vit_cnn import build_model
from lunicyto.training.trainer import Trainer
from lunicyto.utils.models import Config

logger = logging.getLogger(__name__)


def cross_validate(config: Config, n_folds: int = 5) -> dict:
    """Запускает n_folds-кратную стратифицированную кросс-валидацию.

    Каждый фолд:
      - test  = 1/n_folds часть данных
      - val   = 15% от оставшихся (train+val)
      - train = остаток

    Возвращает словарь с результатами каждого фолда и агрегатной статистикой.
    """
    all_samples = collect_samples(config.data.dir)
    labels = np.array([s[1] for s in all_samples])

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.data.seed)
    fold_results: list[dict] = []

    for fold, (trainval_idx, test_idx) in enumerate(
        skf.split(np.zeros(len(labels)), labels), start=1
    ):
        logger.info("\n" + "=" * 55)
        logger.info(f"  Fold {fold}/{n_folds}")
        logger.info("=" * 55)

        fold_output = config.output.dir / f"fold_{fold}"

        # Внутренний сплит train / val
        tv_labels = labels[trainval_idx]
        val_frac = config.data.val_split / (1.0 - 1.0 / n_folds)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=config.data.seed)
        rel_train_idx, rel_val_idx = next(sss.split(np.zeros(len(trainval_idx)), tv_labels))

        train_s = [all_samples[i] for i in trainval_idx[rel_train_idx]]
        val_s = [all_samples[i] for i in trainval_idx[rel_val_idx]]
        test_s = [all_samples[i] for i in test_idx]

        logger.info(f"  Train {len(train_s)} | Val {len(val_s)} | Test {len(test_s)}")

        img_size = config.data.img_size
        train_ds = SipakmedDataset(train_s, transform=get_transform(img_size, is_train=True))
        val_ds = SipakmedDataset(val_s, transform=get_transform(img_size, is_train=False))
        test_ds = SipakmedDataset(test_s, transform=get_transform(img_size, is_train=False))

        kw = dict(
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
            pin_memory=True,
            persistent_workers=config.data.num_workers > 0,
        )
        train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **kw)
        val_loader = DataLoader(val_ds, shuffle=False, **kw)
        test_loader = DataLoader(test_ds, shuffle=False, **kw)

        if config.model.model_type == "baseline":
            model = build_baseline_model(
                num_classes=config.model.num_classes,
                backbone=config.model.backbone,
                dropout=config.model.dropout,
                pretrained=config.model.pretrained,
            )
        else:
            model = build_model(
                num_classes=config.model.num_classes,
                backbone=config.model.backbone,
                transformer_dim=config.model.transformer_dim,
                transformer_heads=config.model.transformer_heads,
                transformer_layers=config.model.transformer_layers,
                mlp_ratio=config.model.mlp_ratio,
                dropout=config.model.dropout,
                drop_path_rate=config.model.drop_path_rate,
                pretrained=config.model.pretrained,
            )

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            output_dir=fold_output,
            learning_rate=config.training.learning_rate,
            backbone_lr_scale=config.training.backbone_lr_scale,
            weight_decay=config.training.weight_decay,
            epochs=config.training.epochs,
            warmup_epochs=config.training.warmup_epochs,
            label_smoothing=config.training.label_smoothing,
            mixup_alpha=config.training.mixup_alpha,
            grad_clip=config.training.grad_clip,
            early_stopping_patience=config.training.early_stopping_patience,
            class_names=CLASS_NAMES,
        )

        metrics = trainer.train()
        fold_results.append(metrics)
        logger.info(
            f"  Fold {fold} — "
            f"acc={metrics['accuracy']:.4f}  "
            f"f1={metrics['f1_macro']:.4f}  "
            f"auc={metrics.get('auc_roc_macro', float('nan')):.4f}"
        )

    # Агрегация
    accs = [r["accuracy"] for r in fold_results]
    f1s = [r["f1_macro"] for r in fold_results]
    aucs = [r["auc_roc_macro"] for r in fold_results if "auc_roc_macro" in r]

    summary = {
        "fold_results": fold_results,
        "n_folds": n_folds,
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "f1_macro_mean": float(np.mean(f1s)),
        "f1_macro_std": float(np.std(f1s)),
    }
    if aucs:
        summary["auc_roc_mean"] = float(np.mean(aucs))
        summary["auc_roc_std"] = float(np.std(aucs))

    # Сохраняем сводку
    summary_path = config.output.dir / "cv_summary.txt"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Cross-Validation Summary ({n_folds}-Fold)\n")
        f.write("=" * 40 + "\n")
        f.write(f"Accuracy : {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}\n")
        f.write(f"F1 macro : {summary['f1_macro_mean']:.4f} ± {summary['f1_macro_std']:.4f}\n")
        if aucs:
            f.write(f"AUC macro: {summary['auc_roc_mean']:.4f} ± {summary['auc_roc_std']:.4f}\n")
        f.write("\nPer-fold results:\n")
        for i, r in enumerate(fold_results, 1):
            f.write(
                f"  Fold {i}: acc={r['accuracy']:.4f}  "
                f"f1={r['f1_macro']:.4f}  "
                f"auc={r.get('auc_roc_macro', float('nan')):.4f}\n"
            )

    logger.info(f"\nСводка CV сохранена: {summary_path}")
    return summary


def main(
    config_path: Path = typer.Option(
        Path("config/train.toml"),
        "--config",
        "-c",
        help="Путь к toml-конфигурации.",
        exists=True,
        file_okay=True,
    ),
    folds: int = typer.Option(5, "--folds", "-k", help="Количество фолдов (K)."),
    output_dir: None | Path = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Переопределить output.dir из конфига.",
    ),
) -> None:
    config = Config.from_toml(config_path)
    if output_dir is not None:
        config.output.dir = output_dir
    summary = cross_validate(config, n_folds=folds)

    typer.echo("\n=== Cross-Validation Results ===")
    typer.echo(
        f"Accuracy : {summary['accuracy_mean'] * 100:.2f}% ± {summary['accuracy_std'] * 100:.2f}%"
    )
    typer.echo(
        f"F1 macro : {summary['f1_macro_mean'] * 100:.2f}% ± {summary['f1_macro_std'] * 100:.2f}%"
    )
    if "auc_roc_mean" in summary:
        typer.echo(
            f"AUC macro: {summary['auc_roc_mean'] * 100:.2f}% ± {summary['auc_roc_std'] * 100:.2f}%"
        )
