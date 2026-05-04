from pathlib import Path

import typer
from torch.utils.data import DataLoader

from lunicyto.datasets.sipakmed import (
    CLASS_NAMES,
    SipakmedDataset,
    collect_samples,
    dataset_info,
    get_transform,
    split_samples,
)
from lunicyto.models.baseline import build_baseline_model
from lunicyto.models.hybrid_vit_cnn import build_model
from lunicyto.training.trainer import Trainer
from lunicyto.utils.models import Config


def train(config: Config, data_dir: None | Path, output_dir: None | Path) -> None:
    if data_dir is not None:
        config.data.dir = data_dir
    if output_dir is not None:
        config.output.dir = output_dir

    typer.echo(f"\nДатасет: {config.data.dir}")
    info = dataset_info(str(config.data.dir))
    typer.echo(f"Всего изображений: {info['total']}")
    for cls, cnt in info["per_class"].items():
        typer.echo(f"  {cls}: {cnt}")
    typer.echo()

    all_samples = collect_samples(config.data.dir)
    train_s, val_s, test_s = split_samples(
        all_samples,
        val_split=config.data.val_split,
        test_split=config.data.test_split,
        seed=config.data.seed,
    )

    img_size = config.data.img_size
    train_ds = SipakmedDataset(train_s, transform=get_transform(img_size, is_train=True))
    val_ds = SipakmedDataset(val_s, transform=get_transform(img_size, is_train=False))
    test_ds = SipakmedDataset(test_s, transform=get_transform(img_size, is_train=False))

    nw = config.data.num_workers
    bs = config.data.batch_size
    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        pin_memory=True,
        drop_last=True,
        persistent_workers=nw > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
        persistent_workers=nw > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
        persistent_workers=nw > 0,
    )

    typer.echo(f"Backbone: {config.model.backbone}  |  model_type: {config.model.model_type}")

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

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    typer.echo(f"Параметры модели: {n_params:.1f}M\n")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        output_dir=str(config.output.dir),
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
        test_samples=test_s,
    )

    test_metrics = trainer.train()

    typer.echo("\n=== final results ===")
    typer.echo(f"Accuracy : {test_metrics['accuracy'] * 100:.2f}%")
    typer.echo(f"F1 macro : {test_metrics['f1_macro'] * 100:.2f}%")
    if "auc_roc_macro" in test_metrics:
        typer.echo(f"AUC macro: {test_metrics['auc_roc_macro'] * 100:.2f}%")
    typer.echo(f"results saved to: {config.output.dir}")


def main(
    config_path: Path = typer.Option(
        Path("config/train.toml"),
        "--config",
        "-c",
        help="Path to toml-config",
        exists=True,
        file_okay=True,
    ),
    data_dir: None | Path = typer.Option(
        None,
        "--data-dir",
        help="Set data.dir",
    ),
    output_dir: None | Path = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Set output.dir",
    ),
) -> None:
    config = Config.from_toml(config_path)
    train(config=config, data_dir=data_dir, output_dir=output_dir)
