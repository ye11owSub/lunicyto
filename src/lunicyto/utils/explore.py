from pathlib import Path
from typing import Annotated

import typer
from PIL import Image

from lunicyto.datasets.sipakmed import collect_samples, dataset_info


def explore_dataset(
    data_dir: Path,
) -> None:
    info = dataset_info(data_dir)
    typer.echo(f"Total images: {info['total']}")
    typer.echo(f"Classes: {info['num_classes']}")
    typer.echo("\nClass distribution:")

    for cls, cnt in info["per_class"].items():
        pct = cnt / info["total"] * 100
        bar = "█" * int(pct)
        typer.echo(f"  {cls:35s} {cnt:5d}  {pct:5.1f}%  {bar}")

    samples = collect_samples(data_dir)
    first = samples[0]
    img = Image.open(first[0]).convert("RGB")
    class_names = info["class_names"]
    typer.echo(f"\nSample image: {first[0]}")
    typer.echo(f"Size: {img.size}, Class: {class_names[first[1]]}")
    typer.echo("\nDataset is correct.")


def main(
    data_dir: Annotated[
        Path,
        typer.Option(
            "--data-dir",
            "-s",
            help="Dataset SIPaKMeD source directory",
        ),
    ] = Path("data/sipakmed"),
) -> None:
    explore_dataset(data_dir=data_dir)
