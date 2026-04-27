import logging
from pathlib import Path
from typing import Annotated

import typer
from kaggle.api.kaggle_api_extended import KaggleApi

logger = logging.getLogger(__name__)


def download_kaggle_dataset(
    dataset: str,
    output_dir: None | Path = None,
    unzip: bool = True,
) -> Path:
    if output_dir is None:
        project_root = Path(__file__).resolve().parent.parent.parent
        dataset_name = dataset.split("/")[-1]
        output_dir = project_root / "data" / dataset_name

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing Kaggle API...")
    api = KaggleApi()
    api.authenticate()

    logger.info(f"Downloading dataset '{dataset}' to {output_dir.absolute()}")

    api.dataset_download_files(
        dataset=dataset,
        path=str(output_dir),
        unzip=unzip,
        quiet=False,
    )

    logger.info(f"Dataset successfully downloaded to {output_dir.absolute()}")
    return output_dir


def main(
    dataset: Annotated[
        str,
        typer.Argument(help="Kaggle dataset identifier (e.g., 'user/dataset-name')"),
    ] = "prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed",
    output_dir: Annotated[
        None | Path,
        typer.Option(
            "--output",
            "-o",
            help="Output directory for the dataset. Defaults to data/sipakmed",
        ),
    ] = Path("data/sipakmed"),
    no_unzip: Annotated[
        bool,
        typer.Option(
            "--no-unzip",
            help="Don't unzip the downloaded files",
        ),
    ] = False,
) -> None:
    download_kaggle_dataset(
        dataset=dataset,
        output_dir=output_dir,
        unzip=not no_unzip,
    )
