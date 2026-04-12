Diploma Thesis, HSE

## Installation

Make sure you have [uv](https://docs.astral.sh/uv/getting-started/installation/) installed.

Install the package in editable mode:

```bash
uv tool install --editable lunicyto@.
```

### Download the Dataset

Use the CLI to download the dataset:

```bash
# Download with default settings (dataset: sipakmed, output: data/sipakmed)
lunicyto download

# Or specify custom dataset and output directory
lunicyto download <dataset-name> --output <path>
```

The default command downloads the SIPaKMeD dataset to `data/sipakmed/` directory.

