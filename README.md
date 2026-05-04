## Installation

Make sure you have [uv](https://docs.astral.sh/uv/getting-started/installation/) installed.

Install the package in editable mode:

```bash
uv tool install --editable lunicyto@.
```

With `--editable`, Python source changes take effect immediately — no reinstall needed for `.py` edits.

> **CUDA note.** On Linux and Windows, `pyproject.toml` redirects `torch` and
> `torchvision` to the PyTorch CUDA 12.4 index
> (`https://download.pytorch.org/whl/cu124`) via `[tool.uv.sources]`.
> uv resolves CUDA wheels automatically — no extra flags needed.
> On macOS the standard PyPI wheels are used (MPS is supported natively).


### Download the Dataset

Use the CLI to download the dataset:

```bash
# Download with default settings (dataset: sipakmed, output: data/sipakmed)
lunicyto download

# Or specify custom dataset and output directory
lunicyto download <dataset-name> --output <path>
```

The default command downloads the SIPaKMeD dataset to `data/sipakmed/` directory.

