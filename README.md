## Installation

Make sure you have [uv](https://docs.astral.sh/uv/getting-started/installation/) installed.

Install the package in editable mode:

```bash
uv tool install --editable lunicyto@.
```

With `--editable`, Python source changes take effect immediately — no reinstall needed for `.py` edits.

### Download the Dataset

Use the CLI to download the dataset:

```bash
lunicyto download
lunicyto download <dataset-name> --output <path>
```

The default command downloads the SIPaKMeD dataset to `data/sipakmed/` directory.

