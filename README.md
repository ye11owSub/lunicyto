Diploma Thesis, HSE

## Installation

Make sure you have [uv](https://docs.astral.sh/uv/getting-started/installation/) installed.

Install the package in editable mode:

```bash
uv tool install --editable lunicyto@.
```

With `--editable`, Python source changes take effect immediately — no reinstall needed for `.py` edits.

### When you must reinstall

Reinstall is required when `pyproject.toml` changes (entry points, dependencies, version bump):

```bash
uv tool install --editable lunicyto@. --reinstall
```

Or explicitly:

```bash
uv tool uninstall lunicyto
uv tool install --editable lunicyto@.
```

### Verify the running version

Always check the version before training to confirm you're running the expected code:

```bash
lunicyto --version        # prints: lunicyto v0.4.0
lunicyto -V               # same, short flag
```

The version is also printed in the first log line of every training run:

```
[INFO] Lunicyto v0.4.0 | device: cuda | epochs: 50
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

