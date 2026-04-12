import logging
from textwrap import dedent

import typer
from lunicyto.utils import download_data


app = typer.Typer(
    help=dedent("""\
        First, make sure you have `uv installed
        <https://docs.astral.sh/uv/getting-started/installation/>`_.

        Recommended:

        install in editable mode: ``uv tool install --editable lunicyto@.``

        Alternatively,

        install current state: ``uv tool install lunicyto@.``

        upgrade to current state: ``uv tool upgrade lunicyto@.``
    """),
    no_args_is_help=True,
)

app.command("download")(download_data.main)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    app()


if __name__ == "__main__":
    main()
