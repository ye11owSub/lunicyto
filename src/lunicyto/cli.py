import logging
from textwrap import dedent

import typer

from lunicyto.utils import download_data, explore

app = typer.Typer(
    help=dedent("""\
        Lunicyto — cervical cell classification (SIPaKMeD).

        Installation: uv tool install --editable lunicyto@.
    """),
    no_args_is_help=True,
)

app.command("download")(download_data.main)
app.command("info")(explore.main)


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
