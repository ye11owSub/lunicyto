import logging
from textwrap import dedent

import typer

import lunicyto
from lunicyto.utils import cross_validate, download_data, explore, train

app = typer.Typer(
    help=dedent("""\
        Lunicyto — cervical cell classification (SIPaKMeD).

        Installation: uv tool install --editable lunicyto@.
    """),
    no_args_is_help=True,
)

app.command("download")(download_data.main)
app.command("info")(explore.main)
app.command("train")(train.main)
app.command("cv")(cross_validate.main)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"lunicyto v{lunicyto.__version__}")
        raise typer.Exit()


@app.callback()
def _global_options(
    version: bool = typer.Option(  # noqa: B008
        False,
        "--version",
        "-V",
        help="Показать версию и выйти.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    pass


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
