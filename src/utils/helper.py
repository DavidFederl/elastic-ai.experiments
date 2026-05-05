import logging
from pathlib import Path


def setup_logging(log_dir: Path, verbose: bool) -> None:
    """Setup Logging Facility.

    Args:
        log_dir (str): Log directory.
        debug (bool): Debug mode.
        verbose (bool): Verbose mode.

    Returns:
        Logger: Logger instance.
    """
    log_dir.mkdir(exist_ok=True, parents=True)

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        filename=log_dir.joinpath("app.log"),
        filemode="a",
        format="%(asctime)s %(levelname)-8s %(filename)s:%(lineno)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    with log_dir.joinpath("app.log").open("a") as log_file:
        log_file.write("\n==================================================\n\n")
