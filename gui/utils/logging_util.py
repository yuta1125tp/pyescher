import logging
import sys
from pathlib import Path

from .datetime_util import get_YmdHMSf_timestamp


def init_logging(
    level: int = logging.INFO,
    format: str = "%(asctime)s:%(filename)s:L%(lineno)d:%(funcName)s:[%(levelname)s] %(message)s",
    logfile: Path = None,
    logdir: Path = Path("logs"),
) -> None:
    """"""
    handlers = [logging.StreamHandler()]

    if logfile is not None:
        if logdir is not None:
            print("ignore logdir")
        if len(str(logfile.parent)):
            logfile.parent.mkdir(exist_ok=True, parents=True)
        handlers.append(logging.FileHandler(str(logfile)))
    else:
        logfile = logdir / "log.{}.{}.txt".format(
            Path(sys.argv[0]).stem, get_YmdHMSf_timestamp()
        )
        if len(str(logfile.parent)):
            logfile.parent.mkdir(exist_ok=True, parents=True)
        handlers.append(logging.FileHandler(str(logfile)))

    logging.basicConfig(
        level=level,
        format=format,
        handlers=handlers,
    )
