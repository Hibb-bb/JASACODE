import logging
import os
import sys
from datetime import datetime
from typing import Optional


def setup_logger(output_dir: str,
                 log_level: str = "INFO",
                 log_file: Optional[str] = None) -> logging.Logger:
    """
    Creates a process-wide logger that logs to stdout and (optionally) a file.
    """
    os.makedirs(output_dir, exist_ok=True)

    logger = logging.getLogger("train")
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.propagate = False  # avoid duplicate logs if root logger configured elsewhere

    # Clear any existing handlers (important in notebooks / re-runs)
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(getattr(logging, log_level.upper()))
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Optional file handler
    if log_file is None:
        # default log filename in output_dir
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(output_dir, f"train_{ts}.log")
    fh = logging.FileHandler(log_file)
    fh.setLevel(getattr(logging, log_level.upper()))
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info(f"Logging to file: {log_file}")
    return logger
