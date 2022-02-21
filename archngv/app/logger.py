"""
Logging utilities.
"""

import logging

LOGGER = logging.getLogger("archngv")


def setup_logging(level):
    """Setup application logger."""
    logging.basicConfig(
        format="%(asctime)s;%(levelname)s;%(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        level=logging.WARNING,
    )
    LOGGER.setLevel(level)
