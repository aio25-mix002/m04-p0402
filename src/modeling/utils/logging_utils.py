"""Simple logging utility for the heart disease project.

The default Python logging module is configured here with a basic
formatter.  Import `logger` from this module whenever you need to log
messages; this avoids configuring the logger in multiple places.
"""

import logging
import sys

logger = logging.getLogger("heart_disease")

if not logger.handlers:
    handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
