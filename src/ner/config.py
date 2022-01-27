"""
 Holds the global configuration.
"""

import os
import logging
from pathlib import Path

# =============================================================================
# region ~ Configurations

DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")

SPLIT_SEED = 42

# endregion


# =============================================================================
# region ~ Logging

logger = logging.getLogger("nerlogger")
LOG_LEVEL = logging.getLevelName("DEBUG" if DEBUG else "INFO")
logger.setLevel(LOG_LEVEL)
logformatter = logging.Formatter(
    "%(asctime)s ; %(levelname)s ; %(message)s", datefmt="%d/%m/%Y %I:%M:%S"
)
loghandler = logging.StreamHandler()
loghandler.setFormatter(logformatter)
logger.addHandler(loghandler)

# endregion
