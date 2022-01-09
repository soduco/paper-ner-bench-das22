"""
 Holds the global configuration used in all python scripts related to NER.
"""

import os
import logging
from pathlib import Path

# =============================================================================
# region ~ Configurations

DATA_DIR = Path(os.environ["DATADIR"])

assert DATA_DIR.exists()

GOLD = DATA_DIR / "gold/gold.csv"
SEED = 42  # Random seed to generate train/validation/test datasets
MIN_TRAINSET_SIZE = 30
DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
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
