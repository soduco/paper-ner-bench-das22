"""
 Holds the global configuration.
Remarks:
- DATA_DIR should be the absolute path to paper-ner-bench-das22/dataset
- WORKDIR is set to the location of this scipt.
- generated train/dev/test datasets and metrics will be outputed in WORKDIR/computed
"""

import os
import logging
from pathlib import Path

# =============================================================================
# region ~ Configurations

DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")

# Paths globals
DATA_DIR = Path(os.getenv("DATADIR"))
assert DATA_DIR.exists(), "DATADIR %s does not exist." % DATA_DIR

WORKDIR_DEFAULT = os.path.dirname(__file__)
WORKDIR = Path(os.getenv("WORKDIR", WORKDIR_DEFAULT))
assert WORKDIR.exists(), "Workdir: %s is not a valid directory" % WORKDIR

# Training globals
GOLD = DATA_DIR / "supervised/10-ref-ocr-ner-json/gold.csv"
SEED = 42  # Ensure the train/dev split to be repeatable

# Experiment 1
EXPERIMENT_1 = WORKDIR / "experiment_1"
MIN_TRAINSET_SIZE = 30

# Experiment 2
EXPERIMENT_2 = WORKDIR / "experiment_2"


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
