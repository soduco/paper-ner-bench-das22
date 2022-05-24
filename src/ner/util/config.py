"""
 Holds the global configuration.
"""

import util.paths
import os
import logging
from pathlib import Path

# =============================================================================
# region ~ Configurations

# If true, text versions of the spacy datasets will be saved along with the .spacy files.
DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")

# Random seed used in train/dev/test. Do not change it if you want to recreate the paper results.
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

# =============================================================================
# region ~ Paths


BASEDIR = util.paths.BASEDIR
DATASETDIR = util.paths.DATASETDIR
NERDIR = util.paths.NERDIR

# endregion


# =============================================================================
# region ~ Repeatability

AS_IN_THE_PAPER = os.getenv(
    "AS_IN_THE_PAPER", "True").lower() in ("true", "1", "t")

# endregion


def show() -> None:
    logger.info("======= CONFIGURATION =======")
    logger.info(f"BASEDIR: {BASEDIR}")
    logger.info(
        f"Input datasets will be loaded from DATASETDIR {DATASETDIR}")
    logger.info(
        f"Training data and models will be saved to NERDIR {NERDIR}")
    logger.info(f"Debug mode is {'ON' if DEBUG else 'OFF.'}")
    logger.info(f"Random seed: {SPLIT_SEED}")
    logger.info(f"Enable reproducibility checks: {AS_IN_THE_PAPER}")
    logger.info("============================")
