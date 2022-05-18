"""
    A srcipt to export the different paths used by the NER experiments: 
        - BASEDIR:    absolute path to the repository in the local environment
        - DATASETDIR: absolute path to the directory dataset/ that should exist in BASEDIR
        - NERDIR:     absolute path to the directory src/ner/

    If the notebooks are run in a Google Colab environment then this script will also mount 
    your Google Drive to the current Google Colab environment. By default GDrive will be
    mounted in /content/drive.
    .
"""

import dotenv
import pathlib
import os.path
import sys

_GDRIVE_MOUNT_POINT = "/content/drive"


def _mount_google_drive() -> str:
    """ Mount Google Drive to the Google Colab environment. Returns the mount point."""
    from google.colab import drive
    drive.mount(_GDRIVE_MOUNT_POINT)
    return pathlib.Path(_GDRIVE_MOUNT_POINT).absolute()


def base_dir() -> pathlib.Path:
    """ Returns the absolute path to the repository in the local environment."""
    is_env_googlecolab = 'google.colab' in str(get_ipython())

    if is_env_googlecolab:
        gdrive_path = _mount_google_drive()
        base_dir = gcolab_mount_point / os.environ["GCOLAB_BASE_DIR"]
    else:
        # If not on GColab, BASE will be the directory of this notebook
        this_notebook_location = os.path.dirname(
            os.path.realpath("__file__"))
        base_dir = pathlib.Path(this_notebook_location).parent.parent
        # make base_dir absolute
        base_dir = base_dir.absolute()

    return base_dir


BASEDIR = base_dir()
DATASETDIR = base_dir() / "dataset"  # Path to the input datasets
# Where to store the models & data produced by the NER experiments
NERDIR = base_dir() / "src/ner"
