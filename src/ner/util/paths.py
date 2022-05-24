"""
    A srcipt to export the different paths used by the NER experiments: 
        - BASEDIR:    absolute path to the repository in the local environment
        - DATASETDIR: absolute path to the directory dataset/ that should exist in BASEDIR
        - NERDIR:     absolute path to the directory src/ner/    .
"""

import pathlib
import os
import sys

_GDRIVE_MOUNT_POINT = pathlib.Path("/content/drive")
_GDRIVE_ROOT_FOLDER = pathlib.Path("MyDrive")

def _mount_google_drive() -> str:
    """ Mount Google Drive to the Google Colab environment. Returns the mount point."""
    from google.colab import drive
    already_mounted = os.path.isdir(_GDRIVE_MOUNT_POINT / _GDRIVE_ROOT_FOLDER)
    if not already_mounted:
        drive.mount(str(_GDRIVE_MOUNT_POINT))
    return _GDRIVE_MOUNT_POINT.absolute()


def base_dir() -> pathlib.Path:
    """ Returns the absolute path to the repository in the local environment."""
    is_env_googlecolab = 'google.colab' in str(get_ipython())

    if is_env_googlecolab:
        gcolab_mount_point = _mount_google_drive()
        base_dir = gcolab_mount_point / _GDRIVE_ROOT_FOLDER / os.environ['GDRIVE_PAPER_FOLDER']
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
