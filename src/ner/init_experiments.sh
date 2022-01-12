#!/usr/bin/env bash

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Creates all train, dev & test datasets required in experiments 1 & 2
# and store them in $DATADIR (env)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ==============================================================================
# Let's get paranoid
set -eu
WORKDIR=`dirname $0` # Set WORKDIR to be the script location
EXP1=$WORKDIR/experiment_1
# ==============================================================================

# Make directories
mkdir -p $EXP1

# Build datasets for all experiments
echo "Initializing training, dev and test datasets"
python $WORKDIR/create_datasets.py