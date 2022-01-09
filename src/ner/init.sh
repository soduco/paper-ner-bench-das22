#!/usr/bin/env bash

# ==============================================================================
# Let's get paranoid
set -eu

WORKDIR=`dirname $0`
# ==============================================================================

# Make directories
mkdir -p $DATADIR/experiment_1
mkdir -p $DATADIR/datasets

# Build datasets for all experiments
echo "Initializing training, dev and test datasets"
python $WORKDIR/create_datasets.py