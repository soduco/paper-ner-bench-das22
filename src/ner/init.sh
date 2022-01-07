#!/bin/bash

# Let's get paranoid
set -eu

WORKDIR=$(dirname "$0")

# Build datasets for all experiments
echo "Initializing training, dev and test datasets"
python ${WORKDIR}/create_datasets.py
