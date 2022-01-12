#!/usr/bin/env bash

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This script trains the CNN, BERT & BERT+PRE models on each subsampled 
# training set created using create_datasets.py and stored
# in $DATADIR/EXP1eriment_1/datasets
# 
# The validation and test sets remain untouched.
# Precision, recall and F1 scores are measured against the test set
# and stored in $DATADIR/EXP1eriment_1/metrics 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ==============================================================================
# Let's get paranoid
set -eu

# Constants
WORKDIR=`dirname $0` # Set WORKDIR to be the script location
EXP1=$WORKDIR/experiment_1
METRICS=$EXP1/metrics
MODEL_PRETRAINED="HueyNemud/berties-pretrained-das22"
# ==============================================================================


# ------------------------------------------------------------------------------
function train_sizes() {
    # Returns: sizes of the subsampled trainsets in $DATADIR in ascending order
    find $EXP1/*.spacy -printf "%f\n" |
    xargs -n1 echo - |
    sed 's/[^0-9]*//g' |
    sort -nk1 |
    uniq
}

# ------------------------------------------------------------------------------
function train_eval_cnn() {
    # Args: trainset sizes
    for size in $@; do
        train=$EXP1"/train_$size.spacy"
        dev=$EXP1"/dev_$size.spacy"
        test=$EXP1"/test_$size.spacy"

        echo "TRAINING: $train"
        echo "DEVELOMPENT: $dev"
        echo "EVALUATION: $test"

        python -m spacy train $WORKDIR/cnn_config.cfg \
            -o $EXP1 \
            --paths.train $train \
            --paths.dev $dev

        python -m spacy evaluate $EXP1/model-best $test \
            -o "$METRICS/cnn_${size}_test.json" \
            -dp $EXP1

        python -m spacy evaluate $EXP1/model-best $dev \
            -o "$METRICS/cnn_${size}_dev.json" \
            -dp $EXP1
    done
}

# ------------------------------------------------------------------------------
function train_eval_camembert() {
    # Args: trainset sizes
    for size in $@; do
        datasetdict="$EXP1/huggingface_$size"
        python $WORKDIR/camembert.py $datasetdict "$METRICS/camembert_${size}"
    done
}

# ------------------------------------------------------------------------------
function train_eval_camembert_pretrained() {
    # Args: trainset sizes
    for size in $@; do
        datasetdict="$EXP1/huggingface_$size"
        python $WORKDIR/camembert.py $datasetdict "$METRICS/camembert_pretrained_${size}" \
            --model $MODEL_PRETRAINED
    done
}

# ------------------------------------------------------------------------------
function main() {
    mkdir -p $METRICS
    train_eval_cnn $(train_sizes)
    #train_eval_camembert $(train_sizes)
    #train_eval_camembert_pretrained $(train_sizes)
}

# ==============================================================================
# Entry point
main
