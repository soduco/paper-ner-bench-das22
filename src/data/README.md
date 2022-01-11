# Experiment data

This directory contains all the data needed to reproduce the experiments 1 and 2 from the paper, along with the gold dataset and the dataset used to pre-train CamemBERT.

The scripts in src/ner will also store their output (metrics, computed datasets and fine-tuned models) here.

In order to run the experiments you will need to set the environment variable `DATADIR` to the absolute path to this directory.

## Details
The data directory contain the following:
- **gold/** the gold dataset
- **pretraining**: the dataset used to pre-train CamemBERT on historical directories
