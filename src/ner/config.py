from pathlib import Path
import logging

# Assuming datadir is located in the parent directory of this module
DATA_DIR = Path(__file__).parents[1] / "data"

GOLD = DATA_DIR / "gold/gold.csv"

N_SPLITS = 2
SEED = 42  # Used to generate train/validation/test datasets
MIN_TRAINSET_SIZE = 50

BERT_BASE_MODEL = "Jean-Baptiste/camembert-ner"

BERT_TRAINER_CONFIG = {
    "evaluation_strategy": "epoch",
    "learning_rate": 1e-4,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "num_train_epochs": 3,
    "weight_decay": 1e-5,
}

# Set logging configuration for the whole project
logging.basicConfig()
logging.root.setLevel(logging.INFO)
