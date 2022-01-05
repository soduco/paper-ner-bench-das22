from pathlib import Path

DATA_DIR = Path("../data")
INPUT_DATASET = DATA_DIR / "ner_annotated_full.csv"

N_SPLITS = 2
RANDOM_STATE = 42

BERT_BASE_MODEL = "Jean-Baptiste/camembert-ner"

BERT_TRAINER_CONFIG = {
    "evaluation_strategy": "epoch",
    "learning_rate": 1e-4,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "num_train_epochs": 3,
    "weight_decay": 1e-5,
}
