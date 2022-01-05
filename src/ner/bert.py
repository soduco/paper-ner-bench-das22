import json
import config as cfg
import os
from datasets import load_from_disk
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification
from transformers import TrainingArguments, Trainer, DataCollatorForTokenClassification
from datasets import load_metric
import numpy as np

label_id_map = {
    "O": 0,
    "I-LOC": 1,
    "I-PER": 2,
    "I-MISC": 3,
    "I-ORG": 4,
    "I-CARDINAL": 5,
    "I-ACT": 6,
    "I-TITRE": 7,
    "I-FT": 8,
}

label2id = label_id_map

id2label = {v: k for k, v in label2id.items()}

model = AutoModelForTokenClassification.from_pretrained(
    cfg.BERT_BASE_MODEL,
    num_labels=len(label2id),
    ignore_mismatched_sizes=True,
    id2label=id2label,
    label2id=label2id,
)

tokenizer = AutoTokenizer.from_pretrained(cfg.BERT_BASE_MODEL)

# TRAINING CONFIGURATION
training_args = TrainingArguments("berties", **cfg.BERT_TRAINER_CONFIG)


###########################
# ~ TRAIN & EVAL LOOP ~
###########################


def train_eval(train, test):
    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model,
        training_args,
        train_dataset=train,
        eval_dataset=test,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # TODO : allow to evaluate already trained models instead of training from scratch
    # at every run.
    trainer.train()
    eval = trainer.evaluate()
    return eval


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    label_list = list(label2id.keys())

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    metric = load_metric("seqeval")
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


##########################
#     ~ ENTRY POINT ~
##########################

if __name__ == "__main__":
    dirs = os.listdir(cfg.DATA_DIR)
    ds_paths = [
        os.path.join(cfg.DATA_DIR, dsname)
        for dsname in dirs
        if dsname.startswith("huggingface")
    ]
    ds = [load_from_disk(ds_loc) for ds_loc in ds_paths]

    metrics_data = []
    for data in ds:
        # TODO Control the train set size here
        perfs = train_eval(data["train"], data["test"])
        metrics_data.append(perfs)

    with open(cfg.DATA_DIR / "ner_bert_metrics.json", "w", encoding="utf-8") as o:
        json.dump(metrics_data, o)
