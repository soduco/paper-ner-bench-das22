import json, argparse, nltk
import config as cfg
import numpy as np
from pathlib import Path
from xml.dom.minidom import parseString
from datasets import Dataset, load_from_disk
from datasets import load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification
from transformers import TrainingArguments, Trainer, DataCollatorForTokenClassification

logger = cfg.logger

# =============================================================================
# region ~ Model parameters

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

CAMEMBERT_MODEL = "Jean-Baptiste/camembert-ner"

CAMEMBERT_TRAINER_CONFIG = {
    "evaluation_strategy": "epoch",
    "learning_rate": 1e-4,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "num_train_epochs": 3,
    "weight_decay": 1e-5,
}

tokenizer = AutoTokenizer.from_pretrained(CAMEMBERT_MODEL)
training_args = TrainingArguments("berties", **CAMEMBERT_TRAINER_CONFIG)

# endregion

# =============================================================================
# region ~ Train & evaluation
def train_eval(train, dev, test):

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model,  # Implicit
        training_args,
        train_dataset=train,
        eval_dataset=dev,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    return trainer.evaluate(test), trainer.evaluate()


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    label_list = list(label_id_map.keys())

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


# endregion

# =============================================================================
# region ~ Main entry point & CLI


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str, help="Path to a DataDict")
    parser.add_argument("metrics", type=str, help="Metrics file, without extention.")
    # parser.add_argument(
    #     "--train_id", type=str, help="Trainset identifier for exp. 1.", required=False
    # )
    parser.add_argument(
        "--model",
        type=str,
        help=f"Name or path to a model. Default is {CAMEMBERT_MODEL}",
        required=False,
        default=CAMEMBERT_MODEL,
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    assert args.data and args.metrics

    if cfg.DEBUG:
        logger.info(f"Model {args.model}")
        logger.info(f"Running on datasets in {args.data}")
        logger.info(f"Metrics will be saved in {args.metrics}")

    model = AutoModelForTokenClassification.from_pretrained(
        args.model,
        num_labels=len(label_id_map),
        ignore_mismatched_sizes=True,
        id2label={v: k for k, v in label_id_map.items()},
        label2id=label_id_map,
    )

    # Train & evaluate
    datasets = load_from_disk(args.data)
    metrics = train_eval(**datasets)

    with open(f"{args.metrics}_test.json", "w", encoding="utf-8") as o:
        json.dump(metrics[0], o)

    with open(f"{args.metrics}_dev.json", "w", encoding="utf-8") as o:
        json.dump(metrics[1], o)

# endregion

# =============================================================================
# region ~ Data conversion utils for Hugginface


def create_huggingface_dataset(entries):
    # Creates a Huggingface Dataset from a set of NER-XML entries
    tokenized_entries = [word_tokens_from_xml(entry) for entry in entries]
    word_tokens, labels = zip(*tokenized_entries)
    dataset = Dataset.from_dict({"tokens": word_tokens, "ner_tags": labels})
    return dataset.map(assign_labels_to_bert_tokens, batched=True)


def assign_labels_to_bert_tokens(examples):
    bert_tokens = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    prev_word_id = None
    labels = []
    for id, label in enumerate(examples["ner_tags"]):
        labels_ids = []
        for word_id in bert_tokens.word_ids(batch_index=id):
            # Set the label only on the first token of each word
            if word_id in [None, prev_word_id]:
                labels_ids.append(-100)
            else:
                label_id = label_id_map[label[word_id]]
                labels_ids.append(label_id)
            prev_word_id = word_id

        labels.append(labels_ids)

    bert_tokens["labels"] = labels
    return bert_tokens


# convenient word tokenizer to create IOB-like data for the BERT models
nltk.download("punkt")


def word_tokens_from_xml(entry):
    w_tokens = []
    labels = []

    entry_xml = f"<x>{entry}</x>"
    x = parseString(entry_xml).getElementsByTagName("x")[0]

    for el in x.childNodes:
        if el.nodeName == "#text":
            cat = "O"
            txt = el.nodeValue
        else:
            cat = f"I-{el.nodeName}"
            txt = el.childNodes[0].nodeValue

        words = nltk.word_tokenize(txt, language="fr", preserve_line=True)
        w_tokens += words
        labels += [cat] * len(words)

    return w_tokens, labels


# endregion
