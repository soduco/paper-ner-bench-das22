import json
import config as cfg
import os
import numpy as np
import nltk
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification
from transformers import TrainingArguments, Trainer, DataCollatorForTokenClassification
from datasets import load_metric
from xml.dom.minidom import parseString

# Shared variables
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
tokenizer = AutoTokenizer.from_pretrained(cfg.BERT_BASE_MODEL)
training_args = TrainingArguments("berties", **cfg.BERT_TRAINER_CONFIG)

# Training & eval
def train_eval(train, valid):
    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model,
        training_args,
        train_dataset=train,
        eval_dataset=valid,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

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
#  ~ MAIN ENTRY POINT ~
##########################

if __name__ == "__main__":

    label2id = label_id_map
    id2label = {v: k for k, v in label2id.items()}

    model = AutoModelForTokenClassification.from_pretrained(
        cfg.BERT_BASE_MODEL,
        num_labels=len(label2id),
        ignore_mismatched_sizes=True,
        id2label=id2label,
        label2id=label2id,
    )

    # Loads datasets, then run train + eval
    dirs = os.listdir(cfg.DATA_DIR)
    ds_paths = [
        os.path.join(cfg.DATA_DIR, dsname)
        for dsname in dirs
        if dsname.startswith("huggingface")
    ]
    ds = [load_from_disk(ds_loc) for ds_loc in ds_paths]

    metrics_data = []
    for data in ds:
        perfs = train_eval(data["train"], data["valid"])
        metrics_data.append(perfs)

    with open(cfg.DATA_DIR / "ner_bert_metrics.json", "w", encoding="utf-8") as o:
        json.dump(metrics_data, o)


###########################
# ~ DATA CONVERSION UTILS ~
###########################


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
