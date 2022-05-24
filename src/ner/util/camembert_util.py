import numpy as np
import nltk
from xml.dom.minidom import parseString
from datasets import load_metric, Dataset
from util.config import logger
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)


LABELS_ID = {
    "O": 0,          # Not an entity
    "I-LOC": 1,      # An address, like "rue du Faub. St.-Antoine"
    "I-PER": 2,      # A person, like "Baboulinet (Vincent)"
    "I-MISC": 3,     # Not used but present in the base model
    "I-ORG": 4,      # Not used but present in the base model
    "I-CARDINAL": 5,  # Not used but present in the base model
    "I-ACT": 6,      # An activity, like "plombier-devin"
    "I-TITRE": 7,    # A person's encoded title, like "O. ::LH::" for "Officier de la Légion d'Honneur"
    # A feature type, like "fabrique" or "dépot" in front of addresses.
    "I-FT": 8,
}


# Entry point
def init_model(model_name, training_config):
    logger.info(f"Model {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    #output_path = save_model_path or "/tmp/bert-model"
    training_args = TrainingArguments(**training_config)

    # Load the model
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS_ID),
        ignore_mismatched_sizes=True,
        id2label={v: k for k, v in LABELS_ID.items()},
        label2id=LABELS_ID
    )
    return model, tokenizer, training_args

# Main loop


def train_eval_loop(model, training_args, tokenizer, train, dev, test, patience=3):
    data_collator = DataCollatorForTokenClassification(tokenizer)
    trainer = Trainer(
        model,
        training_args,
        train_dataset=train,
        eval_dataset=dev,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=patience)
        ],
    )
    trainer.train()
    return trainer.evaluate(test), trainer.evaluate()

# Metrics


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    label_list = list(LABELS_ID.keys())

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    metric = load_metric("seqeval")
    results = metric.compute(
        predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


# =============================================================================
# region ~ Data conversion utils for Hugginface

_convert_tokenizer = AutoTokenizer.from_pretrained(
    "Jean-Baptiste/camembert-ner")


def create_huggingface_dataset(entries):
    # Creates a Huggingface Dataset from a set of NER-XML entries
    tokenized_entries = [word_tokens_from_xml(entry) for entry in entries]
    word_tokens, labels = zip(*tokenized_entries)
    ds = Dataset.from_dict({"tokens": word_tokens, "ner_tags": labels})
    return ds.map(assign_labels_to_bert_tokens, batched=True)


def assign_labels_to_bert_tokens(examples):
    bert_tokens = _convert_tokenizer(
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
                label_id = LABELS_ID[label[word_id]]
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
