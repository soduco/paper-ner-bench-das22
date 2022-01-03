import nltk
import csv, json
from xml.dom.minidom import parseString
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification
from transformers import TrainingArguments, Trainer, DataCollatorForTokenClassification
from datasets import load_metric
from sklearn.model_selection import KFold
import numpy as np

# DATASET

# A text file containing one ner-xml string per line
data_path = "../data/ner_annotated_full.csv"

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


# MODEL
base_model = "Jean-Baptiste/camembert-ner"

model = AutoModelForTokenClassification.from_pretrained(
    base_model,
    num_labels=len(label2id),
    ignore_mismatched_sizes=True,
    id2label=id2label,
    label2id=label2id,
)

tokenizer = AutoTokenizer.from_pretrained(base_model)

# TRAINING CONFIGURATION

k_folds = 2

training_args = TrainingArguments(
    "berties",
    evaluation_strategy="epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=1e-5,
)

metrics_output_path = "../data/ner_bert_metrics.json"


###########################
# ~ TRAIN & EVAL LOOP ~
###########################


def train_eval(entries):
    data_collator = DataCollatorForTokenClassification(tokenizer)

    folds = KFold(n_splits=k_folds, shuffle=True, random_state=42).split(entries)

    evaluation_results = []
    for fold in folds:
        train_data = [entries[i] for i in fold[0]]
        test_data = [entries[i] for i in fold[1]]

        train_set = create_bert_dataset(train_data)
        test_set = create_bert_dataset(test_data)

        trainer = Trainer(
            model,
            training_args,
            train_dataset=train_set,
            eval_dataset=test_set,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        # TODO : allow to evaluate already trained models instead of training from scratch
        # at every run.
        trainer.train()
        eval = trainer.evaluate()
        evaluation_results.append(eval)

    return evaluation_results


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
#   ~ DATA PREPARATION ~
##########################


def create_bert_dataset(entries):
    #
    # Creates a Huggingface Dataset from a set of NER-XML entries

    tokenized_entries = [word_tokens_from_xml(entry) for entry in entries]
    word_tokens, labels = zip(*tokenized_entries)
    dataset_dict = {"tokens": word_tokens, "ner_tags": labels}
    dataset = Dataset.from_dict(dataset_dict)

    return dataset.map(assign_labels_to_bert_tokens, batched=True)


# Tokenizes for BERT and assign labels to each token
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


nltk.download("punkt")


def word_tokens_from_xml(entry):
    w_tokens = []
    labels = []

    #
    # Makes the entry a valid XML string, then parses it to a DOM.
    entry_xml = f"<x>{entry}</x>"
    x = parseString(entry_xml).getElementsByTagName("x")[0]

    #
    # Separates words and assigns an IOB class to each (B-* are ignored).
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


##########################
#     ~ ENTRY POINT ~
##########################

if __name__ == "__main__":
    with open(data_path, encoding="utf-8") as annot:
        full_dataset = [_[0] for _ in csv.reader(annot)]
        full_dataset = full_dataset[:3000]
        evaluation = train_eval(full_dataset)

    with open("data/ner_bert_metrics.json", "w", encoding="utf-8") as o:
        json.dump(evaluation, o)
