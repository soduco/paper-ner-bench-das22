import spacy
import csv
import pickle
import nltk
import config as cfg
from datasets.dataset_dict import DatasetDict
#from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from spacy.tokens import DocBin, Doc
from xml.dom.minidom import parseString
from spacy.training import biluo_tags_to_offsets
from bert import tokenizer, label_id_map
from datasets import Dataset


##########################
#     ~ HUGGINGFACE ~
##########################


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


##########################
#      ~ SPACY ~
##########################


nlp = spacy.blank("fr")


def create_spacy_dataset(entries):
    db = DocBin()

    for entry in entries:
        # Makes the entry a valid XML string, then parses it to a DOM.
        entry_xml = f"<x>{entry}</x>"
        x = parseString(entry_xml).getElementsByTagName("x")[0]

        tags = []
        tokens = []
        text = ""
        for el in x.childNodes:
            el_is_txt = el.nodeName == "#text"

            span = el.nodeValue if el_is_txt else el.childNodes[0].nodeValue
            text += span
            span = (
                span.strip()
            )  # Remove leading whitespaces or they will be kept as tokens

            inner_doc = nlp.make_doc(span)
            nertag = el.nodeName.upper()

            # BILUO as pivot format
            inner_tags = []
            if not el_is_txt:
                if len(inner_doc) == 1:
                    inner_tags = [f"U-{nertag}"]
                elif len(inner_doc) > 1:
                    inner_tags = (
                        [f"B-{nertag}"]
                        + [f"I-{nertag}"] * max(len(inner_doc) - 2, 0)
                        + [f"L-{nertag}"]
                    )
                else:
                    raise ValueError(f"Empty inner doc in {span}: {entry}")
                tags.extend(inner_tags)
            else:
                tags.extend(["O"] * len(inner_doc))

            tokens.extend([t.text for t in inner_doc])

        words, spaces = spacy.util.get_words_and_spaces(tokens, text)
        doc = Doc(nlp.vocab, words=words, spaces=spaces)

        offsets = biluo_tags_to_offsets(doc, tags)
        doc.ents = [doc.char_span(start, end, label=lbl) for start, end, lbl in offsets]
        db.add(doc)

    return db


################################
#    ~ DATASETS CREATION ~
################################


with open(cfg.INPUT_DATASET, encoding="utf-8") as annot:
    full_dataset = [_[0] for _ in csv.reader(annot)]

if not full_dataset:
    exit(1)

# Save for later use
# folds = KFold(n_splits=cfg.N_SPLITS, shuffle=True, random_state=cfg.RANDOM_STATE).split(
#     full_dataset
# )
# splits = [
#     [[full_dataset[i] for i in fold[0]], [full_dataset[i] for i in fold[1]]]
#     for fold in folds
# ]

# 90% train, 10% dev + validation
train_dev_valid = train_test_split(full_dataset, test_size=0.1, random_state=cfg.RANDOM_STATE)
dev_valid = train_test_split(train_dev_valid[1], test_size=0.5, random_state=cfg.RANDOM_STATE)

train = train_dev_valid[0]
dev = dev_valid[0]
valid = dev_valid[1]

print("train:",len(train),"dev:",len(dev),"valid",len(valid))
# Export splits as BERT datasets

bert_train = train + dev

bert_datasets = DatasetDict(
    {
        "train": create_huggingface_dataset(bert_train),
        "test": create_huggingface_dataset(valid),
    }
)

odir = cfg.DATA_DIR / f"huggingface"
bert_datasets.save_to_disk(odir)


# Export splits as SPACY datasets

train_bdata = create_spacy_dataset(train).to_bytes()
with open(cfg.DATA_DIR / f"spacy_train.spacy","wb") as tf:
    tf.write(train_bdata)

dev_bdata = create_spacy_dataset(dev).to_bytes()
with open(cfg.DATA_DIR / f"spacy_dev.spacy","wb") as tf:
    tf.write(dev_bdata)

valid_bdata = create_spacy_dataset(valid).to_bytes()
with open(cfg.DATA_DIR / f"spacy_valid.spacy","wb") as tf:
    tf.write(valid_bdata)
