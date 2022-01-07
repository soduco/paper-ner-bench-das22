import csv
import logging
import os
import config as cfg
from sklearn.model_selection import train_test_split
from datasets.dataset_dict import DatasetDict
from cnn import create_spacy_dataset
from bert import create_huggingface_dataset

# Helper functions
def unwrap(list_of_tuples2):
    return tuple(zip(*list_of_tuples2))


# Loads the gold dataset
with open(cfg.GOLD, encoding="utf-8") as annot:
    gold_dataset = [(_[0], _[1]) for _ in csv.reader(annot)]

logging.info(f"Gold dataset: {len(gold_dataset)} records")

# Generate Train (80%), Validation|Dev (10%) and Test (10%) datasets
# with stratified sampling on the directories names.

# Get train and dev+test
_, classes = unwrap(gold_dataset)
train_devtest = train_test_split(
    gold_dataset, train_size=0.8, shuffle=True, random_state=cfg.SEED, stratify=classes
)
train = train_devtest[0]

# Split dev+test in half to get dev and test
devtest = train_devtest[1]
_, classes = unwrap(devtest)
dev_test = train_test_split(
    devtest, train_size=0.5, shuffle=True, random_state=cfg.SEED, stratify=classes
)
dev = dev_test[0]
test = dev_test[1]

logging.info(
    f"Train ({len(train)} records), Dev ({len(dev)} records), Test ({len(test)} records)"
)

###
# Generate smaller train sets for experiment 1
###

exp_1_trainsets = [train]
k = len(train)
while k > cfg.MIN_TRAINSET_SIZE:
    try:
        current = exp_1_trainsets[-1]
        _, classes = unwrap(current)
        smaller, rest = train_test_split(
            current,
            train_size=0.5,
            shuffle=True,
            random_state=cfg.SEED,
            stratify=classes,
        )
        exp_1_trainsets.append(smaller)
        current_train = rest
        k = len(rest)
    except ValueError as e:
        # Stop now if we encounter the error "The least populated class in y has only 1 member".
        break

logging.info(
    f"Experiment 1: created {len(exp_1_trainsets)} training sets of sizes {[len(_) for _ in exp_1_trainsets]}"
)


###
# Export all datasets in Spacy and Huggingface formats
###


def export(path, ds_names, addendum=None):
    fname = lambda elemts: "_".join(filter(None, elemts))
    # Spacy
    for ds, name in ds_names:
        fullpath = path / fname([name, addendum])
        fullpath = fullpath.with_suffix(".spacy")

        with open(fullpath, "wb") as tf:
            bdata = create_spacy_dataset(ds).to_bytes()
            tf.write(bdata)

    # Huggingface
    bert_ds = DatasetDict(
        {
            "train": create_huggingface_dataset(datasets[0]),
            "dev": create_huggingface_dataset(datasets[1]),
            "test": create_huggingface_dataset(datasets[2]),
        }
    )
    fullpath = path / fname(["huggingface", addendum])
    bert_ds.save_to_disk(fullpath)


# Export now
datasets = [unwrap(train)[0], unwrap(dev)[0], unwrap(test)[0]]
names = ["train", "dev", "test"]

# Export global Train, Dev and Test
export(cfg.DATA_DIR, zip(datasets, names))

# Export datasets for experiment 1
odir = cfg.DATA_DIR / "experiment_1"
os.makedirs(odir, exist_ok=True)

for ts in exp_1_trainsets:
    datasets = [ts, unwrap(dev)[0], unwrap(test)[0]]
    addendum = str(len(ts))
    export(odir, zip(datasets, names), addendum)

# # FIXME Debug
# from spacy import displacy
# displacy.serve(datasets[0].get_docs(nlp.vocab), style="ent")
