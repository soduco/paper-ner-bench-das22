import csv
import numpy as np
import config as cfg
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import train_test_split
from datasets.dataset_dict import DatasetDict
from spacy_util import create_spacy_dataset
from camembert import create_huggingface_dataset

logger = cfg.logger

# =============================================================================
# region ~ Helpers


def unwrap(list_of_tuples2):
    return tuple(zip(*list_of_tuples2))


# endregion

# =============================================================================
# region ~ Sanity checks


def check(expected, actual, msg=""):
    assert actual == expected, msg


def check_experiment1_dev_test_sizes(dev, test):
    actual = len(dev)
    expected = 709
    check(
        expected,
        actual,
        f"Experiment 1: expected dev set of size {expected}, got {actual}.",
    )

    actual = len(test)
    expected = 1690
    check(
        expected,
        actual,
        f"Experiment 1: expected dev set of size {expected}, got {actual}.",
    )


def check_experiment1_training_subsets_sizes(trainsets):
    actual = len(trainsets)
    expected = 8
    check(
        expected,
        actual,
        f"Experiment 1: expected {expected} sub-training sets but got {actual}",
    )

    expected = [6373, 3186, 1593, 796, 398, 199, 99, 49, 24]
    for ix, actual in enumerate(trainsets):
        actual_len = len(actual)
        expected_len = expected[ix]
        check(
            expected_len,
            actual_len,
            f"Expected sub-training set of size {expected_len} but got {actual_len}",
        )


# endregion

# =============================================================================
# region ~ Main processing


def main():
    """Generate all datasets"""
    with open(cfg.GOLD, encoding="utf-8") as gf:
        gold_dataset = np.array([(_[0], _[1]) for _ in csv.reader(gf)])

    logger.info("GOLD: %d records" % len(gold_dataset))

    ##
    # Create the global Train & Dev
    # Stratified split 90/10%
    ##
    _, groups = unwrap(gold_dataset)
    train, dev = train_test_split(
        gold_dataset,
        train_size=0.9,
        shuffle=True,
        random_state=cfg.SEED,
        stratify=groups,
    )

    logger.info("Full train: %s, Full dev: %s" % (len(train), len(dev)))
    # zipped = zip([train, dev], ["train", "dev"])
    # odir = cfg.DATA_DIR / "datasets"
    # export(odir, zipped)

    ##
    # Create datasets for experiment 1
    ##
    odir = cfg.WORKDIR / "experiment_1"

    train, dev, test = make_train_dev_test(gold_dataset)
    exp1_trainsets = create_experiment_1_trainsets(train)

    # Sanity checks
    check_experiment1_dev_test_sizes(dev, test)
    check_experiment1_training_subsets_sizes(exp1_trainsets)

    for subtrain in exp1_trainsets:
        datasets = [subtrain, dev, test]
        addendum = str(len(subtrain))
        export(odir, zip(datasets, ["train", "dev", "test"]), addendum)


def make_train_dev_test(gold):
    """Splits the gold dataset into two subsets where entries from the same directory is garanteed
    to not be in the both sets.
    Subset 1 is the test dataset with approx. ~20% of all entries.
    Subset 2 is splitted into a train set (~70%) and a validation set (~8%).
    For the second split we use stratified sampling based on directories names so the initial proportion
    of entries from each directory is preserved accros subsets.
    """

    # Use GroupKFold to pick a few directories for testing .
    # In the current state of the gold dataset (08/01/2022),
    # entries from 3 dirctories will be selected:
    #   - Bottin1_1820 (267 entries)
    #   - Didot_1851a (1266 entries)
    #   - Duverneuil_et_La_Tynna_1806 (186 entries)

    _, groups = unwrap(gold)
    index_tmp, index_test = list(GroupKFold(n_splits=5).split(gold, groups=groups))[0]
    subset_tmp, test = gold[index_tmp], gold[index_test]

    # Split subset_tmp into train (~90%) and dev (~10%) stratified on directories names
    _, groups = unwrap(subset_tmp)
    train_dev = train_test_split(
        subset_tmp, train_size=0.9, shuffle=True, random_state=cfg.SEED, stratify=groups
    )
    train = train_dev[0]
    dev = train_dev[1]

    logger.info(
        "Train: %d entries, %.1f%%" % (len(train), 100 * len(train) / len(gold))
    )
    logger.info("Dev: %d entries, %.1f%%" % (len(dev), 100 * len(dev) / len(gold)))
    logger.info("Test: %d entries, %.1f%%" % (len(test), 100 * len(test) / len(gold)))

    return train, dev, test


def create_experiment_1_trainsets(gold_train):
    """Create iteratively smaller trainsets for experiment 1 by dividing the training set
    in half at each step.

    Args:
        gold_train ([type]): The full gold trainset

    Returns:
        [type]: [description]
    """
    exp_ts = [gold_train]
    k = len(gold_train)

    while True:
        try:
            current = exp_ts[-1]
            _, groups = unwrap(current)
            smaller, rest = train_test_split(
                current,
                train_size=0.5,
                shuffle=True,
                random_state=cfg.SEED,
                stratify=groups,
            )
            k = len(rest)
            if k < cfg.MIN_TRAINSET_SIZE:
                break
            exp_ts.append(smaller)

        except ValueError:
            # Stop now if we encounter the error "The least populated class in y has only 1 member".
            break

    logger.info(
        "Experiment 1: Generated %d sub-training sets of resp. sizes %s, %s"
        % (
            len(exp_ts),
            [len(s) for s in exp_ts],
            [f"{100*len(s)/len(gold_train):.1f}%" for s in exp_ts],
        )
    )

    return exp_ts


def export(output_dir, ds_with_names, addendum=None):
    """Export all datasets in Spacy and Huggingface native formats."""

    fname = lambda elemts: "_".join(filter(None, elemts))

    hf_dic = {}

    for ds, name in ds_with_names:

        data, _ = unwrap(ds)

        # Spacy
        spacy_file = output_dir / fname([name, addendum])
        spacy_file = spacy_file.with_suffix(".spacy")

        if cfg.DEBUG:
            np.savetxt(spacy_file.with_suffix(".debug"), ds, fmt='"%s","%s"')

        with open(spacy_file, "wb") as tf:
            bdata = create_spacy_dataset(data).to_bytes()
            tf.write(bdata)

        # Huggingface
        hf_dic[name] = create_huggingface_dataset(ds)

    bert_ds = DatasetDict(hf_dic)
    hf_file = output_dir / fname(["huggingface", addendum])
    bert_ds.save_to_disk(hf_file)


# endregion


# =============================================================================
# ENTRY POINT
if __name__ == "__main__":
    main()