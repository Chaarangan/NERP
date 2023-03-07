"""
This script will prepare the sentences and entities from the input BIO format
"""

import os
import pandas as pd
import csv
import random
import logging

from sklearn.model_selection import train_test_split
from loguru import logger

from .utils import unison_shuffled_copies, SentenceGetter


def prepare_data(limit: int = 0, file_path: str = None, sep=',', quoting=3, shuffle=False, seed=42):
    """This function will prepare the sentences and entities from the input BIO format

    Args:
        limit (int, optional): Limit the number of observations to be returned from a given split. Defaults to 0, which implies that the entire data split is returned.
        file_path (str, optional): file where data is cached. Defaults to None.
        sep (str, optional): Delimiter to use
        quoting (str, optional): Control field quoting behavior per csv.QUOTE_* constants.
        shuffle (bool, optional): Shuffle the entire dataset before training
        seed (int, optional): Random state value for a particular experiment

    Returns:
        dict: sentences and corresponding entities
    """
    file_path = os.path.join(file_path)
    assert os.path.isfile(file_path), f'File {file_path} does not exist.'

    data = None
    if quoting:
        data = pd.read_csv(file_path, sep=sep, na_filter=False)
    else:
        data = pd.read_csv(file_path, sep=sep,
                           quoting=quoting, na_filter=False)

    getter = SentenceGetter(data)
    sentences = [[word[0] for word in sentence]
                 for sentence in getter.sentences]
    entities = [[s[1] for s in sentence] for sentence in getter.sentences]

    if limit != 0:
        assert isinstance(limit, int), f"Limit shoud be a int!"
        sentences = sentences[:limit]
        entities = entities[:limit]
        logger.debug("Dataset is limited to {}".format(limit))

    assert len(sentences) == len(
        entities), f"Sentences and entities are having different length."

    if shuffle:
        random.seed(seed)
        sentences, entities = unison_shuffled_copies(sentences, entities)

    return {'sentences': sentences, 'tags': entities}


def prepare_train_valid_data(train_data, valid_data, torch_args, data_args):
    """This function will create training and validation dictionaries from train and valid csv files

    Args:
        train_data (str): Train csv file  path
        valid_data (str): Valid csv file

    Returns:
        dict: Two dictionaries (training and validation)
    """
    if (valid_data == None):
        logger.info("Valid data is None and created from train data!")
        data = prepare_data(data_args.limit, train_data, sep=data_args.sep, quoting=data_args.quoting, shuffle=data_args.shuffle, seed=torch_args.seed)
        train_sentences, val_sentences, train_entities, val_entities = train_test_split(
            data["sentences"], data["tags"], test_size=data_args.train_test_split, random_state=torch_args.seed
        )
        training = {"sentences": train_sentences, "tags": train_entities}
        validation = {"sentences": val_sentences, "tags": val_entities}

        logger.info("Training: ({a}, {b})".format(
            a=str(len(training["sentences"])), b=str(len(training["tags"]))))
        logger.info("Validation: ({a}, {b}".format(
            a=str(len(validation["sentences"])), b=str(len(validation["tags"]))))

    else:
        logger.info("Valid data exists!")
        training = prepare_data(data_args.limit, train_data, sep=data_args.sep, quoting=data_args.quoting, shuffle=data_args.shuffle, seed=torch_args.seed)
        validation = prepare_data(data_args.limit, valid_data)

        logger.info("Training: ({a}, {b})".format(
            a=str(len(training["sentences"])), b=str(len(training["tags"]))))
        logger.info("Validation: ({a}, {b}".format(
            a=str(len(validation["sentences"])), b=str(len(validation["tags"]))))

    logger.info("Train and Valid datasets are prepared!")

    return training, validation


def prepare_test_data(test_data, limit):
    """The function will create a dictionary of sentences and tags for test set

    Args:
        test_data (str): Test csv file
        limit (int): Limit the number of observations to be returned from a given split. Defaults to 0, which implies that the entire data split is returned.

    Returns:
        dict: a dictioanry of sentences and tags
    """
    test = prepare_data(limit, test_data)
    logger.info("Test: ({a}, {b})".format(
        a=str(len(test["sentences"])), b=str(len(test["tags"]))))
    logger.info("Test dataset is prepared!")

    return test


def prepare_kfold_data(seed, train_data, valid_data, test_data, limit, sep, quoting, shuffle, test_on_original):
    """This function will prepare training dictionary for kfold

    Args:
        seed (int, optional): Random state value for a particular experiment
        train_data (str): Train csv file  path
        valid_data (str): Valid csv file  path
        test_data (str): Test csv file  path
        limit (int): Limit the number of observations to be returned from a given split. Defaults to 0, which implies that the entire data split is returned.
        sep (str, optional): Delimiter to use
        quoting (str, optional): Control field quoting behavior per csv.QUOTE_* constants.
        shuffle (bool, optional): Shuffle the entire dataset before training
        test_on_original (bool): True, if you need to test on the original test set for each iteration

    Returns:
        dict: a dictionary of sentences and tags
    """
    sentences = []
    tags = []

    train_data = prepare_data(limit, train_data, sep=sep, quoting=quoting, shuffle=shuffle, seed=seed)
    sentences += train_data["sentences"]
    tags += train_data["tags"]

    if (valid_data != None):
        valid_data = prepare_data(limit, valid_data)
        sentences += valid_data["sentences"]
        tags += valid_data["tags"]

    if(not test_on_original):
        logger.info("Test data is combined with training set!")
        for t in test_data:
            test_data = prepare_test_data(t, limit)
            sentences += test_data["sentences"]
            tags += test_data["tags"]

    else:
        logger.info("Test data is ignored from training set!")

    return {"sentences": sentences, "tags": tags}


def prepare_kfold_train_valid_data(training, test_size, seed):
    """This function will create training and validation dictionaries from train and valid csv files for kfold

    Args:
        training (dict): A dictioanry of sentences and tags from training set
        test_size (float): train/valid split ratio if valid data not exists
        seed (int, optional): Random state value for a particular experiment

    Returns:
        dict: Two dictionaries (training and validation)
    """
    train_sentences, val_sentences, train_entities, val_entities = train_test_split(
        training["sentences"], training["tags"], test_size=test_size, random_state=seed
    )
    training = {"sentences": train_sentences, "tags": train_entities}
    validation = {"sentences": val_sentences, "tags": val_entities}

    logger.info("Training: ({a}, {b})".format(
        a=str(len(training["sentences"])), b=str(len(training["tags"]))))
    logger.info("Validation: ({a}, {b}".format(
        a=str(len(validation["sentences"])), b=str(len(validation["tags"]))))

    logger.success("Train and Valid datasets are prepared!")

    return training, validation
