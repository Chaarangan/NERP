'''
File: NERP/prepare_data.py
Project: NERP
Created Date: Tuesday, May 24th 2022
Author: Charangan Vasantharajan
-----
Last Modified: Friday, Aug 26th 2022
Modified By: Charangan Vasantharajan
-----
Copyright (c) 2022
------------------------------------
This script will prepare the sentences and entities from the input BIO format
'''

import os
import pandas as pd
from NERP.utils import SentenceGetter
from sklearn.model_selection import train_test_split

def prepare_data(limit: int = 0, file_path: str = None):
    """This function will prepare the sentences and entities from the input BIO format 

    Args:
        limit (int, optional): Limit the number of observations to be returned from a given split. Defaults to 0, which implies that the entire data split is returned.
        file_path (str, optional): file where data is cached. Defaults to None.

    Returns:
        dict: sentences and corresponding entities
    """
    file_path = os.path.join(file_path)
    assert os.path.isfile(file_path), f'File {file_path} does not exist.'

    data = pd.read_csv(file_path)
    data = data.fillna(method="ffill")

    getter = SentenceGetter(data)
    sentences = [[word[0] for word in sentence] for sentence in getter.sentences]
    entities = [[s[1] for s in sentence] for sentence in getter.sentences]

    if limit != 0:
        assert isinstance(limit, int), f"Limit shoud be a int!"
        sentences = sentences[:limit]
        entities = entities[:limit]
        print("Dataset is limited to {}".format(limit))

    assert len(sentences) == len(
        entities), f"Sentences and entities are having different length."

    return {'sentences': sentences, 'tags': entities}


def prepare_train_valid_data(train_data, valid_data, limit, test_size):
    """This function will create training and validation dictionaries from train and valid csv files

    Args:
        train_data (str): Train csv file  path
        valid_data (str): Valid csv file
        limit (int): Limit the number of observations to be returned from a given split. Defaults to 0, which implies that the entire data split is returned.
        test_size (float): train/valid split ratio if valid data not exists

    Returns:
        dict: Two dictionaries (training and validation)
    """
    if (valid_data == None):
        print("Valid data is None and created from train data!")
        data = prepare_data(limit, train_data)
        train_sentences, val_sentences, train_entities, val_entities = train_test_split(
            data["sentences"], data["tags"], test_size=test_size
        )
        training = {"sentences": train_sentences, "tags": train_entities}
        validation = {"sentences": val_sentences, "tags": val_entities}

        print("Training: ({a}, {b})".format(
            a=str(len(training["sentences"])), b=str(len(training["tags"]))))
        print("Validation: ({a}, {b}".format(
            a=str(len(validation["sentences"])), b=str(len(validation["tags"]))))

    else:
        print("Valid data exists!")
        training = prepare_data(limit, train_data)
        validation = prepare_data(limit, valid_data)

        print("Training: ({a}, {b})".format(
            a=str(len(training["sentences"])), b=str(len(training["tags"]))))
        print("Validation: ({a}, {b}".format(
            a=str(len(validation["sentences"])), b=str(len(validation["tags"]))))

    print("Train and Valid datasets are prepared!")

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
    print("Test: ({a}, {b})".format(
        a=str(len(test["sentences"])), b=str(len(test["tags"]))))
    print("Test dataset is prepared!")

    return test


def prepare_kfold_data(train_data, valid_data, test_data, limit, test_on_original):
    """This function will prepare training dictionary for kfold

    Args:
        train_data (str): Train csv file  path
        valid_data (str): Valid csv file  path
        test_data (str): Test csv file  path
        limit (int): Limit the number of observations to be returned from a given split. Defaults to 0, which implies that the entire data split is returned.
        test_on_original (bool): True, if you need to test on the original test set for each iteration

    Returns:
        dict: a dictioanry of sentences and tags
    """
    sentences = []
    tags = []

    train_data = prepare_data(limit, train_data)
    sentences += train_data["sentences"]
    tags += train_data["tags"]

    if (valid_data != None):
        valid_data = prepare_data(limit, valid_data)
        sentences += valid_data["sentences"]
        tags += valid_data["tags"]

    test_data = prepare_data(limit, test_data)

    if(not test_on_original):
        print("Test data is combined with training set!")
        sentences += test_data["sentences"]
        tags += test_data["tags"]

    else:
        print("Test data is ignored from training set!")

    return {"sentences": sentences, "tags": tags}


def prepare_kfold_train_valid_data(training, test_size):
    """This function will create training and validation dictionaries from train and valid csv files for kfold

    Args:
        training (dict): A dictioanry of sentences and tags from training set
        test_size (float): train/valid split ratio if valid data not exists

    Returns:
        dict: Two dictionaries (training and validation)
    """
    train_sentences, val_sentences, train_entities, val_entities = train_test_split(
        training["sentences"], training["tags"], test_size=test_size
    )
    training = {"sentences": train_sentences, "tags": train_entities}
    validation = {"sentences": val_sentences, "tags": val_entities}

    print("Training: ({a}, {b})".format(
            a=str(len(training["sentences"])), b=str(len(training["tags"]))))
    print("Validation: ({a}, {b}".format(
            a=str(len(validation["sentences"])), b=str(len(validation["tags"]))))


    print("Train and Valid datasets are prepared!")

    return training, validation
