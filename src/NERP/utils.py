'''
File: NERP/utils.py
Project: NERP
Created Date: Tuesday, May 24th 2022
Author: Charangan Vasantharajan
-----
Last Modified: Sunday, July 31st 2022
Modified By: Charangan Vasantharajan
-----
Copyright (c) 2022
------------------------------------
This script contains a class to create sentences from samples using sentence numbers
'''
from NERP.prepare_data import prepare_data
import pandas as pd

class SentenceGetter(object):
    """This class will group samples using its sentence number and make it as a sentence

    Args:
        object (df): dataframe contains NER samples with BIO tags and sentence numbers
    """

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False

        def agg_func(s): return [
            (w, t) for w, t in zip(s["Word"].values.tolist(), s["Tag"].values.tolist())
        ]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


def prepare_train_valid_data(train_data, valid_data, limit, test_size):
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
    test = prepare_data(limit, test_data)
    print("Test: ({a}, {b})".format(
        a=str(len(test["sentences"])), b=str(len(test["tags"]))))
    print("Test dataset is prepared!")
    
    return test


def prepare_kfold_data(train_data, valid_data, test_data, limit, test_on_original):
    sentences = []
    tags = []
    
    train_data = prepare_data(limit, train_data)
    sentences+=train_data["sentences"]
    tags+=train_data["tags"]
    
    if (valid_data != None):
        valid_data = prepare_data(limit, valid_data)
        sentences+=valid_data["sentences"]
        tags+=valid_data["tags"]
    
    test_data = prepare_data(limit, test_data)
    
    if(not test_on_original):
        print("Test data is combined with training set!")
        sentences+=test_data["sentences"]
        tags+=test_data["tags"]  

    else:
        print("Test data is ignored from training set!")
    
    return {"sentences": sentences, "tags": tags}



