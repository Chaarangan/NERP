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