"""
This script contains a class to create sentences 
from samples using sentence numbers
"""

import random
import numpy as np
import torch
import os

from typing import Callable
from loguru import logger


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


def match_kwargs(function: Callable, **kwargs) -> dict:
    """Matches Arguments with Function

    Match keywords arguments with the arguments of a function.

    Args:
        function (function): Function to match arguments for.
        kwargs: keyword arguments to match against.

    Returns:
        dict: dictionary with matching arguments and their
        respective values.

    """
    arg_count = function.__code__.co_argcount
    args = function.__code__.co_varnames[:arg_count]

    args_dict = {}
    for k, v in kwargs.items():
        if k in args:
            args_dict[k] = v

    return args_dict


def enforce_reproducibility(seed=42) -> None:
    """Enforce Reproducibity

    Enforces reproducibility of models to the furthest 
    possible extent. This is done by setting fixed seeds for
    random number generation etcetera. 

    For atomic operations there is currently no simple way to
    enforce determinism, as the order of parallel operations
    is not known.

    Args:
        seed (int, optional): Fixed seed. Defaults to 42.  
    """
    # Sets seed manually for both CPU and CUDA
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # System based
    random.seed(seed)
    np.random.seed(seed)


def write_accuracy_file(model_dir, results):
    """This function will write the kfold accuracy file

    Args:
        model_dir (str): Output model directory to store results
        results (List[float]): A list of accuracy scores
    """
    with open(os.path.join(model_dir, "k-fold-accuracy-scores.txt"), "w") as wf:
        wf.write("K-Fold Accuracy Scores\n")
        for i in range(len(results)):
            wf.write(f"Step {i+1}: {results[i]} \n")

        wf.write("\n")
        wf.write(f"Mean-Accuracy: {sum(results) / len(results)}")

    logger.debug(f"Mean-Accuracy: {sum(results) / len(results)}")
    logger.success("Accuracy file stored!")


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    c = list(zip(a, b))
    random.shuffle(c)
    a_, b_ = zip(*c)
    return a_, b_


def flatten(l: list):
    """Flattens list"""
    return [item for sublist in l for item in sublist]


def sigmoid_transform(x):
    prob = 1/(1 + np.exp(-x))
    return prob

def check_dir(root_dir, name):
    new_dir = os.path.join(root_dir, name)
    if(not os.path.exists(new_dir)):
        logger.warning("Directory not found: {new_dir}".format(
            new_dir=new_dir))
        os.makedirs(new_dir)
        logger.success("Directory created: {new_dir}".format(
            new_dir=new_dir))
    return new_dir
