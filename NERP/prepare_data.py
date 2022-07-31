"""
This section covers functionality for preparing Named Entity 
Recognition data sets.
Author: Charangan Vasantharajan
"""
import os
import pandas as pd
from .utils import SentenceGetter


def prepare_data(limit: int = None, file_path: str = None):
    """
    Args:
        limit (int, optional): Limit the number of observations to be 
            returned from a given split. Defaults to None, which implies 
            that the entire data split is returned.
        file_path (str, required): file where data is cached.

    Returns:
        list of sentences, and entities
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
