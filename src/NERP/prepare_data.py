'''
File: NERP/prepare_data.py
Project: NERP
Created Date: Tuesday, May 24th 2022
Author: Charangan Vasantharajan
-----
Last Modified: Sunday, July 31st 2022
Modified By: Charangan Vasantharajan
-----
Copyright (c) 2022
------------------------------------
This script will prepare the sentences and entities from the input BIO format
'''

import os
import pandas as pd
from NERP.utils import SentenceGetter

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
