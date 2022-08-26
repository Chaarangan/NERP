'''
File: NERP/compile_model.py
Project: NERP
Created Date: Tuesday, May 24th 2022
Author: Charangan Vasantharajan
-----
Last Modified: Friday, Aug 26th 2022
Modified By: Charangan Vasantharajan
-----
Copyright (c) 2022
------------------------------------
This script will prepare training and validation datasets and compile the model
'''

from NERDA_framework.models import NERDA
from NERP.prepare_data import prepare_data

def compile_model(archi, device, training, validation, tag_scheme, o_tag_cr, hyperparameters, tokenizer_parameters, max_len, dropout, pretrained):
    """This function will prepare training and validation datasets and compile the model

    Args:
        archi (str): the desired architecture for the model
        device (str): the desired device to use for computation
        training (str): Training dictionary
        validation (str): Validation dictionary
        tag_scheme (List[str]): All available NER tags for the given data set EXCLUDING the special outside tag, that is handled separately
        o_tag_cr (bool): To include O tag in the classification report
        hyperparameters (dict): Hyperparameters for the model
        tokenizer_parameters (dict): Parameters for the tokenizer
        max_len (int): The maximum sentence length
        dropout (float): dropout probability
        pretrained (str): which pretrained 'huggingface' transformer to use

    Returns:
        object: compiled model
    """

    model = NERDA(
        archi=archi,
        device=device,
        dataset_training=training,
        dataset_validation=validation,
        tag_scheme=tag_scheme,
        tag_outside='O',
        o_tag_cr=o_tag_cr,
        transformer=pretrained,
        dropout=dropout,
        max_len=max_len,
        hyperparameters=hyperparameters,
        tokenizer_parameters=tokenizer_parameters
    )
    print("Model compiled with {archi} architecture!".format(archi=archi))
    return model
