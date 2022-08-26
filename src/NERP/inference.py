'''
File: NERP/inference.py
Project: NERP
Created Date: Tuesday, May 24th 2022
Author: Charangan Vasantharajan
-----
Last Modified: Friday, Aug 26th 2022
Modified By: Charangan Vasantharajan
-----
Copyright (c) 2022
------------------------------------
This script will compute predictions for a single text input as well as CSV file input
'''

from NERDA_framework.models import NERDA
from NERP.utils import SentenceGetter
import pandas as  pd
import os
from typing import List

def load_model(archi, device, tag_scheme, pretrained, max_len, model_path, tokenizer_path, hyperparameters, tokenizer_parameters):
    """This function will load the trained model with tokenizer if exists

    Args:
        archi (str): the desired architecture for the model
        device (str): the desired device to use for computation
        tag_scheme (List[str]): All available NER tags for the given data set EXCLUDING the special outside tag, that is handled separately
        pretrained (str): which pretrained 'huggingface' transformer to use
        max_len (int): The maximum sentence length
        model_path (str): Trained model path
        tokenizer_path (str): Existing tokenizer path if exist: otherwise it loads from huggingface.
        hyperparameters (dict): Hyperparameters for the model
        tokenizer_parameters (dict): Parameters for the tokenizer

    Returns:
        object: compiled model
    """
    # compile model
    model = NERDA(
        archi=archi,
        device=device,
        tag_scheme=tag_scheme,
        tag_outside='O',
        transformer=pretrained,
        max_len=max_len,
        hyperparameters=hyperparameters,
        tokenizer_parameters=tokenizer_parameters
    )

    # getting inference vars
    assert os.path.isfile(model_path), f'File {model_path} does not exist.'

    if(tokenizer_path != None):
        model.load_network_from_file(model_path=model_path, tokenizer_path=tokenizer_path)
    else:
        model.load_network_from_file(model_path=model_path)
    print("Model weights loaded!")
    return model

def predict_bulk(model, in_file_path, out_file_path):
    """This function will make predictions on the CSV input file

    Args:
        model (object): Compiled model from load_model function
        in_file_path (str): Input csv file path
        out_file_path (str): Output csv file path
    """
    data = pd.read_csv(in_file_path)
    data = data.fillna(method="ffill")
    getter = SentenceGetter(data)
    sentences = [[word[0] for word in sentence]
                 for sentence in getter.sentences]
    sentences = [" ".join(line) for line in sentences]

    sentence_no = []
    words = []
    tags = []
    i = 0
    for sentence in sentences:
        print(
            "Predicted on sentence no: {no} - {sentence}".format(no=i, sentence=sentence))
        results = model.predict_text(sentence)
        words += results[0][0]
        tags += results[1][0]

        for word in results[0][0]:
            sentence_no.append("Sentence: " + str(i))

        i += 1

    assert len(words) == len(
        tags), f'Words and Tags are not having equal dimensions'

    out_file_path = os.path.join(out_file_path)
    df = pd.DataFrame({"Sentence #": sentence_no,
                       "Word": words, "Tag": tags})
    df.to_csv(out_file_path)
    print("Predictions stored!")


def inference_pipeline(archi,
                       device, 
                       model_path,
                       tokenizer_path,
                       out_file_path, 
                       in_file_path,
                       pretrained: str = "roberta-base",                       
                       is_bulk: bool = False,                       
                       text: str = "Hello from NERP",
                       tag_scheme: List[str] = [
                           'B-PER',
                           'I-PER',
                           'B-ORG',
                           'I-ORG',
                           'B-LOC',
                           'I-LOC',
                           'B-MISC',
                           'I-MISC'
                       ],
                       hyperparameters: dict = {"train_batch_size": 64},
                       tokenizer_parameters: dict = {"do_lower_case": True},
                       max_len: int = 128):

    model = load_model(archi, device, tag_scheme, pretrained, max_len,
                       model_path, tokenizer_path, hyperparameters, tokenizer_parameters)

    if(is_bulk):
        print("Bulk Mode!")
        predict_bulk(model, in_file_path, out_file_path)

        return None, "Predictions are stored successfully!"

    else:
        return model.predict_text(text), "Predicted successfully!"
    
