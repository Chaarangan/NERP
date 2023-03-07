"""
This script will compute predictions for a single text input as well as CSV file input
"""

import pandas as pd
import os

from typing import List
from loguru import logger

from .trainer import Trainer
from .utils import SentenceGetter


def load_model(torch_args, data_args, model_args, training_args, inference_args):
    """This function will load the trained model with tokenizer if exists

    Returns:
        object: compiled model
    """
    # compile model
    model = Trainer(torch_args, data_args, model_args, training_args, inference_args, inference_args.pretrained)

    # getting inference vars
    assert os.path.isfile(
        inference_args.model_path), f'File {inference_args.model_path} does not exist.'

    if(inference_args.tokenizer_path != None):
        model.load_network_from_file(
            model_path=inference_args.model_path, tokenizer_path=inference_args.tokenizer_path)
    else:
        model.load_network_from_file(model_path=inference_args.model_path)
    logger.success("Model weights loaded!")
    return model


def predict_bulk(model, in_file_path, out_file_path):
    """This function will make predictions on the CSV input file

    Args:
        model (object): Compiled model from load_model function
        in_file_path (str): Input csv file path
        out_file_path (str): Output csv file path
    """
    assert os.path.isfile(
        in_file_path), f'File {in_file_path} does not exist.'

    if not out_file_path.endswith(".csv"):
        raise TypeError("Check output file path (out_file_path)!")
    
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
        logger.info(
            "Prediction on sentence no: {no}".format(no=i))
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
    logger.success("Predictions stored!")


def inference_pipeline(is_bulk, text, torch_args, data_args, model_args, training_args, inference_args):

    model = load_model(torch_args, data_args, model_args,
                       training_args, inference_args)

    if(is_bulk):
        logger.info("Bulk Mode!")
        predict_bulk(model, inference_args.in_file_path, inference_args.out_file_path)

        logger.success("Predictions are stored successfully!")

    else:
        output = model.predict_text(text)
        logger.success("Predicted successfully! - " + str(output))
        return output
