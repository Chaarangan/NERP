"""
This section covers functionality for training Named Entity Recognition models.
    - with k-fold
    - without k-fold
"""

import os
import pandas as pd
import torch
import csv

from tqdm import tqdm
from typing import List
from sklearn.model_selection import KFold
from loguru import logger

from .prepare_data import prepare_data, prepare_train_valid_data, prepare_kfold_data, prepare_test_data, prepare_kfold_train_valid_data
from .trainer import Trainer
from .utils import write_accuracy_file, check_dir
from .kfold import do_kfold_training


def do_train(pretrained, training, validation, testing, output_dir, results, torch_args, data_args, model_args, training_args):
    """This function will initiate/load model, do the training and write the classification report

    Args:
        archi (str): the desired architecture for the model
        device (str): the desired device to use for computation
        training (str): Training dictionary
        validation (str): Validation dictionary
        testing (str): Testing dictionary
        tag_scheme (List[str]): All available NER tags for the given data set EXCLUDING the special outside tag, that is handled separately
        o_tag_cr (bool): To include O tag in the classification report
        hyperparameters (dict): Hyperparameters for the model
        tokenizer_parameters (dict): Parameters for the tokenizer
        max_len (int):  The maximum sentence length
        dropout (float): dropout probability
        pretrained (str): which pretrained 'huggingface' transformer to use
        isModelExists (bool): True if trained model exist and want to retrain on its weights, otherwise False.
        model_path (str): Trained model path if isModelExist is True, otherwise leave it as empty.
        tokenizer_path (str): Existing tokenizer path if isModelExist is True, otherwise leave it as empty.
        output_dir (str): Output directory to save trained model and clasification report
        results (List[float]): A list of accuracy scores
    """
    model = Trainer(torch_args, data_args, model_args,
                    training_args, pretrained, training, validation,)
    
    if(training_args.continue_from_checkpoint):
        assert os.path.isfile(
            training_args.checkpoint_path), f'File {training_args.checkpoint_path} does not exist.'
        logger.info("Model weights loading...")
        if(training_args.checkpoint_tokenizer_path != None):
            model.load_network_from_file(
                model_path=training_args.checkpoint_path, tokenizer_path=training_args.checkpoint_tokenizer_path)
        else:
            model.load_network_from_file(
                model_path=training_args.checkpoint_path)
        logger.success("Model weights loaded!")

    logger.info("=====================================")
    logger.info("Training started!")
    model.train()
    logger.info("Training finished!")
    logger.info("=====================================")

    # save model
    model.save_network(output_dir=output_dir)
    logger.success("Model stored!")

    logger.info("Evaluating performance on testing dataset...")
    # evaluate on test set
    c_reports = [
        model.evaluate_performance(t) for t in testing
    ]

    # write logs
    report_names = [
        os.path.join(output_dir, "classification_report-" + str(len(results)) +
                     ".txt") if len(results) == 0 else os.path.join(output_dir, f"classification_report_{i+1}.txt")
        for i, c in enumerate(c_reports)
    ]

    for i, r in enumerate(report_names):
        with open(r, "w") as wf:
            wf.write(c_reports[i]["f1"])

    if(training_args.return_accuracy):
        results.append(c_reports[0]["accuracy"])
    logger.success("Evaluation metrics stored!")


def training_pipeline(torch_args, data_args, model_args, training_args, kfold_args) -> None:

    # getting vars
    for pretrained in model_args.pretrained_models:
        torch.cuda.empty_cache()
        
        # Creating model directory if not exists
        model_dir = check_dir(training_args.output_dir, pretrained.split("/")[-1] if len(pretrained.split("/")) > 1 else pretrained)

        if(kfold_args.is_kfold):
            # kfold training
            model_dir, results = do_kfold_training(
                pretrained, model_dir, torch_args, data_args, model_args, training_args, kfold_args)
            
            # write accuracy file
            write_accuracy_file(model_dir, results)

        else:
            training, validation = prepare_train_valid_data(train_data, valid_data, torch_args, data_args)
            testing = [prepare_test_data(t, data_args.limit) for t in test_data]

            logger.info(
                "Training {model} without K-Fold!".format(model=pretrained))
            do_train(pretrained, training, validation, testing, model_dir, [0], torch_args, data_args, model_args, training_args)

    return "Training finished successfully!"
