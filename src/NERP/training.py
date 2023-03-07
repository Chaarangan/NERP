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


def do_train(pretrained, training, validation, testing, output_dir, results, torch_args, data_args, model_args, training_args):
    """This function will initiate/load model, do the training and write the classification report

    Args:
        pretrained (str): which pretrained 'huggingface' transformer to use
        training (str): Training dictionary
        validation (str): Validation dictionary
        testing (str): Testing dictionary
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


def do_kfold_training(pretrained, model_dir, torch_args, data_args, model_args, training_args, kfold_args):
    """This function will do the kfold training

    Args:
        pretrained (str): which pretrained 'huggingface' transformer to use
        model_dir (str): Output directory to save trained model and clasification report
    
    Returns:
        str: Output directory
        List[float]: accuracy scores of all folds
    """
    logger.info("Training {model} with K-Fold!".format(model=pretrained))

    # check for directory
    model_dir = check_dir(model_dir, "kfold")

    # create df
    data = prepare_kfold_data(
        torch_args.seed,
        data_args.train_data,
        data_args.valid_data,
        data_args.test_data,
        data_args.limit,
        data_args.sep,
        data_args.quoting,
        data_args.shuffle,
        kfold_args.test_on_original)

    # Creating dataset directory if not exists
    dataset_dir = check_dir(model_dir, "datasets")

    # prepare cross validation
    kf = KFold(n_splits=kfold_args.splits,
               random_state=torch_args.seed, shuffle=True)

    results = []
    for train_index, test_index in kf.split(data["sentences"]):
        k_fold_step = str(len(results) + 1)
        logger.warning("K-Fold Step: " + k_fold_step)

        # splitting Dataframe (dataset not included)
        training = {"sentences": [data["sentences"][i] for i in train_index], "tags": [
                    data["tags"][i] for i in train_index]}
        testing = {
            "sentences": [data["sentences"][i] for i in test_index], "tags": [data["tags"][i] for i in test_index]}

        if(kfold_args.test_on_original):
            validation = testing
            testing = [prepare_test_data(t, data_args.limit)
                       for t in test_data]

        else:
            training, validation = prepare_kfold_train_valid_data(
                training, data_args.train_valid_split, torch_args.seed)

            logger.info("Test: ({a}, {b})".format(
                        a=str(len(testing["sentences"])), b=str(len(testing["tags"]))))
            logger.info("Test dataset is prepared!")

            df_test = pd.DataFrame(
                testing, columns=["sentences", "tags"])
            df_test.to_csv(os.path.join(
                dataset_dir, "test -{n}.csv".format(n=k_fold_step)), index=False)

            testing = [testing]

        df_train = pd.DataFrame(
            training, columns=["sentences", "tags"])
        df_train.to_csv(os.path.join(
            dataset_dir, "train-{n}.csv".format(n=k_fold_step)), index=False)

        df_valid = pd.DataFrame(
            validation, columns=["sentences", "tags"])
        df_valid.to_csv(os.path.join(
            dataset_dir, "valid-{n}.csv".format(n=k_fold_step)), index=False)

        do_train(pretrained, training, validation, testing, os.path.join(
            model_dir, k_fold_step), results, torch_args, data_args, model_args, training_args)

    return model_dir, results

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
            if(training_args.return_accuracy):
                write_accuracy_file(model_dir, results)

        else:
            training, validation = prepare_train_valid_data(data_args.train_data, data_args.valid_data, torch_args, data_args)
            testing = [prepare_test_data(t, data_args.limit) for t in data_args.test_data]

            logger.info(
                "Training {model} without K-Fold!".format(model=pretrained))
            do_train(pretrained, training, validation, testing, model_dir, [0], torch_args, data_args, model_args, training_args)

    return "Training finished successfully!"
