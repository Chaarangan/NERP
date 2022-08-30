'''
File: NERP/training.py
Project: NERP
Created Date: Tuesday, May 24th 2022
Author: Charangan Vasantharajan
-----
Last Modified: Friday, Aug 26th 2022
Modified By: Charangan Vasantharajan
-----
Copyright (c) 2022
------------------------------------
This section covers functionality for training Named Entity Recognition models.
    - with k-fold
    - without k-fold
'''

from typing import List
import os
import pandas as pd
from sklearn.model_selection import KFold
import torch
from NERP.compile_model import compile_model
from NERP.prepare_data import prepare_data, prepare_train_valid_data, prepare_kfold_data, prepare_test_data, prepare_kfold_train_valid_data

def do_train(archi, device, training, validation, testing, tag_scheme, o_tag_cr, hyperparameters, tokenizer_parameters, max_len, dropout, pretrained, isModelExists, model_path, tokenizer_path, model_dir, results, return_accuracy):
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
        model_dir (str): Output directory to save trained model and clasification report
        results (List[float]): A list of accuracy scores
        return_accuracy (bool): To return accuracy during training
    """
    model = compile_model(archi, device, training, validation, tag_scheme, o_tag_cr,
                          hyperparameters, tokenizer_parameters, max_len, dropout, pretrained)
    if(isModelExists):
      print("Model weights loading..")
      if(tokenizer_path != None):
          model.load_network_from_file(
              model_path=model_path, tokenizer_path=tokenizer_path)
      else:
          model.load_network_from_file(model_path=model_path)
      print("Model weights loaded!")

    print("Training started!")
    model.train()

    # save model
    model.save_network(output_dir=model_dir)
    print("Model stored!")

    print("Evaluating performance on testing dataset...!")
    # evaluate on test set
    c_report = model.evaluate_performance(
        testing, return_accuracy=return_accuracy)

    # write logs
    report_name = os.path.join(model_dir, "classification_report-" + str(len(results)) +
                               ".txt") if len(results) == 0 else os.path.join(model_dir, "classification_report.txt")
    with open(report_name, "w") as wf:
      wf.write(c_report["f1"])

    if(return_accuracy):
        results.append(c_report["accuracy"])
    print("Evaluation metrics stored!")

def write_accuracy_file(model_dir, results):
    """This function will write the kfold accuracy file

    Args:
        model_dir (str): Output model directory to store results
        results (List[float]): A list of accuracy scores
    """
    with open(os.path.join(model_dir, "k-fold-accuracy-scores.txt"), "w") as wf:
        wf.write("K-Fold Accuracy Scores\n")
        for i in range(len(results)):
            wf.write(f"Step {i}: {results[i]} \n")

        wf.write("\n")
        wf.write(f"Mean-Accuracy: {sum(results) / len(results)}")

    print(f"Mean-Accuracy: {sum(results) / len(results)}")
    print("Accuracy file stored!")


def training_pipeline(archi,
                      device,
                      train_data,
                      valid_data,
                      test_data,
                      existing_model_path,
                      existing_tokenizer_path,
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
                      o_tag_cr: bool = True,
                      limit: int = 0,
                      test_size: float = 0.2,
                      is_model_exists: bool = False,
                      output_dir: str = "./output",
                      pretrained_models: List[str] = ["roberta-base"],
                      hyperparameters: dict = {"epochs": 5,
                                               "warmup_steps": 500,
                                               "train_batch_size": 64,
                                               "learning_rate": 0.0001},
                      tokenizer_parameters: dict = {"do_lower_case": True},
                      max_len: int = 128,
                      dropout: float = 0.1,
                      kfold: int = 0,
                      seed: int = 42,
                      test_on_original: bool = False) -> str:

    # getting vars
    for pretrained in pretrained_models:
        torch.cuda.empty_cache()
        # Creating model directory if not exists
        model_dir = os.path.join(output_dir, pretrained.split(
            "/")[-1] if len(pretrained.split("/")) > 1 else pretrained)
        if(not os.path.exists(model_dir)):
            print("Directory not found: {model_dir}".format(
                model_dir=model_dir))
            os.makedirs(model_dir)
            print("Directory created: {model_dir}".format(model_dir=model_dir))

        if(kfold != 0):
            print("Training {model} with K-Fold!".format(model=pretrained))

            model_dir = os.path.join(model_dir, "kfold")
            if(not os.path.exists(model_dir)):
                print("Directory not found: {model_dir}".format(
                    model_dir=model_dir))
                os.makedirs(model_dir)
                print("Directory created: {model_dir}".format(
                    model_dir=model_dir))

            # create df
            data = prepare_kfold_data(
                train_data, valid_data, test_data, limit, test_on_original)

            # Creating dataset directory if not exists
            dataset_dir = os.path.join(model_dir, "datasets")
            if(not os.path.exists(dataset_dir)):
                print("Directory not found: {dataset_dir}".format(
                    dataset_dir=dataset_dir))
                os.makedirs(dataset_dir)
                print("Directory created: {dataset_dir}".format(
                    dataset_dir=dataset_dir))

            # prepare cross validation
            kf = KFold(n_splits=kfold, random_state=seed, shuffle=True)

            results = []
            for train_index, test_index in kf.split(data["sentences"]):
                k_fold_step = str(len(results) + 1)
                print("K-Fold Step: " + k_fold_step)

                # splitting Dataframe (dataset not included)
                training = {"sentences": [data["sentences"][i] for i in train_index], "tags": [
                    data["tags"][i] for i in train_index]}
                testing = {
                    "sentences": [data["sentences"][i] for i in test_index], "tags": [data["tags"][i] for i in test_index]}
                
                if(test_on_original):
                    validation = testing
                    testing = prepare_test_data(test_data, limit)  
        
                else:  
                    training, validation = prepare_kfold_train_valid_data(
                        training, test_size)
                    
                    print("Test: ({a}, {b})".format(
                        a=str(len(testing["sentences"])), b=str(len(testing["tags"]))))
                    print("Test dataset is prepared!")
                    
                    df_test = pd.DataFrame(
                        testing, columns=["sentences", "tags"])
                    df_test.to_csv(os.path.join(
                        dataset_dir, "test-{n}.csv".format(n=k_fold_step)), index=False)

                df_train = pd.DataFrame(
                        training, columns=["sentences", "tags"])
                df_train.to_csv(os.path.join(
                    dataset_dir, "train-{n}.csv".format(n=k_fold_step)), index=False)
                
                df_valid = pd.DataFrame(
                    validation, columns=["sentences", "tags"])
                df_valid.to_csv(os.path.join(
                    dataset_dir, "valid-{n}.csv".format(n=k_fold_step)), index=False)               
                
                do_train(archi, device, training, validation, testing, tag_scheme, o_tag_cr, hyperparameters, tokenizer_parameters, max_len,
                         dropout, pretrained, is_model_exists, existing_model_path, existing_tokenizer_path, os.path.join(model_dir, k_fold_step), results, True)

            # write accuracy file
            write_accuracy_file(model_dir, results)

        else:
            training, validation = prepare_train_valid_data(train_data, valid_data, limit, test_size)
            testing = prepare_test_data(test_data, limit)
            
            print("Training {model} without K-Fold!".format(model=pretrained))
            do_train(archi, device, training, validation, testing, tag_scheme, o_tag_cr, hyperparameters, tokenizer_parameters, max_len,
                     dropout, pretrained, is_model_exists, existing_model_path, existing_tokenizer_path,  model_dir, [0], False)

    return "Training finished successfully!"
