"""
This section covers functionality for training Named Entity 
Recognition models.
- with k-fold
- without k-fold
Author: Charangan Vasantharajan
"""
from typing import List
import os
import pandas as pd
from sklearn.model_selection import KFold
import torch


def do_train(archi, device, train_data, valid_data, test_data, limit, tag_scheme, hyperparameters, tokenizer_parameters, max_len, dropout, pretrained, test_size, isModelExists, model_path, tokenizer_path, model_dir, results, return_accuracy):
    """
    Args:
        archi (str, optional): the desired architecture for the model
        device (str, optional): the desired device to use for computation. 
                If not provided by the user, we take a guess.
        train_data (str, required): Train csv file path
        valid_data (str, optional): Valid csv file path
        test_data (str, required): Test csv file path
        limit (int, optional): Limit the number of observations to be 
            returned from a given split. Defaults to None, which implies 
            that the entire data split is returned
        tag_scheme (List[str], optional): All available NER 
                tags for the given data set EXCLUDING the special outside tag, 
                that is handled separately.
        hyperparameters (dict, optional): Hyperparameters for the model
        tokenizer_parameters (dict, optional): Parameters for the tokenizer
        max_len (int, required): The maximum sentence length
        dropout (float, required): dropout probability
        pretrianed (str, required): which pretrained 'huggingface' 
                transformer to use
        test_size (float, optional): train/test split ratio
        isModelExists (bool, required): True if trained model exist and want to retrain on its weights, otherwise False.
        model_path (str, optional): Trained model path if isModelExist is True, otherwise leave it as empty.
        tokenizer_path (str, optional): Existing tokenizer path if isModelExist is True, otherwise leave it as empty.
        model_dir (str, required): Output directory to save trained model and clasification report
        results (List[float], required): A list of accuracy scores

    Returns:
        results (List[float]): A list of accuracy scores
        Save trained model and classification report


    """
    model = compile_model(archi, device, train_data, valid_data, limit, tag_scheme,
                          hyperparameters, tokenizer_parameters, max_len, dropout, pretrained, test_size)
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

    # evaluate on test set
    test = prepare_data(limit, test_data)
    print("Test: ({a}, {b})".format(
        a=str(len(test["sentences"])), b=str(len(test["tags"]))))
    print("Test dataset is prepared!")

    c_report = model.evaluate_performance(
        test, return_accuracy=return_accuracy)

    # write logs
    report_name = os.path.join(model_dir, "classification_report-" + str(len(results)) +
                               ".txt") if len(results) == 0 else os.path.join(model_dir, "classification_report.txt")
    with open(report_name, "w") as wf:
      wf.write(c_report["f1"])

    if(return_accuracy):
        results.append(c_report["accuracy"])
    print("Evaluation metrics stored!")


def write_accuracy_file(model_dir, results):
    """
    Args:
        model_dir (str, required): Output model directory to store results
        results (List[float], required): A list of accuracy scores

    Returns:
        Nothing. Save results as a text file in the output directory.
    """
    with open(os.path.join(model_dir, "k-fold-accuracy-scores.txt"), "w") as wf:
        wf.write("K-Fold Accuracy Scores\n")
        for i in range(len(results)):
            wf.write(f"Step {i}: {results[i]} \n")

        wf.write("\n")
        wf.write(f"Mean-Accuracy: {sum(results) / len(results)}")

    print(results)
    print(f"Mean-Accuracy: {sum(results) / len(results)}")


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
                      seed: int = 42) -> str:

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
            df_train = pd.read_csv(train_data)
            df_test = pd.read_csv(test_data)
            df = pd.concat([df_train, df_test])

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
            for train_index, val_index in kf.split(df):
                k_fold_step = str(len(results) + 1)
                print("K-Fold Step: " + k_fold_step)

                # splitting Dataframe (dataset not included)
                train_df = df.iloc[train_index]
                test_df = df.iloc[val_index]

                train_data = os.path.join(
                    dataset_dir, "train-{n}.csv".format(n=k_fold_step))
                test_data = os.path.join(
                    dataset_dir, "test-{n}.csv".format(n=k_fold_step))
                train_df.to_csv(train_data, index=False)
                test_df.to_csv(test_data, index=False)

                do_train(archi, device, train_data, valid_data, test_data, limit, tag_scheme, hyperparameters, tokenizer_parameters, max_len,
                         dropout, pretrained, test_size, is_model_exists, existing_model_path, existing_tokenizer_path, os.path.join(model_dir, k_fold_step), results, True)

            # write accuracy file
            write_accuracy_file(model_dir, results)

        else:
            print("Training {model} without K-Fold!".format(model=pretrained))
            do_train(archi, device, train_data, valid_data, test_data, limit, tag_scheme, hyperparameters, tokenizer_parameters, max_len,
                     dropout, pretrained, test_size, is_model_exists, existing_model_path, existing_tokenizer_path,  model_dir, [0], False)

    return "Training finished successfully!"
