import os
import pandas as pd
from sklearn.model_selection import KFold
from NERP.compile_model import compile_model
from NERP.prepare_data import prepare_data

# if you get a warning regarding TOKENIZERS_PARALLELISM, uncomment the below line.
#os.environ["TOKENIZERS_PARALLELISM"] = "false"


def do_train(train_data, test_data, limit, tag_scheme, hyperparameters, tokenizer_parameters, max_len, dropout, pretrained, test_size, isModelExists, model_path, model_dir, results):
    model = compile_model(train_data, limit, tag_scheme,
                          hyperparameters, tokenizer_parameters, max_len, dropout, pretrained, test_size)
    if(isModelExists):
      print("Model weights loading..")
      model.load_network_from_file(model_path=model_path)
      print("Model weights loaded!")

    print("Training started!")
    model.train()

    # save model
    #model_name = os.path.join(model_dir, "pytorch_model-" + str(len(results)) + ".bin") if len(results) == 0 else os.path.join(model_dir, "pytorch_model.bin")
    model.save_network(output_dir=model_dir)
    print("Model stored!")

    # evaluate on test set
    test = prepare_data(limit, test_data)
    print("Test dataset is prepared!")
    c_report = model.evaluate_performance(test, return_accuracy=True)

    # write logs
    report_name = os.path.join(model_dir, "classification_report-" + str(len(results)) + ".csv") if len(results) == 0 else os.path.join(model_dir, "classification_report.csv")
    c_report["f1"].to_csv(report_name, index=False)
    results.append(c_report["accuracy"])
    print("Evaluation metrics stored!")


def write_accuracy_file(model_dir, results):
    with open(os.path.join(model_dir, "k-fold-accuracy-scores.txt"), "w") as wf:
        wf.write("K-Fold Accuracy Scores\n")
        for i in range(len(results)):
            wf.write(f"Step {i}: {results[i]} \n")

        wf.write("\n")
        wf.write(f"Mean-Accuracy: {sum(results) / len(results)}")

    print(results)
    print(f"Mean-Accuracy: {sum(results) / len(results)}")


def training_pipeline(train_data,
                      test_data,
                      limit,
                      test_size,
                      is_model_exists,
                      existing_model_path,
                      output_dir,
                      kfold,
                      seed,
                      tag_scheme,
                      hyperparameters,
                      tokenizer_parameters,
                      max_len,
                      dropout,
                      pretrained_models):

    # getting vars
    for pretrained in pretrained_models:
        # Creating model directory if not exists
        model_dir = os.path.join(output_dir, pretrained.split(
            "/")[-1] if len(pretrained.split("/")) > 1 else pretrained)
        if(not os.path.exists(model_dir)):
            print("Directory not found: {model_dir}".format(model_dir=model_dir))
            os.makedirs(model_dir)
            print("Directory created: {model_dir}".format(model_dir=model_dir))

        if(kfold != 0):
            print("Training {model} with K-Fold!".format(model=pretrained))

            model_dir = os.path.join(model_dir, "kfold")
            if(not os.path.exists(model_dir)):
                print("Directory not found: {model_dir}".format(model_dir=model_dir))
                os.makedirs(model_dir)
                print("Directory created: {model_dir}".format(model_dir=model_dir))

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

                do_train(train_data, test_data, limit, tag_scheme, hyperparameters, tokenizer_parameters, max_len,
                         dropout, pretrained, test_size, is_model_exists, existing_model_path, os.path.join(model_dir, k_fold_step), results)

            # write accuracy file
            write_accuracy_file(model_dir, results)

        else:
            print("Training {model} without K-Fold!".format(model=pretrained))
            do_train(train_data, test_data, limit, tag_scheme, hyperparameters, tokenizer_parameters, max_len,
                     dropout, pretrained, test_size, is_model_exists, existing_model_path, model_dir, [0])
    
    return "Training finished successfully!"
