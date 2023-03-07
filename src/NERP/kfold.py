from .utils import check_dir
from .training import do_train

def do_kfold_training(pretrained, model_dir, torch_args, data_args, model_args, training_args, kfold_args):
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
    kf = KFold(n_splits=kfold,
                       random_state=torch_args.seed, shuffle=data_args.shuffle)

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
            testing = [prepare_test_data(t, data_args.limit) for t in test_data]

        else:
            training, validation = prepare_kfold_train_valid_data(
                training, data_args.train_test_split, torch_args.seed)

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

        do_train(pretrained, training, validation, testing, os.path.join(model_dir, k_fold_step), results, torch_args, data_args, model_args, training_args)
        
    return model_dir, results