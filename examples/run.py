from NERP.models import NERP

if __name__ == '__main__':
    model = NERP("env.yaml")

    # simple baseline training
    message = model.train()
    print(message)

    # training with k-fold cros validation
    message = model.train_with_kfold()
    print(message)

    # train a already trained model by loading its weights
    message = model.train_after_load_network()
    print(message)

    # train with k-fold cross validation a already trained model by loading its weights
    message = model.train_with_kfold_after_load_network()
    print(message)

    # make predictions on a single sentence
    output, message = model.inference_text()
    print(message)
    print(output)

    # make predictions on a csv file
    output, message = model.inference_bulk()
    print(message)


