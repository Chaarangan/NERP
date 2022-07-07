from NERP.models import NERP

if __name__ == '__main__':
    model = NERP("env.yaml")

    # simple baseline training
    model.train()

    # training with k-fold cros validation
    model.train_with_kfold()

    # train a already trained model by loading its weights
    model.train_after_load_network()

    # train with k-fold cross validation a already trained model by loading its weights
    model.train_with_kfold_after_load_network()

    # make predictions on a single sentence
    output = model.inference_text()
    print(output)

    # make predictions on a single sentence via direct input
    output1 = model.predict("Hello from NERP")
    print(output1)

    # make predictions on a csv file
    model.inference_bulk()


