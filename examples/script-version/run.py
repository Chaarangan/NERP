from NERP.models import NERP

if __name__ == '__main__':
    model = NERP("env.yaml")

    # simple baseline training
    model.train()

    # make predictions on a single sentence
    output = model.predict_text("Hello from NERP")
    print(output)

    # make predictions on a input file
    model.predict()