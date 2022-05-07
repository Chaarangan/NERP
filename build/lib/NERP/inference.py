from NERDA.models import NERDA
from NERP.utils import SentenceGetter
import pandas as  pd
import os

# if you get a warning regarding TOKENIZERS_PARALLELISM, uncomment the below line.
#os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_model(tag_scheme, pretrained, max_len, model_path, tokenizer_path, hyperparameters, tokenizer_parameters):

    # compile model
    model = NERDA(
        tag_scheme=tag_scheme,
        transformer=pretrained,
        max_len=max_len,
        tokenizer_parameters=tokenizer_parameters,
        hyperparameters=hyperparameters
    )

    # getting inference vars
    assert os.path.isfile(model_path), f'File {model_path} does not exist.'
    assert os.path.isdir(tokenizer_path), f'Folder {tokenizer_path} does not exist.'

    model.load_network_from_file(model_path=model_path, tokenizer_path=tokenizer_path)
    print("Model weights loaded!")
    return model


def predict_bulk(model, in_file_path, out_file_path):
    """
      Args:
          model (object, required): Compiled model
          in_file_path (str, required): Input csv file path
          out_file_path (str, required): Output csv file path

      Returns:
          Nothing. Saves predictions to a file as a side-effect
    """
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
        print(
            "Predicted on sentence no: {no} - {sentence}".format(no=i, sentence=sentence))
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
    print("Predictions stored!")


def inference_pipeline(pretrained,
                       model_path,
                       tokenizer_path,
                       out_file_path,
                       is_bulk,
                       in_file_path,
                       text,
                       tag_scheme,
                       hyperparameters,
                       tokenizer_parameters,
                       max_len):

    model = load_model(tag_scheme, pretrained, max_len,
                       model_path, tokenizer_path, hyperparameters, tokenizer_parameters)

    if(is_bulk):
        print("Bulk Mode!")
        predict_bulk(model, in_file_path, out_file_path)

    else:
        print(model.predict_text(text))
    
    return "Successfully finished!"
    
