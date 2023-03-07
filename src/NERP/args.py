import csv
import yaml

from typing import List
from dictparse import DictionaryParser

def flatten_dict(init_dict):
    res_dict = {}
    if type(init_dict) is not dict:
        return res_dict

    for k, v in init_dict.items():
        if type(v) == dict:
            res_dict.update(flatten_dict(v))
        else:
            res_dict[k] = v

    return res_dict

def torch_args(dict):
    parser = DictionaryParser()

    parser.add_param('device', str, required=False, default="cpu",
                     description="The desired device to use for computation.")
    parser.add_param('seed', int, required=False,
                     default=42, description="Random state value for a particular experiment")
    args = parser.parse_dict(dict)
    return args


def data_args(dict):
    parser = DictionaryParser()

    parser.add_param('train_data', str, required=False, default=None,
                     description="Path to training file")
    parser.add_param('valid_data', str, required=False,
                     default=None, description="Path to validation file")
    parser.add_param('train_valid_split', float(), required=False,
                     default=0.2, description="Train/valid split ratio if valid data not exists")
    parser.add_param('test_data', list, required=False,
                     default=None, description="Path to test file")
    parser.add_param('sep', str, required=False,
                     default=',', description="Delimiter to use")
    parser.add_param('quoting', int, choices=[0,1,2,3], required=False,
                     default=3, description="Control field quoting behavior per csv.QUOTE_* constants. Refer: https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html")
    parser.add_param('shuffle', bool, required=False,
                     default=False, description="Shuffle the entire dataset before training")
    parser.add_param('limit', int, required=False,
                     default=0, description="Limit the number of observations to be returned from a given split")
    parser.add_param('tags', list, required=True,
                     default=[], description="All available NER tags for the given data set EXCLUDING the special outside tag, that is handled separately")

    args = parser.parse_dict(dict)
    return args


def model_args(dict):
    parser = DictionaryParser()

    parser.add_param('archi', str, required=False, default="baseline",
                     description="The desired architecture for the model (baseline, bilstm-crf, bilstm, crf)")
    parser.add_param('max_len', int, required=False, default=128,
                     description="The maximum sentence length (number of tokens after applying the transformer tokenizer)")
    parser.add_param('dropout', float, required=False, default=0.1,
                     description="Dropout probability")
    parser.add_param('num_workers', int, required=False, default=1,
                     description="Number of workers/threads for data loader")
    parser.add_param('epochs', int, required=False, default=10,
                     description="Number of epochs")
    parser.add_param('warmup_steps', int, required=False, default=500,
                     description="Number of learning rate warmup steps")
    parser.add_param('train_batch_size', int, required=False, default=64,
                     description="Batch Size for training dataLoader")
    parser.add_param('valid_batch_size', int, required=False, default=8,
                     description="Batch Size for validation dataLoader")
    parser.add_param('lr', float, required=False, default=0.0001,
                     description="Learning rate")
    parser.add_param('do_lower_case', bool, required=False, default=False,
                     description="Lowercase the input sequence")
    parser.add_param('pretrained_models', list, required=False, default=["bert-base-uncased"],
                     description="List of 'huggingface' transformer models")
    
    args = parser.parse_dict(dict)
    return args


def training_args(dict):
    parser = DictionaryParser()

    parser.add_param('continue_from_checkpoint', bool, required=False, default=False,
                     description="Continue training from a checkpoint")
    parser.add_param('checkpoint_path', str, required=False, default=None,
                     description="Model derived from the transformer/ Trained transformer model")
    parser.add_param('checkpoint_tokenizer_path', str, required=False, default=None,
                     description="Tokenizer derived from the transformer")
    parser.add_param('output_dir', str, required=False, default="./output",
                     description="Path to output directory")
    parser.add_param('o_tag_cr', bool, required=False, default=True,
                     description="Include O tag in the classification report")
    parser.add_param('return_accuracy', bool, required=False, default=False,
                     description="Return accuracy after training")

    args = parser.parse_dict(dict)
    return args


def kfold_args(dict):
    parser = DictionaryParser()

    parser.add_param('is_kfold', bool, required=False, default=False,
                     description="Experiment with kfold validation")
    parser.add_param('splits', int, required=False, default=0,
                     description="Number of folds in the KFold experiment")
    parser.add_param('test_on_original', bool, required=False, default=False,
                     description="Test on the original test set for each iteration")

    args = parser.parse_dict(dict)
    return args


def inference_args(dict):
    parser = DictionaryParser()

    parser.add_param('pretrained', str, required=False, default=None,
                     description="'huggingface' transformer model")
    parser.add_param('model_path', str, required=False, default=None,
                     description="Path to trained model")
    parser.add_param('tokenizer_path', str, required=False, default=None,
                     description="Path to saved tokenizer directory")
    parser.add_param('in_file_path', str, required=False, default=None,
                     description="Path to input file if you are predicting on bulk mode")
    parser.add_param('out_file_path', str, required=False, default=None,
                     description="Path to output file if you are predicting on bulk mode")

    args = parser.parse_dict(dict)
    return args


def parse_env_variables(config):
    stream = open(config, 'r')
    dictionary = yaml.load(stream, Loader=yaml.FullLoader)
    flatted_dict = flatten_dict(dictionary)
    
    return torch_args(flatted_dict), data_args(flatted_dict), model_args(flatted_dict), training_args(flatted_dict), kfold_args(flatted_dict), inference_args(flatted_dict)
