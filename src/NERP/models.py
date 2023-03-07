"""
This section covers the interface for `NERP` models, that is implemented as its own Python class [NERP.models.NERP][]. The interface enables you to easily 
    - specify your own [NERP.models.NERP][] model
    - train it
    - evaluate it
    - use it to predict entities in new texts.
"""

import yaml
import os
import shutil

from transformers import logging

from .inference import inference_pipeline
from .training import training_pipeline

logging.set_verbosity_error()


class NERP:
    def __init__(self,
                 config: str = "env.yaml") -> None:

        stream = open(config, 'r')
        dictionary = yaml.load(stream, Loader=yaml.FullLoader)

        self.device = dictionary["torch"]["device"]
        if self.device == None:
            self.device = "cpu"
        self.tag_scheme = dictionary["data"]["tags"]
        self.o_tag_cr = dictionary["train"]["o_tag_cr"]
        self.hyperparameters = dictionary["model"]["hyperparameters"]
        if(self.hyperparameters["epochs"] == None):
            self.hyperparameters["epochs"] = 5
        if(self.hyperparameters["warmup_steps"] == None):
            self.hyperparameters["warmup_steps"] = 500
        if(self.hyperparameters["batch_size"]["train"] == None):
            self.hyperparameters["batch_size"]["train"] = 64
        if(self.hyperparameters["batch_size"]["valid"] == None):
            self.hyperparameters["batch_size"]["valid"] = 64
        if(self.hyperparameters["lr"] == None):
            self.hyperparameters["lr"] = 0.0001
        if(self.hyperparameters["seed"] == None):
            self.hyperparameters["seed"] = 42
        self.tokenizer_parameters = dictionary["model"]["tokenizer_parameters"]
        if(self.tokenizer_parameters["do_lower_case"] == None):
            self.tokenizer_parameters["do_lower_case"] = True

        self.archi = dictionary["model"]["archi"]
        if self.archi == None:
            self.archi = "baseline"
        self.max_len = dictionary["model"]["max_len"]
        if self.max_len == None:
            self.max_len = 256
        self.dropout = dictionary["model"]["dropout"]
        if self.dropout == None:
            self.dropout = 0
        self.num_workers = dictionary["model"]["num_workers"]
        if self.num_workers == None:
            self.num_workers = 0
        self.pretrained_models = dictionary["model"]["pretrained_models"]
        if self.pretrained_models == [None]:
            self.pretrained_models = ["roberta-base"]
        self.train_data = dictionary["data"]["train"]
        self.train_data_parameters = dictionary["data"]["parameters"]
        if(self.train_data_parameters["sep"] == None):
            self.train_data_parameters["sep"] = ','
        if(self.train_data_parameters["quoting"] == None):
            self.train_data_parameters["quoting"] = True
        if(self.train_data_parameters["shuffle"] == None):
            self.train_data_parameters["shuffle"] = True
        self.valid_data = dictionary["data"]["valid"]
        if self.valid_data == "":
            self.valid_data = None
        self.train_valid_split = dictionary["data"]["train_valid_split"]
        if self.train_valid_split == None:
            self.train_valid_split = 0.2
        self.test_data = dictionary["data"]["test"]
        self.limit = dictionary["data"]["limit"]
        if self.limit == None:
            self.limit = 0
        self.existing_model_path = dictionary["train"]["existing_model_path"]
        self.existing_tokenizer_path = dictionary["train"]["existing_tokenizer_path"]

        self.output_dir = dictionary["train"]["output_dir"]
        if self.output_dir == None:
            self.output_dir = "./output"
        if os.path.exists(self.output_dir):
            try:
                shutil.rmtree(self.output_dir)
            except OSError as e:
                logger.error("Error: %s - %s." % (e.filename, e.strerror))

        self.kfold = dictionary["kfold"]["splits"]
        if self.kfold == None:
            self.kfold = 0
        self.test_on_original = dictionary["kfold"]["test_on_original"]
        if(self.test_on_original == None):
            self.test_on_original = False

        self.pretrained = dictionary["inference"]["pretrained"]
        self.model_path = dictionary["inference"]["model_path"]
        self.tokenizer_path = dictionary["inference"]["tokenizer_path"]
        self.in_file_path = dictionary["inference"]["bulk"]["in_file_path"]
        self.out_file_path = dictionary["inference"]["bulk"]["out_file_path"]
        if self.out_file_path == None:
            self.out_file_path = "output.csv"
        self.text = dictionary["inference"]["individual"]["text"]
        if self.text == None:
            self.text = "Hello from NERP"
        self.infer_max_len = dictionary["inference"]["max_len"]
        if self.infer_max_len == None:
            self.infer_max_len = 128

    def train(self) -> str:
        message = training_pipeline(archi=self.archi,
                                    device=self.device,
                                    train_data=self.train_data,
                                    valid_data=self.valid_data,
                                    test_data=self.test_data,
                                    existing_model_path=None,
                                    existing_tokenizer_path=None,
                                    tag_scheme=self.tag_scheme,
                                    o_tag_cr=self.o_tag_cr,
                                    limit=self.limit,
                                    test_size=self.train_valid_split,
                                    is_model_exists=False,
                                    output_dir=self.output_dir,
                                    pretrained_models=self.pretrained_models,
                                    hyperparameters=self.hyperparameters,
                                    tokenizer_parameters=self.tokenizer_parameters,
                                    train_data_parameters=self.train_data_parameters,
                                    max_len=self.max_len,
                                    dropout=self.dropout,
                                    kfold=0,
                                    test_on_original=False,
                                    num_workers=self.num_workers)

        return message

    def train_after_load_network(self) -> str:
        assert os.path.isfile(
            self.existing_model_path), f'File {self.existing_model_path} does not exist.'

        message = training_pipeline(archi=self.archi,
                                    device=self.device,
                                    train_data=self.train_data,
                                    valid_data=self.valid_data,
                                    test_data=self.test_data,
                                    existing_model_path=self.existing_model_path,
                                    existing_tokenizer_path=self.existing_tokenizer_path,
                                    tag_scheme=self.tag_scheme,
                                    o_tag_cr=self.o_tag_cr,
                                    limit=self.limit,
                                    test_size=self.train_valid_split,
                                    is_model_exists=True,
                                    output_dir=self.output_dir,
                                    pretrained_models=self.pretrained_models,
                                    hyperparameters=self.hyperparameters,
                                    tokenizer_parameters=self.tokenizer_parameters,
                                    train_data_parameters=self.train_data_parameters,
                                    max_len=self.max_len,
                                    dropout=self.dropout,
                                    kfold=0,
                                    test_on_original=False,
                                    num_workers=self.num_workers)

        return message

    def train_with_kfold(self) -> str:
        assert self.kfold >= 2, f'Number of splits are {self.kfold}. Should be greater or equal to 2.'

        message = training_pipeline(archi=self.archi,
                                    device=self.device,
                                    train_data=self.train_data,
                                    valid_data=self.valid_data,
                                    test_data=self.test_data,
                                    existing_model_path=None,
                                    existing_tokenizer_path=None,
                                    tag_scheme=self.tag_scheme,
                                    o_tag_cr=self.o_tag_cr,
                                    limit=self.limit,
                                    test_size=self.train_valid_split,
                                    is_model_exists=False,
                                    output_dir=self.output_dir,
                                    pretrained_models=self.pretrained_models,
                                    hyperparameters=self.hyperparameters,
                                    train_data_parameters=self.train_data_parameters,
                                    tokenizer_parameters=self.tokenizer_parameters,
                                    max_len=self.max_len,
                                    dropout=self.dropout,
                                    kfold=self.kfold,
                                    test_on_original=self.test_on_original,
                                    num_workers=self.num_workers)

        return message

    def train_with_kfold_after_load_network(self) -> str:
        assert os.path.isfile(
            self.existing_model_path), f'File {self.existing_model_path} does not exist.'
        assert self.kfold >= 2, f'Number of splits are {self.kfold}. Should be greater or equal to 2.'

        message = training_pipeline(archi=self.archi,
                                    device=self.device,
                                    train_data=self.train_data,
                                    valid_data=self.valid_data,
                                    test_data=self.test_data,
                                    existing_model_path=self.existing_model_path,
                                    existing_tokenizer_path=self.existing_tokenizer_path,
                                    tag_scheme=self.tag_scheme,
                                    o_tag_cr=self.o_tag_cr,
                                    limit=self.limit,
                                    test_size=self.train_valid_split,
                                    is_model_exists=True,
                                    output_dir=self.output_dir,
                                    pretrained_models=self.pretrained_models,
                                    hyperparameters=self.hyperparameters,
                                    train_data_parameters=self.train_data_parameters,
                                    tokenizer_parameters=self.tokenizer_parameters,
                                    max_len=self.max_len,
                                    dropout=self.dropout,
                                    kfold=self.kfold,
                                    test_on_original=self.test_on_original,
                                    num_workers=self.num_workers)

        return message

    def inference_text(self) -> str:
        assert os.path.isfile(
            self.model_path), f'File {self.model_path} does not exist.'

        output = inference_pipeline(archi=self.archi,
                                    device=self.device,
                                    model_path=self.model_path,
                                    tokenizer_path=self.tokenizer_path,
                                    out_file_path=None,
                                    in_file_path=None,
                                    pretrained=self.pretrained,
                                    is_bulk=False,
                                    text=self.text,
                                    tag_scheme=self.tag_scheme,
                                    hyperparameters=self.hyperparameters,
                                    tokenizer_parameters=self.tokenizer_parameters,
                                    max_len=self.infer_max_len)

        return output

    def predict(self, text) -> str:
        assert os.path.isfile(
            self.model_path), f'File {self.model_path} does not exist.'

        assert text != None, "Please input a text!"

        output = inference_pipeline(archi=self.archi,
                                    device=self.device,
                                    model_path=self.model_path,
                                    tokenizer_path=self.tokenizer_path,
                                    out_file_path=None,
                                    in_file_path=None,
                                    pretrained=self.pretrained,
                                    is_bulk=False,
                                    text=text,
                                    tag_scheme=self.tag_scheme,
                                    hyperparameters=self.hyperparameters,
                                    tokenizer_parameters=self.tokenizer_parameters,
                                    max_len=self.infer_max_len)

        return output

    def inference_bulk(self) -> str:
        assert os.path.isfile(
            self.model_path), f'File {self.model_path} does not exist.'
        assert os.path.isfile(
            self.in_file_path), f'File {self.in_file_path} does not exist.'

        if not self.out_file_path.endswith(".csv"):
            raise TypeError("Check output file path (out_file_path)!")

        inference_pipeline(archi=self.archi,
                           device=self.device,
                           model_path=self.model_path,
                           tokenizer_path=self.tokenizer_path,
                           out_file_path=self.out_file_path,
                           in_file_path=self.in_file_path,
                           pretrained=self.pretrained,
                           is_bulk=True,
                           text=None,
                           tag_scheme=self.tag_scheme,
                           hyperparameters=self.hyperparameters,
                           tokenizer_parameters=self.tokenizer_parameters,
                           max_len=self.infer_max_len)
