'''
File: NERP/models.py
Project: NERP
Created Date: Tuesday, May 24th 2022
Author: Charangan Vasantharajan
-----
Last Modified: Friday, Aug 26th 2022
Modified By: Charangan Vasantharajan
-----
Copyright (c) 2022
------------------------------------
This section covers the interface for `NERP` models, that is implemented as its own Python class [NERP.models.NERP][]. The interface enables you to easily 
    - specify your own [NERP.models.NERP][] model
    - train it
    - evaluate it
    - use it to predict entities in new texts.
'''

from typing import List
from NERP.inference import inference_pipeline
from NERP.training import training_pipeline
import yaml
import os

class NERP:
    def __init__(self,
                config: str = "env.yaml") -> None:

        stream = open(config, 'r')
        dictionary = yaml.load(stream, Loader=yaml.FullLoader)

        self.device = dictionary["torch"]["device"]
        if self.device == None:
            self.device = "cpu"
        self.tag_scheme = dictionary["data"]["tag_scheme"]
        self.o_tag_cr = dictionary["model"]["o_tag_cr"]
        self.hyperparameters = dictionary["model"]["hyperparameters"]
        if(self.hyperparameters["epochs"] == None):
            self.hyperparameters["epochs"] = 5
        if(self.hyperparameters["warmup_steps"] == None):
            self.hyperparameters["warmup_steps"] = 500
        if(self.hyperparameters["train_batch_size"] == None):
            self.hyperparameters["train_batch_size"] = 64
        if(self.hyperparameters["learning_rate"] == None):
            self.hyperparameters["learning_rate"] = 0.0001
        self.tokenizer_parameters = dictionary["model"]["tokenizer_parameters"]
        if(self.tokenizer_parameters["do_lower_case"] == None):
            self.tokenizer_parameters["do_lower_case"] = True
        self.archi = dictionary["model"]["archi"]
        if self.archi == None:
            self.archi = "baseline"
        self.max_len = dictionary["model"]["max_len"]
        if self.max_len == None:
            self.max_len = 128
        self.dropout = dictionary["model"]["dropout"]
        if self.dropout == None:
            self.dropout = 0.1
        self.pretrained_models = dictionary["model"]["pretrained_models"]
        if self.pretrained_models == [None]:
            self.pretrained_models = ["roberta-base"]
        self.train_data = dictionary["data"]["train_data"]
        self.valid_data = dictionary["data"]["valid_data"]
        if self.valid_data == "":
            self.valid_data = None
        self.train_valid_split = dictionary["data"]["train_valid_split"]
        if self.train_valid_split == None:
            self.train_valid_split = 0.2
        self.test_data = dictionary["data"]["test_data"]
        self.limit = dictionary["data"]["limit"]
        if self.limit == None:
            self.limit = 0
        self.existing_model_path = dictionary["train"]["existing_model_path"]
        self.existing_tokenizer_path = dictionary["train"]["existing_tokenizer_path"]
        self.output_dir = dictionary["train"]["output_dir"]
        if self.output_dir == None:
            self.output_dir = "./output"
        self.kfold = dictionary["kfold"]["splits"]
        if self.kfold == None:
            self.kfold = 0
        self.seed = dictionary["kfold"]["seed"]
        if self.seed == None:
            self.seed = 42
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
        message = training_pipeline(archi = self.archi,
                                    device = self.device, 
                                    train_data = self.train_data,
                                    valid_data=self.valid_data,
                                    test_data = self.test_data,
                                    existing_model_path=None,
                                    existing_tokenizer_path=None,
                                    tag_scheme=self.tag_scheme,
                                    o_tag_cr=self.o_tag_cr,
                                    limit = self.limit,
                                    test_size=self.train_valid_split,
                                    is_model_exists=False,
                                    output_dir=self.output_dir,
                                    pretrained_models=self.pretrained_models,
                                    hyperparameters=self.hyperparameters,
                                    tokenizer_parameters=self.tokenizer_parameters,
                                    max_len=self.max_len,
                                    dropout=self.dropout,
                                    kfold = 0,
                                    seed=42,
                                    test_on_original=False)
        
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
                                    max_len=self.max_len,
                                    dropout=self.dropout,
                                    kfold=0,
                                    seed=42,
                                    test_on_original=False)

        return message

    def train_with_kfold(self) -> str:
        assert self.kfold>=2, f'Number of splits are {self.kfold}. Should be greater or equal to 2.'
                
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
                                    max_len=self.max_len,
                                    dropout=self.dropout,
                                    kfold=self.kfold,
                                    seed=self.seed,
                                    test_on_original=self.test_on_original)

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
                                    tokenizer_parameters=self.tokenizer_parameters,
                                    max_len=self.max_len,
                                    dropout=self.dropout,
                                    kfold=self.kfold,
                                    seed=self.seed,
                                    test_on_original=self.test_on_original)

        return message
    
    def inference_text(self) -> str:
        assert os.path.isfile(
            self.model_path), f'File {self.model_path} does not exist.'

        output, message = inference_pipeline(archi=self.archi, 
                                             device=self.device,
                                            model_path=self.model_path,
                                            tokenizer_path = self.tokenizer_path,
                                            out_file_path = None,
                                            in_file_path=None,
                                            pretrained = self.pretrained,
                                            is_bulk = False,
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

        output, message = inference_pipeline(archi=self.archi, 
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
            
        output, message = inference_pipeline(archi=self.archi, 
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

        return message
    


