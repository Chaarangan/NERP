from typing import List
from NERP.inference import inference_pipeline
from NERP.training import training_pipeline
import yaml

class NERP:
    def __init__(self,
                config: str = "env.yaml") -> None:

        stream = open(config, 'r')
        dictionary = yaml.load(stream, Loader=yaml.FullLoader)

        self.tag_scheme = dictionary["data"]["tag_scheme"]
        self.hyperparameters = dictionary["model"]["hyperparameters"]
        self.tokenizer_parameters = dictionary["model"]["tokenizer_parameters"]
        self.max_len = dictionary["model"]["max_len"]
        self.dropout = dictionary["model"]["dropout"]
        self.pretrained_models = dictionary["model"]["pretrained_models"]
        self.train_data = dictionary["data"]["train_data"]
        self.train_valid_split = dictionary["data"]["train_valid_split"]
        self.test_data = dictionary["data"]["test_data"]
        self.limit = dictionary["data"]["limit"]
        self.is_model_exists = dictionary["train"]["is_model_exists"]
        self.existing_model_path = dictionary["train"]["existing_model_path"]
        self.existing_tokenizer_path = dictionary["train"]["existing_tokenizer_path"]
        self.output_dir = dictionary["train"]["output_dir"]
        self.kfold = dictionary["kfold"]["splits"]
        self.seed = dictionary["kfold"]["seed"]
        self.pretrained = dictionary["inference"]["pretrained"]
        self.model_path = dictionary["inference"]["model_path"]
        self.tokenizer_path = dictionary["inference"]["tokenizer_path"]
        self.in_file_path = dictionary["inference"]["bulk"]["in_file_path"]
        self.out_file_path = dictionary["inference"]["bulk"]["out_file_path"]
        self.text = dictionary["inference"]["individual"]["text"]
    
    def train(self) -> str:
        message = training_pipeline(train_data = self.train_data,
                                    test_data = self.test_data,
                                    existing_model_path=None,
                                    existing_tokenizer_path=None,
                                    tag_scheme=self.tag_scheme,
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
                                    seed=42)
        
        return message
    
    def train_after_load_network(self) -> str:
        message = training_pipeline(train_data=self.train_data,
                                    test_data=self.test_data,
                                    existing_model_path=self.existing_model_path,
                                    existing_tokenizer_path=self.existing_tokenizer_path,
                                    tag_scheme=self.tag_scheme,
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
                                    seed=42)

        return message

    def train_with_kfold(self) -> str:
        message = training_pipeline(train_data=self.train_data,
                                    test_data=self.test_data,
                                    existing_model_path=None,
                                    existing_tokenizer_path=None,
                                    tag_scheme=self.tag_scheme,
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
                                    seed=self.seed)

        return message
    
    def train_with_kfold_after_load_network(self) -> str:
        message = training_pipeline(train_data=self.train_data,
                                    test_data=self.test_data,
                                    existing_model_path=self.existing_model_path,
                                    existing_tokenizer_path=self.existing_tokenizer_path,
                                    tag_scheme=self.tag_scheme,
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
                                    seed=self.seed)

        return message
    
    def inference_text(self) -> str:
        output, message = inference_pipeline(model_path = self.model_path,
                                    tokenizer_path = self.tokenizer_path,
                                    out_file_path = None,
                                    in_file_path=None,
                                    pretrained = self.pretrained,
                                    is_bulk = False,
                                    text=self.text,
                                    tag_scheme=self.tag_scheme, 
                                    hyperparameters=self.hyperparameters,
                                    tokenizer_parameters=self.tokenizer_parameters,
                                    max_len=self.max_len)
        
        return output, message
    
    def inference_bulk(self) -> str:
        output, message = inference_pipeline(model_path=self.model_path,
                                             tokenizer_path=self.tokenizer_path,
                                             out_file_path=self.in_file_path,
                                             in_file_path=self.in_file_path,
                                             pretrained=self.pretrained,
                                             is_bulk=True,
                                             text=None,
                                             tag_scheme=self.tag_scheme,
                                             hyperparameters=self.hyperparameters,
                                             tokenizer_parameters=self.tokenizer_parameters,
                                             max_len=self.max_len)

        return message
    


