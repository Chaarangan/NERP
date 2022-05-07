from typing import List
from NERP.inference import inference_pipeline
from NERP.training import training_pipeline

class NERP:
    def __init__(self,
                train_csv_file: str = None,
                test_csv_file: str = None,
                limit: int = 0,
                tag_scheme: List[str] = [
                     'B-PER',
                     'I-PER',
                     'B-ORG',
                     'I-ORG',
                     'B-LOC',
                     'I-LOC',
                     'B-MISC',
                     'I-MISC'
                ],
                hyperparameters: dict = {'epochs': 5,
                                          'warmup_steps': 500,
                                          'train_batch_size': 64,
                                          'learning_rate': 0.0001},
                tokenizer_parameters: dict = {'do_lower_case': True},
                max_len: int = 128,
                dropout: float = 0.1,
                pretrained_models: List[str] = [
                    'roberta-base'
                ], 
                test_size: float = 0.2,
                isModelExists: bool = False,
                existing_model_path: str = None,
                output_dir: str = 'models/',
                kfold: int = 0,
                seed: int = 42) -> None:

        self.tag_scheme = tag_scheme
        self.hyperparameters = hyperparameters
        self.tokenizer_parameters = tokenizer_parameters
        self.max_len = max_len
        self.dropout = dropout
        self.pretrained_models = pretrained_models
    
    def train(self, train_data: str = None,
              test_data: str = None,
              limit: int = 0,
              test_size: float = 0.2,
              is_model_exists: bool = False,
              existing_model_path: str = None,
              output_dir: str = "models/",
              kfold: int = 0,
              seed: int = 42) -> str:
        message = training_pipeline(train_data,
                                    test_data,
                                    limit,
                                    test_size,
                                    is_model_exists,
                                    existing_model_path,
                                    output_dir,
                                    kfold,
                                    seed,
                                    tag_scheme = self.tag_scheme,
                                    hyperparameters = self.hyperparameters,
                                    tokenizer_parameters = self.tokenizer_parameters,
                                    max_len = self.max_len,
                                    dropout = self.dropout,
                                    pretrained_models=self.pretrained_models)
        
        return message
    
    def inference(self, pretrained: str = None,
            model_path: str = None,
            tokenizer_path: str = None,
            out_file_path: str = "output.csv",
            is_bulk: bool = False,
            in_file_path: str = None,
            text: str = "Sample NERP input") -> str:
        message = inference_pipeline(pretrained,
                                    model_path,
                                    tokenizer_path,
                                    out_file_path,
                                    is_bulk,
                                    in_file_path,
                                    text, 
                                    tag_scheme=self.tag_scheme, 
                                    hyperparameters=self.hyperparameters,
                                    tokenizer_parameters=self.tokenizer_parameters,
                                    max_len=self.max_len)
        
        return message
    


