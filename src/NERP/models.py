"""
This section covers the interface for `NERP` models, that is implemented as its own Python class [NERP.models.NERP][]. The interface enables you to easily 
    - specify your own [NERP.models.NERP][] model
    - train it
    - evaluate it
    - use it to predict entities in new texts.
"""

import os
import shutil

from transformers import logging

from .inference import inference_pipeline
from .training import training_pipeline
from .args import parse_env_variables

logging.set_verbosity_error()


class NERP:
    def __init__(self, config: str = None) -> None:
        
        assert os.path.isfile(os.path.join(config)), f'Configuration file does not exist in {os.path.join(config)}.'
        self.torch_args, self.data_args, self.model_args, self.training_args, self.kfold_args, self.inference_args = parse_env_variables(config)

    def train(self) -> str:
        if os.path.exists(self.training_args.output_dir):
            try:
                shutil.rmtree(self.training_args.output_dir)
            except OSError as e:
                logger.error("Error: %s - %s." % (e.filename, e.strerror))
        
        training_pipeline(torch_args=self.torch_args,
                          data_args=self.data_args,
                          model_args=self.model_args,
                          training_args=self.training_args,
                          kfold_args=self.kfold_args)

    def predict(self) -> str:
        inference_pipeline(is_bulk=True,
                           text=None,
                           torch_args=self.torch_args,                
                           data_args=self.data_args,
                           model_args=self.model_args,
                           training_args=self.training_args,
                           inference_args=self.inference_args)

    
    def predict_text(self, text) -> str:
        assert text != None, "Please input a text!"

        predicted_tags = inference_pipeline(is_bulk=False,
                                            text=text,
                                            torch_args=self.torch_args,
                                            data_args=self.data_args,       
                                            model_args=self.model_args,
                                            training_args=self.training_args,
                                            inference_args=self.inference_args)

        return predicted_tags
