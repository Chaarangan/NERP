"""
This section covers the interface for `NERP` models, that is 
implemented as its own Python class [NERP.models.Trainer][].
"""

import pandas as pd
import numpy as np
import torch
import os
import sklearn.preprocessing

from sklearn.metrics import accuracy_score
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import List
from loguru import logger

from .networks import NERPNetwork, TransformerBiLSTM, TransformerBiLSTMCRF, TransformerCRF
from .predictions import predict, predict_text
from .performance import compute_f1_scores, flatten
from .trainer_helpher import train_model


class Trainer:
    """Trainer class for train, predict and evaluate performance
    """

    def __init__(self,
                 torch_args,
                 data_args,
                 model_args,
                 training_args,
                 transformer,
                 dataset_training: dict = None,
                 dataset_validation: dict = None,
                 network: torch.nn.Module = NERPNetwork) -> None:
        """Initialize model

        Args:
            transformer (str, optional): which pretrained 'huggingface' 
                transformer to use. 
            dataset_training (dict, optional): the training data. Must consist 
                of 'sentences': word-tokenized sentences and 'tags': corresponding 
                NER tags.
                Defaults to None, in which case the English CoNLL-2003 data set is used. 
            dataset_validation (dict, optional): the validation data. Must consist
                of 'sentences': word-tokenized sentences and 'tags': corresponding 
                NER tags.
                Defaults to None, in which case the English CoNLL-2003 data set 
                is used.
            network (torch.nn.module, optional): network to be trained. Defaults
                to a default generic `NERPNetwork`. Can be replaced with your own 
                customized network architecture. It must however take the same 
                arguments as `NERPNetwork`.

        """

        # set device automatically if not provided by user.
        if torch_args.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info("Device automatically set to: " + self.device.upper())
        else:
            self.device = torch_args.device
            logger.info("Device set to: " + torch_args.device.upper())
            
        self.transformer = transformer
        self.dataset_training = dataset_training
        self.dataset_validation = dataset_validation
        
        self.torch_args = torch_args
        self.data_args = data_args
        self.model_args = model_args
        self.training_args = training_args

        self.tokenizer_parameters = {"do_lower_case": model_args.do_lower_case}
        self.tag_outside = "O"
        tag_complete = [self.tag_outside] + data_args.tags
        
        # fit encoder to _all_ possible tags.
        self.tag_encoder = sklearn.preprocessing.LabelEncoder()
        self.tag_encoder.fit(tag_complete)
        self.transformer_model = AutoModel.from_pretrained(transformer)
        self.transformer_tokenizer = AutoTokenizer.from_pretrained(
            transformer, **tokenizer_parameters)
        self.transformer_config = AutoConfig.from_pretrained(transformer)

        if(model_args.archi == "baseline"):
            self.network = NERPNetwork(
                self.transformer_model, self.device, len(tag_complete), dropout=model_args.dropout, fixed_seed=torch_args.seed)
        elif (model_args.archi == "bilstm-crf"):
            self.network = TransformerBiLSTMCRF(
                self.transformer_model, self.device, len(tag_complete), dropout=model_args.dropout, fixed_seed=torch_args.seed)
        elif (model_args.archi == "crf"):
            self.network = TransformerCRF(
                self.transformer_model, self.device, len(tag_complete), dropout=model_args.dropout, fixed_seed=torch_args.seed)
        elif (model_args.archi == "bilstm"):
            self.network = TransformerBiLSTM(
                self.transformer_model, self.device, len(tag_complete), dropout=model_args.dropout, fixed_seed=torch_args.seed)

        self.network.to(self.device)
        self.train_losses = []
        self.valid_f1 = np.nan
        self.quantized = False
        self.halved = False

    def train(self) -> str:
        """Train Network

        Trains the network from the NERP model specification.

        Returns:
            str: a message saying if the model was trained succesfully.
            The network in the 'network' attribute is trained as a 
            side-effect. Training losses and validation loss are saved 
            in 'training_losses' and 'valid_loss' 
            attributes respectively as side-effects.
        """
        network, train_losses, valid_f1 = train_model(network=self.network,
                                                      tag_encoder=self.tag_encoder,
                                                      tag_outside=self.tag_outside,
                                                      transformer_tokenizer=self.transformer_tokenizer,
                                                      transformer_config=self.transformer_config,
                                                      dataset_training=self.dataset_training,
                                                      dataset_validation=self.dataset_validation,
                                                      device=self.device,
                                                      torch_args=torch_args,
                                                      data_args=data_args,
                                                      model_args=model_args,
                                                      training_args=training_args)

        # attach as attributes to class
        setattr(self, "network", network)
        setattr(self, "train_losses", train_losses)
        setattr(self, "valid_f1", valid_f1)

        return "Model trained successfully"

    def load_network_from_file(self, model_path="model.bin", tokenizer_path=None) -> str:
        """Load Pretrained NERP Network from file

        Loads weights for a pretrained NERP Network from file.

        Args:
            model_path (str, optional): Path for model file. 
                Defaults to "model.bin".

        Returns:
            str: message telling if weights for network were
            loaded succesfully.
        """
        # TODO: change assert to Raise.
        assert os.path.exists(
            model_path), "File does not exist. You can download network with download_network()"
        self.network.load_state_dict(torch.load(
            model_path, map_location=torch.device(self.device)))

        if(tokenizer_path != None):
            assert os.path.isdir(
                tokenizer_path), f'Folder {tokenizer_path} does not exist.'
            self.transformer_tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path)
        else:
            self.transformer_tokenizer = AutoTokenizer.from_pretrained(
                self.transformer, **self.tokenizer_parameters)

        self.network.device = self.device
        return f'Weights for network loaded from {model_path}'

    def save_network(self, output_dir: str = "./output_dir") -> None:
        """Save Weights of NERP Network

        Saves weights for a fine-tuned NERP Network to file.

        Args:
            model_path (str, optional): Path for model file. 
                Defaults to "model.bin".

        Returns:
            Nothing. Saves model to file as a side-effect.
        """
        if(not os.path.exists(output_dir)):
            os.makedirs(os.path.join(output_dir, "tokenizer"))

        torch.save(self.network.state_dict(),
                   os.path.join(output_dir, "model.bin"))
        self.transformer_tokenizer.save_pretrained(
            os.path.join(output_dir, "tokenizer"))
        logger.success(f"Network written to file {output_dir}")

    def predict(self, sentences: List[List[str]],
                return_confidence: bool = False,
                **kwargs) -> List[List[str]]:
        """Predict Named Entities in Word-Tokenized Sentences

        Predicts word-tokenized sentences with trained model.

        Args:
            sentences (List[List[str]]): word-tokenized sentences.
            kwargs: arbitrary keyword arguments. For instance
                'batch_size' and 'num_workers'.
            return_confidence (bool, optional): if True, return
                confidence scores for all predicted tokens. Defaults
                to False.

        Returns:
            List[List[str]]: Predicted tags for sentences - one
            predicted tag/entity per word token.
        """
        return predict(network=self.network,
                       sentences=sentences,
                       transformer_tokenizer=self.transformer_tokenizer,
                       transformer_config=self.transformer_config,
                       max_len=self.model_args.max_len,
                       device=self.device,
                       tag_encoder=self.tag_encoder,
                       tag_outside=self.tag_outside,
                       num_workers=self.model_args.num_workers,
                       return_confidence=return_confidence,
                       **kwargs)

    def predict_text(self, text: str,
                     return_confidence: bool = False, **kwargs) -> list:
        global tag_complete
        """Predict Named Entities in a Text

        Args:
            text (str): text to predict entities in.
            kwargs: arbitrary keyword arguments. For instance
                'batch_size' and 'num_workers'.
            return_confidence (bool, optional): if True, return
                confidence scores for all predicted tokens. Defaults
                to False.

        Returns:
            tuple: word-tokenized sentences and predicted 
            tags/entities.
        """
        return predict_text(network=self.network,
                            text=text,
                            transformer_tokenizer=self.transformer_tokenizer,
                            transformer_config=self.transformer_config,
                            max_len=self.model_args.max_len,
                            device=self.device,
                            tag_encoder=self.tag_encoder,
                            tag_outside=self.tag_outside,
                            num_workers=self.model_args.num_workers,
                            return_confidence=return_confidence,
                            **kwargs)

    def evaluate_performance(self, dataset: dict, **kwargs) -> pd.DataFrame:
        """Evaluate Performance

        Evaluates the performance of the model on an arbitrary
        data set.

        Args:
            dataset (dict): Data set that must consist of
                'sentences' and NER'tags'.
            kwargs: arbitrary keyword arguments for predict. For
                instance 'batch_size' and 'num_workers'.

        Returns:
            str: F1-scores, Precision and Recall. 
            int: accuracy, if return_accuracy is set to True.
        """
        tags_predicted = self.predict(dataset.get('sentences'),
                                      **kwargs)

        # compute F1 scores by entity type
        if(self.training_args.o_tag_cr == True):
            labels = ["O"] + self.data_args.tags
        else:
            labels = self.data_args.tags

        f1, y_true = compute_f1_scores(y_pred=tags_predicted,
                                       y_true=dataset.get('tags'),
                                       labels=labels)

        # compute and return accuracy if desired
        if self.training_args.return_accuracy:
            accuracy = accuracy_score(y_pred=flatten(tags_predicted),
                                      y_true=y_true)
            return {'f1': f1, 'accuracy': accuracy}

        return {"f1": f1}
