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
    """models

    A model object containing a complete model configuration.
    The model can be trained with the `train` method. Afterwards
    new observations can be predicted with the `predict` and
    `predict_text` methods. The performance of the model can be
    evaluated on a set of new observations with the 
    `evaluate_performance` method.
    """

    def __init__(self,
                 archi: str = "baseline",
                 device: str = None,
                 dataset_training: dict = None,
                 dataset_validation: dict = None,
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
                 o_tag_cr: bool = True,
                 hyperparameters: dict = {"epochs": 5,
                                          "warmup_steps": 500,
                                          "batch_size": {
                                              "train": 64,
                                              "valid": 8
                                          },
                                          "lr": 0.0001,
                                          "seed": 42},
                 tokenizer_parameters: dict = {'do_lower_case': True},
                 max_len: int = 128,
                 dropout: float = 0.1,
                 transformer: str = 'bert-base-multilingual-uncased',
                 num_workers: int = 1,
                 tag_outside: str = 'O',
                 network: torch.nn.Module = NERPNetwork) -> None:
        """Initialize model

        Args:
            transformer (str, optional): which pretrained 'huggingface' 
                transformer to use. 
            device (str, optional): the desired device to use for computation. 
                If not provided by the user, we take a guess.
            tag_scheme (List[str], optional): All available NER 
                tags for the given data set EXCLUDING the special outside tag, 
                that is handled separately.
            tag_outside (str, optional): the value of the special outside tag. 
                Defaults to 'O'.
            dataset_training (dict, optional): the training data. Must consist 
                of 'sentences': word-tokenized sentences and 'tags': corresponding 
                NER tags.
                Defaults to None, in which case the English CoNLL-2003 data set is used. 
            dataset_validation (dict, optional): the validation data. Must consist
                of 'sentences': word-tokenized sentences and 'tags': corresponding 
                NER tags.
                Defaults to None, in which case the English CoNLL-2003 data set 
                is used.
            max_len (int, optional): the maximum sentence length (number of 
                tokens after applying the transformer tokenizer) for the transformer. 
                Sentences are truncated accordingly. Look at your data to get an 
                impression of, what could be a meaningful setting. Also be aware 
                that many transformers have a maximum accepted length. Defaults 
                to 128. 
            network (torch.nn.module, optional): network to be trained. Defaults
                to a default generic `NERPNetwork`. Can be replaced with your own 
                customized network architecture. It must however take the same 
                arguments as `NERPNetwork`.
            dropout (float, optional): dropout probability. Defaults to 0.1.
            hyperparameters (dict, optional): Hyperparameters for the model. Defaults
                to {'epochs' : 3, 'warmup_steps' : 500, 'train_batch_size': 16, 
                'learning_rate': 0.0001}.
            tokenizer_parameters (dict, optional): parameters for the transformer 
                tokenizer. Defaults to {'do_lower_case' : True}.
            validation_batch_size (int, optional): batch size for validation. Defaults
                to 8.
            num_workers (int, optional): number of workers for data loader.
        """

        # set device automatically if not provided by user.
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info("Device automatically set to: " + self.device.upper())
        else:
            self.device = device
            logger.info("Device set to: " + device.upper())
        self.transformer = transformer
        self.dataset_training = dataset_training
        self.dataset_validation = dataset_validation
        self.hyperparameters = hyperparameters
        self.tokenizer_parameters = tokenizer_parameters
        self.tag_outside = tag_outside
        self.tag_scheme = tag_scheme
        tag_complete = [tag_outside] + tag_scheme
        self.o_tag_cr = o_tag_cr
        # fit encoder to _all_ possible tags.
        self.max_len = max_len
        self.tag_encoder = sklearn.preprocessing.LabelEncoder()
        self.tag_encoder.fit(tag_complete)
        self.transformer_model = AutoModel.from_pretrained(transformer)
        self.transformer_tokenizer = AutoTokenizer.from_pretrained(
            transformer, **tokenizer_parameters)
        self.transformer_config = AutoConfig.from_pretrained(transformer)

        if(archi == "baseline"):
            self.network = NERPNetwork(
                self.transformer_model, self.device, len(tag_complete), dropout=dropout, fixed_seed=hyperparameters['seed'])
        elif (archi == "bilstm-crf"):
            self.network = TransformerBiLSTMCRF(
                self.transformer_model, self.device, len(tag_complete), dropout=dropout, fixed_seed=hyperparameters['seed'])
        elif (archi == "crf"):
            self.network = TransformerCRF(
                self.transformer_model, self.device, len(tag_complete), dropout=dropout, fixed_seed=hyperparameters['seed'])
        elif (archi == "bilstm"):
            self.network = TransformerBiLSTM(
                self.transformer_model, self.device, len(tag_complete), dropout=dropout, fixed_seed=hyperparameters.seed)

        self.network.to(self.device)
        self.num_workers = num_workers
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
                                                      max_len=self.max_len,
                                                      device=self.device,
                                                      num_workers=self.num_workers,
                                                      tag_scheme=self.tag_scheme,
                                                      o_tag_cr=self.o_tag_cr,
                                                      fixed_seed=self.hyperparameters["seed"],
                                                      train_batch_size=self.hyperparameters["batch_size"]["train"],
                                                      validation_batch_size=self.hyperparameters["batch_size"]["valid"],
                                                      epochs=self.hyperparameters["epochs"],
                                                      learning_rate=self.hyperparameters["lr"],
                                                      warmup_steps=self.hyperparameters["warmup_steps"])

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
                       max_len=self.max_len,
                       device=self.device,
                       tag_encoder=self.tag_encoder,
                       tag_outside=self.tag_outside,
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
                            max_len=self.max_len,
                            device=self.device,
                            tag_encoder=self.tag_encoder,
                            tag_outside=self.tag_outside,
                            return_confidence=return_confidence,
                            **kwargs)

    def evaluate_performance(self, dataset: dict,
                             return_accuracy: bool = False,
                             **kwargs) -> pd.DataFrame:
        """Evaluate Performance

        Evaluates the performance of the model on an arbitrary
        data set.

        Args:
            dataset (dict): Data set that must consist of
                'sentences' and NER'tags'.
            kwargs: arbitrary keyword arguments for predict. For
                instance 'batch_size' and 'num_workers'.
            return_accuracy (bool): Return accuracy
                as well? Defaults to False.


        Returns:
            str: F1-scores, Precision and Recall. 
            int: accuracy, if return_accuracy is set to True.
        """
        tags_predicted = self.predict(dataset.get('sentences'),
                                      **kwargs)

        # compute F1 scores by entity type
        if(self.o_tag_cr == True):
            labels = ["O"] + self.tag_scheme
        else:
            labels = self.tag_scheme

        f1, y_true = compute_f1_scores(y_pred=tags_predicted,
                                       y_true=dataset.get('tags'),
                                       labels=labels)

        # compute and return accuracy if desired
        if return_accuracy:
            accuracy = accuracy_score(y_pred=flatten(tags_predicted),
                                      y_true=y_true)
            return {'f1': f1, 'accuracy': accuracy}

        return {"f1": f1}
