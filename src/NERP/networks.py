"""
This section covers `torch` networks for `NERP`
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig
# from TorchCRF import CRF
from torchcrf import CRF


from .utils import match_kwargs
from .utils import enforce_reproducibility

log_soft = F.log_softmax

class NERPNetwork(nn.Module):
    """A Generic Network for NERP models.

    Can be replaced with a custom user-defined network with 
    the restriction, that it must take the same arguments.
    """

    def __init__(self, transformer: nn.Module, device: str, n_tags: int, dropout: float = 0.1, seed=42) -> None:
        """Initialize a NERP Network

        Args:
            transformer (nn.Module): huggingface `torch` transformer.
            device (str): Computational device.
            n_tags (int): Number of unique entity tags (incl. outside tag)
            dropout (float, optional): Dropout probability. Defaults to 0.1.
        """
        super(NERPNetwork, self).__init__()
        enforce_reproducibility(seed)

        # extract AutoConfig, from which relevant parameters can be extracted.
        transformer_config = AutoConfig.from_pretrained(
            transformer.name_or_path)
        hidden_size = transformer_config.hidden_size

        self.transformer = transformer
        self.dropout = nn.Dropout(dropout)
        self.device = device
        
        self.tags = nn.Linear(hidden_size, n_tags)
        

    # NOTE: 'offsets 'are not used in model as-is, but they are expected as output
    # down-stream. So _DON'T_ remove! :)
    def forward(self,
                input_ids: torch.Tensor,
                masks: torch.Tensor,
                token_type_ids: torch.Tensor,
                target_tags: torch.Tensor,
                offsets: torch.Tensor) -> torch.Tensor:
        """Model Forward Iteration

        Args:
            input_ids (torch.Tensor): Input IDs.
            masks (torch.Tensor): Attention Masks.
            token_type_ids (torch.Tensor): Token Type IDs.
            target_tags (torch.Tensor): Target tags. Are not used 
                in model as-is, but they are expected downstream,
                so they can not be left out.
            offsets (torch.Tensor): Offsets to keep track of original
                words. Are not used in model as-is, but they are 
                expected as down-stream, so they can not be left out.

        Returns:
            torch.Tensor: predicted values.
        """

        # TODO: can be improved with ** and move everything to device in a
        # single step.
        transformer_inputs = {
            'input_ids': input_ids.to(self.device),
            'masks': masks.to(self.device),
            'token_type_ids': token_type_ids.to(self.device)
        }

        # match args with transformer
        transformer_inputs = match_kwargs(
            self.transformer.forward, **transformer_inputs)
        outputs = self.transformer(**transformer_inputs)[0]
        outputs = self.dropout(outputs)
        outputs = self.tags(outputs)

        return outputs


class TransformerCRF(nn.Module):
    """Transformer + CRF
    """

    def __init__(self, transformer: nn.Module, device: str, n_tags: int, dropout: float = 0.1, seed=42) -> None:
        """Initialize a NERP Network

        Args:
            transformer (nn.Module): huggingface `torch` transformer.
            device (str): Computational device.
            n_tags (int): Number of unique entity tags (incl. outside tag)
            dropout (float, optional): Dropout probability. Defaults to 0.1.
        """
        super(TransformerCRF, self).__init__()
        enforce_reproducibility(seed)

        # extract AutoConfig, from which relevant parameters can be extracted.
        transformer_config = AutoConfig.from_pretrained(
            transformer.name_or_path)
        hidden_size = transformer_config.hidden_size

        self.transformer = transformer
        self.dropout = nn.Dropout(dropout)
        self.device = device

        self.classifier = nn.Linear(hidden_size, n_tags)
        #self.crf = CRF(n_tags) 
        self.crf = CRF(n_tags, batch_first=True)

    # NOTE: 'offsets 'are not used in model as-is, but they are expected as output
    # down-stream. So _DON'T_ remove! :)
    def forward(self,
                input_ids: torch.Tensor,
                masks: torch.Tensor,
                token_type_ids: torch.Tensor,
                target_tags: torch.Tensor,
                offsets: torch.Tensor) -> torch.Tensor:
        """Model Forward Iteration

        Args:
            input_ids (torch.Tensor): Input IDs.
            masks (torch.Tensor): Attention Masks.
            token_type_ids (torch.Tensor): Token Type IDs.
            target_tags (torch.Tensor): Target tags. Are not used 
                in model as-is, but they are expected downstream,
                so they can not be left out.
            offsets (torch.Tensor): Offsets to keep track of original
                words. Are not used in model as-is, but they are 
                expected as down-stream, so they can not be left out.

        Returns:
            torch.Tensor: predicted values.
        """

        # TODO: can be improved with ** and move everything to device in a
        # single step.
        transformer_inputs = {
            'input_ids': input_ids.to(self.device),
            'masks': masks.to(self.device),
            'token_type_ids': token_type_ids.to(self.device)
        }

        # match args with transformer
        transformer_inputs = match_kwargs(
            self.transformer.forward, **transformer_inputs)

        padded_sequence_output = self.transformer(**transformer_inputs)[0]
        padded_sequence_output = self.dropout(padded_sequence_output)
        logits = self.classifier(padded_sequence_output)

        # outputs = (logits,)
        # target_tags = target_tags.to(self.device)
        # if target_tags is not None:
        #     loss_mask = target_tags.gt(-1)
        #     loss = self.crf(logits, target_tags, loss_mask) * (-1)
        #     outputs = (loss,) + outputs

        # # contain: (loss), scores
        # return outputs[1]
        
        target_tags = target_tags.to(self.device)
        if target_tags is not None:
            loss = -self.crf(log_soft(logits, 2), target_tags,
                             mask=transformer_inputs.masks.type(torch.uint8), reduction='mean')
            prediction = self.crf.decode(
                logits, mask=transformer_inputs.masks.type(torch.uint8))
            outputs = (loss,) + prediction
        
        return outputs[1]
        


class TransformerBiLSTMCRF(nn.Module):
    """Transformer + BiLSTM + CRF
    """

    def __init__(self, transformer: nn.Module, device: str, n_tags: int, dropout: float = 0.1, seed=42) -> None:
        """Initialize a NERP Network

        Args:
            transformer (nn.Module): huggingface `torch` transformer.
            device (str): Computational device.
            n_tags (int): Number of unique entity tags (incl. outside tag)
            dropout (float, optional): Dropout probability. Defaults to 0.1.
        """
        super(TransformerBiLSTMCRF, self).__init__()
        enforce_reproducibility(seed)

        # extract AutoConfig, from which relevant parameters can be extracted.
        transformer_config = AutoConfig.from_pretrained(
            transformer.name_or_path)
        hidden_size = transformer_config.hidden_size

        self.transformer = transformer
        self.dropout = nn.Dropout(dropout)
        self.device = device

        self.bilstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            batch_first=True,
            num_layers=2,
            dropout=dropout,
            bidirectional=True
        )

        self.classifier = nn.Linear(hidden_size, n_tags)
        self.crf = CRF(n_tags)

    # NOTE: 'offsets 'are not used in model as-is, but they are expected as output
    # down-stream. So _DON'T_ remove! :)
    def forward(self,
                input_ids: torch.Tensor,
                masks: torch.Tensor,
                token_type_ids: torch.Tensor,
                target_tags: torch.Tensor,
                offsets: torch.Tensor) -> torch.Tensor:
        """Model Forward Iteration

        Args:
            input_ids (torch.Tensor): Input IDs.
            masks (torch.Tensor): Attention Masks.
            token_type_ids (torch.Tensor): Token Type IDs.
            target_tags (torch.Tensor): Target tags. Are not used 
                in model as-is, but they are expected downstream,
                so they can not be left out.
            offsets (torch.Tensor): Offsets to keep track of original
                words. Are not used in model as-is, but they are 
                expected as down-stream, so they can not be left out.

        Returns:
            torch.Tensor: predicted values.
        """

        # TODO: can be improved with ** and move everything to device in a
        # single step.
        transformer_inputs = {
            'input_ids': input_ids.to(self.device),
            'masks': masks.to(self.device),
            'token_type_ids': token_type_ids.to(self.device)
        }

        # match args with transformer
        transformer_inputs = match_kwargs(
            self.transformer.forward, **transformer_inputs)

        padded_sequence_output = self.transformer(**transformer_inputs)[0]
        padded_sequence_output = self.dropout(padded_sequence_output)
        lstm_output, _ = self.bilstm(padded_sequence_output)
        logits = self.classifier(lstm_output)

        outputs = (logits,)
        target_tags = target_tags.to(self.device)
        if target_tags is not None:
            loss_mask = target_tags.gt(-1)
            loss = self.crf(logits, target_tags, loss_mask) * (-1)
            outputs = (loss,) + outputs

        # contain: (loss), scores
        return outputs[1]


class TransformerBiLSTM(nn.Module):
    """Transformer + BiLSTM
    """

    def __init__(self, transformer: nn.Module, device: str, n_tags: int, dropout: float = 0.1, seed=42) -> None:
        """Initialize a NERP Network

        Args:
            transformer (nn.Module): huggingface `torch` transformer.
            device (str): Computational device.
            n_tags (int): Number of unique entity tags (incl. outside tag)
            dropout (float, optional): Dropout probability. Defaults to 0.1.
        """
        super(TransformerBiLSTM, self).__init__()
        enforce_reproducibility(seed)

        # extract AutoConfig, from which relevant parameters can be extracted.
        transformer_config = AutoConfig.from_pretrained(
            transformer.name_or_path)
        hidden_size = transformer_config.hidden_size

        self.transformer = transformer
        self.dropout = nn.Dropout(dropout)
        self.device = device

        self.bilstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            batch_first=True,
            num_layers=2,
            dropout=dropout,
            bidirectional=True
        )

        self.classifier = nn.Linear(hidden_size, n_tags)

    # NOTE: 'offsets 'are not used in model as-is, but they are expected as output
    # down-stream. So _DON'T_ remove! :)
    def forward(self,
                input_ids: torch.Tensor,
                masks: torch.Tensor,
                token_type_ids: torch.Tensor,
                target_tags: torch.Tensor,
                offsets: torch.Tensor) -> torch.Tensor:
        """Model Forward Iteration

        Args:
            input_ids (torch.Tensor): Input IDs.
            masks (torch.Tensor): Attention Masks.
            token_type_ids (torch.Tensor): Token Type IDs.
            target_tags (torch.Tensor): Target tags. Are not used 
                in model as-is, but they are expected downstream,
                so they can not be left out.
            offsets (torch.Tensor): Offsets to keep track of original
                words. Are not used in model as-is, but they are 
                expected as down-stream, so they can not be left out.

        Returns:
            torch.Tensor: predicted values.
        """

        # TODO: can be improved with ** and move everything to device in a
        # single step.
        transformer_inputs = {
            'input_ids': input_ids.to(self.device),
            'masks': masks.to(self.device),
            'token_type_ids': token_type_ids.to(self.device)
        }

        # match args with transformer
        transformer_inputs = match_kwargs(
            self.transformer.forward, **transformer_inputs)

        padded_sequence_output = self.transformer(**transformer_inputs)[0]
        padded_sequence_output = self.dropout(padded_sequence_output)
        lstm_output, _ = self.bilstm(padded_sequence_output)
        outputs = self.classifier(lstm_output)

        # contain: (loss), scores
        return outputs
