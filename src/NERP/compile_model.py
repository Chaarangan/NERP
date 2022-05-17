"""
This section covers functionality for compiling Named Entity 
Recognition models.
Author: Charangan Vasantharajan
"""
from NERDA_framework.models import NERDA
from NERP.prepare_data import prepare_data
from sklearn.model_selection import train_test_split

def compile_model(device, train_data, limit, tag_scheme, hyperparameters, tokenizer_parameters, max_len, dropout, pretrained, test_size):
    """
    Args:
        device (str, optional): the desired device to use for computation. 
                If not provided by the user, we take a guess.
        train_data (str, required): Train csv file path
        limit (int, optional): Limit the number of observations to be 
            returned from a given split. Defaults to None, which implies 
            that the entire data split is returned
        tag_scheme (List[str], optional): All available NER 
                tags for the given data set EXCLUDING the special outside tag, 
                that is handled separately.
        hyperparameters (dict, optional): Hyperparameters for the model
        tokenizer_parameters (dict, optional): Parameters for the tokenizer
        max_len (int, required): The maximum sentence length
        dropout (float, required): dropout probability
        pretrianed (str, optional): which pretrained 'huggingface' 
                transformer to use
        test_size (float, optional): train/test split ratio

    Returns:
        compiled model
    """
    data = prepare_data(limit, train_data)
    train_sentences, val_sentences, train_entities, val_entities = train_test_split(
        data["sentences"], data["tags"], test_size=test_size
    )

    training = {"sentences": train_sentences, "tags": train_entities}
    validation = {"sentences": val_sentences, "tags": val_entities}
    print("Train and Valid datasets are prepared!")

    tag_scheme = tag_scheme
    transformer = pretrained

    # hyperparameters for network
    dropout = dropout
    max_len = max_len
    # hyperparameters for training
    training_hyperparameters = hyperparameters

    model = NERDA(
        device=device,
        dataset_training=training,
        dataset_validation=validation,
        tag_scheme=tag_scheme,
        tag_outside='O',
        transformer=transformer,
        dropout=dropout,
        max_len=max_len,
        hyperparameters=training_hyperparameters,
        tokenizer_parameters=tokenizer_parameters
    )
    print("Model compiled!")
    return model
