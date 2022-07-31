"""
This section covers functionality for compiling Named Entity
Recognition models.
Author: Charangan Vasantharajan
"""
from NERP.NERDA_framework.models import NERDA
from sklearn.model_selection import train_test_split
from NERP.prepare_data import prepare_data

def compile_model(archi, device, train_data, valid_data, limit, tag_scheme, o_tag_cr, hyperparameters, tokenizer_parameters, max_len, dropout, pretrained, test_size):
    """
    Args:
        archi (str, optional): the desired architecture for the model
        device (str, optional): the desired device to use for computation. 
                If not provided by the user, we take a guess.
        train_data (str, required): Train csv file path
        valid_data (str, optional): Valid csv file path
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

    if (valid_data == None):
        print("Valid data is None and created from train data!")
        data = prepare_data(limit, train_data)
        train_sentences, val_sentences, train_entities, val_entities = train_test_split(
            data["sentences"], data["tags"], test_size=test_size
        )
        training = {"sentences": train_sentences, "tags": train_entities}
        validation = {"sentences": val_sentences, "tags": val_entities}

        print("Training: ({a}, {b})".format(
            a=str(len(training["sentences"])), b=str(len(training["tags"]))))
        print("Validation: ({a}, {b}".format(
            a=str(len(validation["sentences"])), b=str(len(validation["tags"]))))

    else:
        print("Valid data exists!")
        training = prepare_data(limit, train_data)
        validation = prepare_data(limit, valid_data)

        print("Training: ({a}, {b})".format(
            a=str(len(training["sentences"])), b=str(len(training["tags"]))))
        print("Validation: ({a}, {b}".format(
            a=str(len(validation["sentences"])), b=str(len(validation["tags"]))))

    print("Train and Valid datasets are prepared!")

    tag_scheme = tag_scheme
    transformer = pretrained

    # hyperparameters for network
    dropout = dropout
    max_len = max_len
    # hyperparameters for training
    training_hyperparameters = hyperparameters

    model = NERDA(
        archi=archi,
        device=device,
        dataset_training=training,
        dataset_validation=validation,
        tag_scheme=tag_scheme,
        tag_outside='O',
        o_tag_cr=o_tag_cr,
        transformer=transformer,
        dropout=dropout,
        max_len=max_len,
        hyperparameters=training_hyperparameters,
        tokenizer_parameters=tokenizer_parameters
    )
    print("Model compiled with {archi} architecture!".format(archi=archi))
    return model
