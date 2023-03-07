"""
This section covers functionality for helping trainer module
"""
import torch

from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from loguru import logger

from .datasets import create_dataloader
from .performance import compute_f1_scores, compute_loss
from .utils import enforce_reproducibility


def train(model, data_loader, optimizer, device, scheduler, n_tags):
    """One Iteration of Training"""

    model.train()
    final_loss = 0.0

    for dl in tqdm(data_loader, total=len(data_loader)):

        optimizer.zero_grad()
        outputs = model(**dl)
        loss = compute_loss(outputs,
                            dl.get('target_tags'),
                            dl.get('masks'),
                            device,
                            n_tags)
        loss.backward()
        optimizer.step()
        scheduler.step()
        final_loss += loss.item()

    # Return average loss
    return final_loss / len(data_loader)


def train_model(network,
                tag_encoder,
                tag_outside,
                transformer_tokenizer,
                transformer_config,
                dataset_training,
                dataset_validation,
                max_len,
                device,
                num_workers,
                tag_scheme,
                o_tag_cr,
                fixed_seed,
                train_batch_size,
                validation_batch_size,
                epochs,
                learning_rate,
                warmup_steps):

    if fixed_seed is not None:
        enforce_reproducibility(fixed_seed)

    # compute number of unique tags from encoder.
    n_tags = tag_encoder.classes_.shape[0]

    # prepare datasets for modelling by creating data readers and loaders
    dl_train = create_dataloader(sentences=dataset_training.get('sentences'),
                                 tags=dataset_training.get('tags'),
                                 transformer_tokenizer=transformer_tokenizer,
                                 transformer_config=transformer_config,
                                 max_len=max_len,
                                 batch_size=train_batch_size,
                                 tag_encoder=tag_encoder,
                                 tag_outside=tag_outside,
                                 num_workers=num_workers)
    dl_validate = create_dataloader(sentences=dataset_validation.get('sentences'),
                                    tags=dataset_validation.get('tags'),
                                    transformer_tokenizer=transformer_tokenizer,
                                    transformer_config=transformer_config,
                                    max_len=max_len,
                                    batch_size=validation_batch_size,
                                    tag_encoder=tag_encoder,
                                    tag_outside=tag_outside,
                                    num_workers=num_workers)

    optimizer_parameters = network.parameters()

    num_train_steps = int(len(dataset_training.get(
        'sentences')) / train_batch_size * epochs)

    optimizer = torch.optim.AdamW(optimizer_parameters, lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_steps
    )

    train_losses = []
    # best_valid_loss = np.inf
    best_valid_f1 = 0.0
    best_parameters = network.state_dict()

    for epoch in range(epochs):

        logger.debug('\n Epoch {:} / {:}'.format(epoch + 1, epochs))

        train_loss = train(network, dl_train, optimizer,
                           device, scheduler, n_tags)
        train_losses.append(train_loss)
        valid_loss, valid_tags_predicted = validate(
            network, dl_validate, device, n_tags, tag_encoder)
#        logger.info(valid_tags_predicted)
#        logger.info(dataset_validation.get('tags'))
        if(o_tag_cr == True):
            labels = ["O"] + tag_scheme
        else:
            labels = tag_scheme

        report, _ = compute_f1_scores(y_pred=valid_tags_predicted,
                                      y_true=dataset_validation.get('tags'),
                                      labels=labels)
        valid_f1 = report.split('\n')[len(labels) + 4].split()[3]
        valid_f1 = float(valid_f1)

        logger.warning(
            f"Train Loss = {train_loss} Valid Loss = {valid_loss} Valid F1 = {valid_f1}")

        if valid_f1 > best_valid_f1:
            best_parameters = network.state_dict()
            best_valid_f1 = valid_f1

    # return best model
    network.load_state_dict(best_parameters)

    return network, train_losses, best_valid_f1


def validate(model, data_loader, device, n_tags, tag_encoder):
    """One Iteration of Validation"""

    model.eval()
    final_loss = 0.0
    predictions = []
    for dl in tqdm(data_loader, total=len(data_loader)):

        outputs = model(**dl)
        loss = compute_loss(outputs,
                            dl.get('target_tags'),
                            dl.get('masks'),
                            device,
                            n_tags)
        final_loss += loss.item()
        for i in range(outputs.shape[0]):
            # extract prediction and transform.
            # find max by row.
            values, indices = outputs[i].max(dim=1)
            preds = tag_encoder.inverse_transform(indices.cpu().numpy())

            # subset predictions for original word tokens.
            preds = [prediction for prediction, offset in zip(
                preds.tolist(), dl.get('offsets')[i]) if offset]
            # Remove special tokens ('CLS' + 'SEP').
            preds = preds[1:-1]

            # make sure resulting predictions have same length as
            # original sentence.
            predictions.append(preds)

    # Return average loss.
    return (final_loss / len(data_loader), predictions)
