import numpy as np
from .preprocessing import create_dataloader
from transformers import get_linear_schedule_with_warmup
import random
import torch
from tqdm import tqdm
from .performance import compute_f1_scores

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
            preds = [prediction for prediction, offset in zip(preds.tolist(), dl.get('offsets')[i]) if offset]        
            # Remove special tokens ('CLS' + 'SEP').
            preds = preds[1:-1]
        
            # make sure resulting predictions have same length as
            # original sentence.
            predictions.append(preds)

    
    # Return average loss.
    return (final_loss / len(data_loader), predictions)

def compute_loss(preds, target_tags, masks, device, n_tags):
    
    # initialize loss function.
    lfn = torch.nn.CrossEntropyLoss()

    # Compute active loss to not compute loss of paddings
    active_loss = masks.view(-1) == 1

    active_logits = preds.view(-1, n_tags)
    active_labels = torch.where(
        active_loss,
        target_tags.view(-1),
        torch.tensor(lfn.ignore_index).type_as(target_tags)
    )

    active_labels = torch.as_tensor(active_labels, device = torch.device(device), dtype = torch.long)
    
    # Only compute loss on actual token predictions
    loss = lfn(active_logits, active_labels)

    return loss

def enforce_reproducibility(seed = 42) -> None:
    """Enforce Reproducibity

    Enforces reproducibility of models to the furthest 
    possible extent. This is done by setting fixed seeds for
    random number generation etcetera. 

    For atomic operations there is currently no simple way to
    enforce determinism, as the order of parallel operations
    is not known.

    Args:
        seed (int, optional): Fixed seed. Defaults to 42.  
    """
    # Sets seed manually for both CPU and CUDA
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # System based
    random.seed(seed)
    np.random.seed(seed)

def train_model(network,
                tag_encoder,
                tag_outside,
                transformer_tokenizer,
                transformer_config,
                dataset_training, 
                dataset_validation, 
                max_len = 128,
                train_batch_size = 16,
                validation_batch_size = 8,
                epochs = 5,
                warmup_steps = 0,
                learning_rate = 5e-5,
                device = None,
                fixed_seed = 42,
                num_workers = 1,
                tag_scheme = ['B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC'],
                o_tag_cr = True):
    
    if fixed_seed is not None:
        enforce_reproducibility(fixed_seed)
    
    # compute number of unique tags from encoder.
    n_tags = tag_encoder.classes_.shape[0]

    # prepare datasets for modelling by creating data readers and loaders
    dl_train = create_dataloader(sentences = dataset_training.get('sentences'),
                                 tags = dataset_training.get('tags'), 
                                 transformer_tokenizer = transformer_tokenizer, 
                                 transformer_config = transformer_config,
                                 max_len = max_len, 
                                 batch_size = train_batch_size, 
                                 tag_encoder = tag_encoder,
                                 tag_outside = tag_outside,
                                 num_workers = num_workers)
    dl_validate = create_dataloader(sentences = dataset_validation.get('sentences'), 
                                    tags = dataset_validation.get('tags'),
                                    transformer_tokenizer = transformer_tokenizer,
                                    transformer_config = transformer_config, 
                                    max_len = max_len, 
                                    batch_size = validation_batch_size, 
                                    tag_encoder = tag_encoder,
                                    tag_outside = tag_outside,
                                    num_workers = num_workers)

    optimizer_parameters = network.parameters()

    num_train_steps = int(len(dataset_training.get('sentences')) / train_batch_size * epochs)
    
    optimizer = torch.optim.AdamW(optimizer_parameters, lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps = warmup_steps, num_training_steps = num_train_steps
    )

    train_losses = []
    best_parameters = network.state_dict()
    # best_valid_loss = np.inf
    best_valid_f1 = 0.0

    for epoch in range(epochs):
        
        print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))

        train_loss = train(network, dl_train, optimizer, device, scheduler, n_tags)
        train_losses.append(train_loss)
        valid_loss, valid_tags_predicted = validate(network, dl_validate, device, n_tags, tag_encoder)
        
        if(o_tag_cr == True):
            labels = ["O"] + tag_scheme
        else:
            labels = tag_scheme
            
        report, _ = compute_f1_scores(y_pred=valid_tags_predicted,
                               y_true=dataset_validation.get('tags'),
                               labels=labels)

        f1_row = report.split('\n')[len(labels) + 3].split()
        valid_f1 = f1_row[1] if f1_row[0] == 'accuracy' else f1_row[3]
        valid_f1 = float(valid_f1)

        print(f"Train Loss = {train_loss} Valid Loss = {valid_loss} Valid F1 = {valid_f1}")

        # if valid_loss < best_valid_loss:
        #     best_parameters = network.state_dict()            
        #     best_valid_loss = valid_loss

        if valid_f1 > best_valid_f1:
            best_parameters = network.state_dict() 
            best_valid_f1 = valid_f1           

    # return best model
    network.load_state_dict(best_parameters)

    return network, train_losses, best_valid_f1
