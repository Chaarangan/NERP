## NERP - Pipeline for training NER models

### **Config**

The user interface consists of only one file config as a YAML.
Change it to create the desired configuration.

Sample ```env.yaml``` file
```yaml
torch:
  device: "cuda"
data:
  train_data: 'data/train.csv'
  train_valid_split: 0.2
  test_data: 'data/test.csv'
  limit: 10
  tag_scheme: ['B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

model: 
  max_len: 128 
  dropout: 0.1
  hyperparameters:
    epochs: 1
    warmup_steps: 500
    train_batch_size: 64
    learning_rate: 0.0001
  tokenizer_parameters: 
    do_lower_case: True
  pretrained_models: 
    - roberta-base

train:
  is_model_exists: True
  existing_model_path: "roberta-base/model.bin"
  existing_tokenizer_path: "roberta-base/tokenizer"
  output_dir: "output/"

kfold: 
  splits: 2
  seed: 42

inference:
  max_len: 128 
  pretrained: "roberta-base"
  model_path: "roberta-base/model.bin"
  tokenizer_path: "roberta-base/tokenizer"
  bulk:
    in_file_path: "data/test.csv"
    out_file_path: "data/output.csv"
  individual:
    text: "Hello from NERP"
```


#### Training Parameters
| Parameters | Description | Default |
| ------------- | ------------- | ------------- |
| device |device (str, optional): the desired device to use for computation. If not provided by the user, we take a guess. | |
| train_data | path to training csv file | |
| train_valid_split | train/valid split ratio | 0.2 |
| test_data | path to testing csv file | |
| limit | Limit the number of observations to be returned from a given split. Defaults to None, which implies that the entire data split is returned. (it shoud be a ```int```) | 0 (whole data) |
| tag_scheme | All available NER tags for the given data set EXCLUDING the special outside tag, that is handled separately | |
| max_len | the maximum sentence length (number of tokens after applying the transformer tokenizer) | 128 |
| dropout | dropout probability (float) | 0.1 |
| epochs | number of epochs (int) | 5 |
| warmup_steps | number of learning rate warmup steps (int) | 500 |
| train_batch_size | batch Size for DataLoader (int) | 64 |
| learning_rate | learning rate (float) | 0.0001 |
| tokenizer_parameters | list of hyperparameters for tokenizer | do_lower_case: True |
| pretrained_models | 'huggingface' transformer model | roberta-base |
| is_model_exists | ```True``` for loading existing transformer model's weights otherwise ```False``` | False |
| existing_model_path | model derived from the transformer | |
| existing_tokenizer_path | tokenizer derived from the transformer | |
| output_dir | path to output directory | models/ |
| kfold | number of splits | 0 |
| seed | random state value for k-fold | 42 |

#### Inference Parameters
| Parameters | Description |
| ------------- | ------------- |
| max_len | the maximum sentence length (number of tokens after applying the transformer tokenizer) | 128 |
| pretrained | 'huggingface' transformer model | roberta-base |
| model_path | path to trained model | |
| tokenizer_path | path to saved tokenizer folder | |
| tag_scheme | All available NER tags for the given data set EXCLUDING the special outside tag, that is handled separately | |
| in_file_path | path to inference file otherwise leave it as empty | |
| out_file_path | path to output file if the input is file, otherwise leave it as empty | |
| text | sample inference text for individual prediction if **is_bulk** ```False``` | |
---

### **Data Format**

Pipeline works with csv file containing separated tokens and labels on each line. Sentences can be found in the `Sentence #` column. Labels should already be in necessary format, e.g. IO, BIO, BILUO, ... The csv file must contains last three columns as same as below.

, | Unnamed: 0 | Sentence # | Word | Tag 
--- | --- | --- | --- | ---
0 | 0 | Sentence: 0 | i | o
1 | 1 | Sentence: 0 | was | O
2 | 2 | Sentence: 0 | at | O
3 | 3 | Sentence: 0 | h.w. | B-place
4 | 4 | Sentence: 0 | holdings | I-place
5 | 5 | Sentence: 0 | pte | I-place

---

### **Output**

After training the model, the pipeline will return the following files in the output directory:

* model.bin - pytorch NER model
* tokenizer files
* classification-report.csv - logging file

---

### **Models**

All huggingface transformer-based models are allowed.

---

### Usage
#### Environment Setup
1. Activate a new conda/python environment
2. Execute the following to install NERP
```python
pip install NERP
```

#### Initialize NERP
```python
from NERP.models import NERP
model = NERP("env.yaml")
```

#### Training a NER model using NERP

1. Train the base model
```python
message = model.train()
print(message)
```

2. Training by using a trained model weights to initialize the base model 
```python
message = model.train_after_load_network()
print(message)
```

3. Training with KFold Cross Validation
```python
message = model.train_with_kfold()
print(message)
```

4. Training with KFold Cross Validation after load trained model weights to initialize the base model 
```python
message = model.train_with_kfold_after_loading_netwr()
print(message)
```

#### Inference of a NER model using NERP 

1. Prediction on a sample text
```python
output, message = model.inference_text()
print(output)
```

2. Prediction on a csv file
```python
message = model.inference_bulk()
print(message)
```

## Shout-outs
- Thanks to [NERDA](https://github.com/ebanalyse/NERDA) package to have initiated us to develop this pipeline.

## Cite this work

```
@inproceedings{nerp,
  title = {NERP},
  author = {Charangan Vasantharajan},
  year = {2022},
  publisher = {{GitHub}},
  url = {https://github.com/Chaarangan/NERP.git}
}
```

## Contact
We hope, that you will find `NERP` useful.

If you want to contribute to NERP open a
[PR](https://github.com/chaarangan/NERP/pulls).

If you encounter a bug or want to suggest an enhancement, please 
[open an issue](https://github.com/chaarangan/NERP/issues).