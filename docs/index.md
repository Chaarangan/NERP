# NERP - NER Pipeline

## What is it?
NERP (Named Entity Recognition Pipeline) is a python package that offers an easy-to-use pipeline for fine-tuning pre-trained transformers for Named Entity Recognition (NER) tasks.

## Main Features
- Different Architectures (BiLSTM, CRF, BiLSTM+CRF)
- Fine-tune a pretrained model
- Save and reload model and train it on a new training data
- Fine-tune a pretrained model with K-Fold Cross-Validation
- Save and reload model and train it on a new training data with K-Fold Cross Validation
- Fine-tune multiple pretrained models
- Prediction on a single text
- Prediction on a CSV file

## Package Diagram

<table>
  <tr>
    <td>NERP Main Component</td>
  </tr>
  <tr>
    <td><img alt="NERP Main Component" src="https://github.com/Chaarangan/NERP/blob/master/diagrams/1.png"/></td>
  </tr>
  <tr>
    <td>Component of NERP K-Fold Cross Validation</td>
  </tr>
  <tr>
    <td><img align="left" alt="Component of NERP K-Fold Cross Validation" src="https://github.com/Chaarangan/NERP/blob/master/diagrams/2.png" height="200"/></td>
  </tr>
  <tr>
    <td>Component of NERP Inference</td>
  </tr>
  <tr>
    <td><img align="left" alt="Component of NERP Inference" src="https://github.com/Chaarangan/NERP/blob/master/diagrams/3.png" height="200"/></td>
  </tr>
 </table>

## **Config**

The user interface consists of only one file config as a YAML. Change it to create the desired configuration.

Sample ```env.yaml``` file
```yaml
torch:
  device: "cuda"
data:
  train_data: 'data/train.csv'
  valid_data: 'data/valid.csv'
  train_valid_split: None
  test_data: 'data/test.csv'
  limit: 10
  tag_scheme: ['B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

model: 
  archi: "baseline"
  o_tag_cr: True
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
  existing_model_path: "roberta-base/model.bin"
  existing_tokenizer_path: "roberta-base/tokenizer"
  output_dir: "output/"

kfold: 
  splits: 2
  seed: 42

inference:
  archi: "bilstm-crf"
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
| Parameters | Description | Default | Type |
| ------------- | ------------- | ------------- | ------------- |
| device |device: the desired device to use for computation. If not provided by the user, we take a guess. | ```cuda``` or ```cpu```| optional | 
| train_data | path to training csv file | | required |
| valid_data | path to validation csv file | | optional |
| train_valid_split | train/valid split ratio if valid data not exists | 0.2 | optional | 
| test_data | path to testing csv file | | required |
| limit | Limit the number of observations to be returned from a given split. Defaults to None, which implies that the entire data split is returned. (it shoud be a ```int```) | 0 (whole data) | optional |
| tag_scheme | All available NER tags for the given data set EXCLUDING the special outside tag, that is handled separately | | required |
| archi | The desired architecture for the model (baseline, bilstm-crf, bilstm, crf) (str) | baseline | optional |
| o_tag_cr | To include O tag in the classification report (bool) | True | optional |
| max_len | the maximum sentence length (number of tokens after applying the transformer tokenizer) | 128 | optional |
| dropout | dropout probability (float) | 0.1 | optional |
| epochs | number of epochs (int) | 5 | optional |
| warmup_steps | number of learning rate warmup steps (int) | 500 | optional |
| train_batch_size | batch Size for DataLoader (int) | 64 | optional |
| learning_rate | learning rate (float) | 0.0001 | optional |
| tokenizer_parameters | list of hyperparameters for tokenizer | do_lower_case: True | optional |
| pretrained_models | 'huggingface' transformer model | roberta-base | required |
| existing_model_path | model derived from the transformer | | optional |
| existing_tokenizer_path | tokenizer derived from the transformer | | optional |
| output_dir | path to output directory | models/ | optional |
| kfold | number of splits | 0 (no k-fold) | optional |
| seed | random state value for k-fold | 42 | optional |

#### Inference Parameters
| Parameters | Description | Default | Type |
| ------------- | ------------- | ------------- | ------------- |
| archi | The architecture for the trained model (baseline, bilstm-crf, bilstm, crf) (str) | baseline | optional |
| max_len | the maximum sentence length (number of tokens after applying the transformer tokenizer) | 128 | optional |
| pretrained | 'huggingface' transformer model | roberta-base | required |
| model_path | path to trained model | | required  |
| tokenizer_path | path to saved tokenizer folder | | optional |
| tag_scheme | All available NER tags for the given data set EXCLUDING the special outside tag, that is handled separately | | required |
| in_file_path | path to inference file otherwise leave it as empty | | optional |
| out_file_path | path to the output file if the input is a file, otherwise leave it as empty | | optional |
| text | sample inference text for individual prediction | "Hello from NERP" | optional |
---

### **Data Format**

Pipeline works with CSV files containing separated tokens and labels on each line. Sentences can be found in the `Sentence #` column. Labels should already be in the necessary format, e.g. IO, BIO, BILUO, ... The CSV file must contain the last three columns as same as below.

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

* model.bin - PyTorch NER model
* tokenizer files
* classification-report.csv - logging file
* If k-fold - split datasets, models and tokenizers for each iteration and accuracy file

---

### **Models**

All huggingface transformer-based models are allowed.

---

## Usage
### Environment Setup
1. Activate a new conda/python environment
2. Install NERP
- via pip
```bash
pip install NERP==1.0.2
```

- via repository
```bash
git clone --branch v1.0.2 https://github.com/Chaarangan/NERP.git
cd NERP & pip install -e .
```

### Initialize NERP
```python
from NERP.models import NERP
model = NERP("env.yaml")
```

### Training a NER model using NERP

1. Train a base model
```python
model.train()
```

2. Train an already trained model by loading its weights
```python
model.train_after_load_network()
```

3. Training with K-Fold Cross-Validation
```python
model.train_with_kfold()
```

4. Train an already trained model with K-Fold Cross Validation after loading its weights
```python
model.train_with_kfold_after_loading_network()
```

### Inference of a NER model using NERP 

1. Prediction on a single text through YAML file
```python
output = model.inference_text()
print(output)
```

1. Prediction on a single text through direct input
```python
output = model.predict("Hello from NERP")
print(output)
```

3. Prediction on a CSV file
```python
model.inference_bulk()
```

## License
MIT License

## Shout-outs
- Thanks to [NERDA](https://github.com/ebanalyse/NERDA) package to have initiated us to develop this pipeline. We have integrated the NERDA framework with NERP with some modifications from v1.0.0.

Changes from the NERDA(1.0.0) to our NERDA submodule.
1. Method for saving and loading tokenizer
2. Selected pull requests' solutions were added from [NERDA PRs](https://github.com/ebanalyse/NERDA/pulls) 
3. Implementation of the classification report
4. Added multiple network architecture support

## Cite this work

```
@inproceedings{nerp,
  title = {NERP},
  author = {Charangan Vasantharajan, Kyaw Zin Tun, Lim Zhi Hao, Chng Eng Siong},
  year = {2022},
  publisher = {{GitHub}},
  url = {https://github.com/Chaarangan/NERP.git}
}
```

## Contributing to NERP
All contributions, bug reports, bug fixes, documentation improvements, enhancements, and ideas are welcome.

Feel free to ask questions and send feedbacks on the [mailing list](https://groups.google.com/g/ner-pipeline).

If you want to contribute NERP, open a [PR](https://github.com/Chaarangan/NERP/pulls).

If you encounter a bug or want to suggest an enhancement, please open an [issue](https://github.com/Chaarangan/NERP/issues).
