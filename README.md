# NERP - NER Pipeline

## What is it?
NERP (Named Entity Recognition Pipeline) is a python package that offers an easy-to-use pipeline for fine-tuning pre-trained transformers for Named Entity Recognition (NER) tasks.

## Main Features include:
- Support for multiple architectures, such as BiLSTM, CRF, and BiLSTM+CRF.
- Fine-tuning of pre-trained models.
- Ability to save and reload models and train them with new training data.
- Fine-tuning of pre-trained models using K-Fold Cross-Validation.
- Ability to save and reload models and train them with new training data using K-Fold Cross-Validation.
- Fine-tuning of multiple pre-trained models.
- Prediction on a single text.
- Prediction on a CSV file.

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
  seed: 42

data:
  train_data: 'data/train.csv'
  valid_data: 'data/valid.csv'
  train_valid_split: 0.2
  test_data: 'data/test.csv'
  parameters:
        sep: ','
        quoting: 3
        shuffle: False
  limit: 0
  tag_scheme: ['B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

model: 
  archi: "baseline"
  max_len: 128 
  dropout: 0.1
  num_workers: 1
  hyperparameters:
    epochs: 1
    warmup_steps: 500
    train_batch_size: 64
    valid_batch_size: 64
    lr: 0.0001
  tokenizer_parameters: 
    do_lower_case: True
  pretrained_models: 
    - roberta-base

training:
  continue_from_checkpoint: False
  checkpoint_path: "roberta-base/model.bin"
  checkpoint_tokenizer_path: "roberta-base/tokenizer"
  output_dir: "output/"
  o_tag_cr: True
  return_accuracy: False

kfold: 
  is_kfold: False
  splits: 2
  test_on_original: False

inference:
  pretrained: "roberta-base"
  model_path: "roberta-base/model.bin"
  tokenizer_path: "roberta-base/tokenizer"
  in_file_path: "data/test.csv"
  out_file_path: "data/output.csv"
```


#### Torch Parameters
| Parameters | Description | Default | Type |
| ------------- | ------------- | ------------- | ------------- |
| device | the desired device to use for computation. If not provided by the user, we take a guess. | ```cuda``` or ```cpu```| str | 
| seed | Random state value for a particular experiment | 42 | int |
---

#### Data Parameters
| Parameters | Description | Default | Type |
| ------------- | ------------- | ------------- | ------------- |
| train_data | path to training csv file | | str |
| valid_data | path to validation csv file | | str |
| train_valid_split | train/valid split ratio if valid data not exists | 0.2 | float | 
| test_data | path to testing csv file | | str |
| sep | Delimiter to use | ',' | str |
| quoting | Control field quoting behavior per csv.QUOTE_* constants. | 3 | int |
| shuffle | Shuffle the entire dataset before training | False | bool |
| limit | Limit the number of observations to be returned from a given split. Defaults to None, which implies that the entire data split is returned. | 0 (whole data) | int |
| tags | All available NER tags for the given data set EXCLUDING the special outside tag, that is handled separately | | List[str] |
---

#### Model Parameters
| Parameters | Description | Default | Type |
| ------------- | ------------- | ------------- | ------------- |
| archi | The desired architecture for the model (baseline, bilstm-crf, bilstm, crf) | baseline | str |
| max_len | the maximum sentence length (number of tokens after applying the transformer tokenizer) | 128 | int |
| dropout | dropout probability  | 0.1 | float |
| epochs | number of epochs | 5 | int |
| num_workers | number of workers/threads for data processing | 1 | int |
| warmup_steps | number of workers/threads for data loader | 500 | int |
| train_batch_size | batch Size for training DataLoader | 64 | int |
| valid_batch_size | batch Size for validation DataLoader | 64 | int |
| lr | learning rate (float) | 0.0001 | float |
| do_lower_case | Lowercase the sequence during the tokenization | True | bool |
| pretrained_models | list of 'huggingface' transformer models | roberta-base | str |
---

#### Training Parameters
| Parameters | Description | Default | Type |
| ------------- | ------------- | ------------- | ------------- |
| continue_from_checkpoint | Continue training from previous checkpoint | False | bool | 
| checkpoint_path | model derived from the transformer  | | str |
| checkpoint_tokenizer_path | tokenizer derived from the transformer | | str |
| output_dir | path to output directory  | output/ | str |
| o_tag_cr | To include O tag in the classification report  | True | bool |
| return_accuracy | Return accuracy for every training step | False | bool |
---

#### Training Parameters
| Parameters | Description | Default | Type |
| ------------- | ------------- | ------------- | ------------- |
| is_kfold | Train with KFold Cross-Validation | False | bool |
| splits | number of splits | 0 | int |
| test_on_original | True, if you need to test on the original test set for each iteration | False | bool |
---

#### Inference Parameters
| Parameters | Description | Default | Type |
| ------------- | ------------- | ------------- | ------------- |
| pretrained | 'huggingface' transformer model | roberta-base | str |
| model_path | path to trained model |  | str  |
| tokenizer_path | path to saved tokenizer folder |  | str |
| in_file_path | path to inference file otherwise leave it as empty | | str |
| out_file_path | path to the output file if the input is a file, otherwise leave it as empty | | str |
---

### **Data Format**

Pipeline works with CSV files containing separated tokens and labels on each line. Sentences can be found in the `Sentence #` column. Labels should already be in the necessary format, e.g. IO, BIO, BILUO, ... The CSV file must contain the last three columns as same as below.

| Sentence # | Word | Tag 
| --- | --- | ---
| Sentence: 0 | i | o
| Sentence: 0 | was | O
| Sentence: 0 | at | O
| Sentence: 0 | h.w. | B-place
| Sentence: 0 | holdings | I-place
| Sentence: 0 | pte | I-place

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
pip install NERP==1.1
```

- via repository
```bash
git clone --branch v1.1 https://github.com/Chaarangan/NERP.git
cd NERP && pip install -e .
```

### Initialize NERP
```python
from NERP.models import NERP
model = NERP("env.yaml")
```

### Training a NER model using NERP

Train a base model
```python
model.train()
```

### Inference of a NER model using NERP 

1. Prediction on a single text
```python
output = model.predict_text("Hello from NERP")
print(output)
```

2. Prediction on a input file
```python
model.predict()
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
5. Support for enforcing reproducibility in data preparation and model training


## Contributing to NERP
- All contributions, bug reports, bug fixes, documentation improvements, enhancements, and ideas are welcome.
- Feel free to ask questions and send feedbacks on the [mailing list](https://groups.google.com/g/ner-pipeline).
- If you want to contribute NERP, open a [PR](https://github.com/Chaarangan/NERP/pulls).
- If you encounter a bug or want to suggest an enhancement, please open an [issue](https://github.com/Chaarangan/NERP/issues).

### Contributors
<a href="https://github.com/Chaarangan/NERP/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Chaarangan/NERP"/>
</a>

## Cite this work

```
@inproceedings{medbert,
    author={Vasantharajan, Charangan and Tun, Kyaw Zin and Thi-Nga, Ho and Jain, Sparsh and Rong, Tong and Siong, Chng Eng},
    booktitle={2022 Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)},
    title={MedBERT: A Pre-trained Language Model for Biomedical Named Entity Recognition},
    year={2022},
    volume={},
    number={},
    pages={1482-1488},
    doi={10.23919/APSIPAASC55919.2022.9980157}
}
@inproceedings{nerp,
  title = {NERP},
  author = {Charangan Vasantharajan, Kyaw Zin Tun, Lim Zhi Hao, Chng Eng Siong},
  year = {2022},
  publisher = {{GitHub}},
  url = {https://github.com/Chaarangan/NERP.git}
}
```