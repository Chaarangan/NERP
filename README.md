# NERP - NER Pipeline

## What is it?
NERP (Named Entity Recognition Pipeline) is a Python package that provides a user-friendly pipeline for fine-tuning pre-trained transformers for Named Entity Recognition (NER) tasks.

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
| device | The desired device to use for computation. If not provided by the user, the package will make a guess | ```cuda``` or ```cpu```| str | 
| seed | A random state value used for a specific experiment | 42 | int |

---

#### Data Parameters
| Parameters | Description | Default | Type |
| ------------- | ------------- | ------------- | ------------- |
| train_data | The path to the training CSV file | | str |
| valid_data | The path to the validation CSV file | | str |
| train_valid_split | The train/validation split ratio if there's no validation data | 0.2 | float | 
| test_data | The path to the testing CSV file | | str |
| sep | The delimiter to use | ',' | str |
| quoting | The behavior for field quoting per csv.QUOTE_* constants | 3 | int |
| shuffle | Whether to shuffle the entire dataset before training | False | bool |
| limit | The maximum number of observations to be returned from a given split. Defaults to 0, which returns the entire data split | 0 | int |
| tags | A list of all the available NER tags for the given dataset, excluding the special outside tag, which is handled separately | | List[str] |

---

#### Model Parameters
| Parameters | Description | Default | Type |
| ------------- | ------------- | ------------- | ------------- |
| archi | The desired architecture for the model. It can be one of the following: baseline, bilstm-crf, bilstm, or crf | baseline | str |
| max_len | The maximum sentence length (number of tokens after applying the transformer tokenizer) | 128 | int |
| dropout | The dropout probability  | 0.1 | float |
| epochs | The number of epochs | 5 | int |
| num_workers | The number of workers/threads for data processing | 1 | int |
| warmup_steps | The number of warmup steps for the optimizer | 500 | int |
| train_batch_size | The batch size for training DataLoader | 64 | int |
| valid_batch_size | The batch size for validation DataLoader | 64 | int |
| lr | The learning rate | 0.0001 | float |
| do_lower_case | Lowercase the sequence during the tokenization | True | bool |
| pretrained_models | A list of 'huggingface' transformer models | roberta-base | str |

---

#### Training Parameters
| Parameters | Description | Default | Type |
| ------------- | ------------- | ------------- | ------------- |
| continue_from_checkpoint | Boolean flag to continue training from a previous checkpoint | False | bool | 
| checkpoint_path | Path to the pre-trained model derived from the transformer  | | str |
| checkpoint_tokenizer_path | Path to the tokenizer derived from the transformer | | str |
| output_dir | Path to the output directory  | output/ | str |
| o_tag_cr | Boolean flag to include O tag in the classification report  | True | bool |
| return_accuracy | Boolean flag to return accuracy for every training step | False | bool |

---

#### KFold Parameters
| Parameters | Description | Default | Type |
| ------------- | ------------- | ------------- | ------------- |
| is_kfold | Enable K-Fold Cross-Validation for training | False | bool |
| splits | Number of splits for K-Fold Cross-Validation | 0 | int |
| test_on_original | Evaluate on the original test set for each iteration if set to True | False | bool |

---

#### Inference Parameters
| Parameters | Description | Default | Type |
| ------------- | ------------- | ------------- | ------------- |
| pretrained | A 'huggingface' transformer model to use for inference | roberta-base | str |
| model_path | Path to the trained model file |  | str  |
| tokenizer_path | Path to the saved tokenizer folder |  | str |
| in_file_path | Path to the input file to be used for inference | | str |
| out_file_path | Path to the output file for saving the inference results | | str |

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

Once the model training is complete, the pipeline will generate the following files in the output directory:

* model.bin - PyTorch NER model
* Tokenizer files
* Classification-report.csv - a logging file
* In case of k-fold training, the pipeline generates split datasets, models, tokenizers, and accuracy files for each iteration.

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
pip install NERP==1.1-rc1
```

- via repository
```bash
git clone --branch v1.1-rc1 https://github.com/Chaarangan/NERP.git
cd NERP && pip install -e .
```

### Initialize NERP
```python
from NERP.models import NERP
model = NERP("env.yaml")
```

### Training
- Common function to call
```python
model.train()
```
There are several options depending on your needs:
- Casual Training: Configure the YAML file and set ```continue_from_checkpoint``` as ```False``` and ```is_kfold``` as ```False```. Then call ```model.train()```.
- Training from a previous checkpoint: Configure the YAML file and set ```continue_from_checkpoint``` as ```True``` and ```is_kfold``` as ```False```. You will need to specify the ```checkpoint_path```. Then call ```model.train()```.
- Training with KFold: Configure the YAML file and set ```continue_from_checkpoint``` as ```False``` and ```is_kfold``` as ```True```. You will need to specify the number of ```splits```. If you wish to test each fold with your original test set rather than its own test split, set the ```test_on_original``` variable as ```True```. Then call ```model.train()```.
- Training from a previous checkpoint with KFold: Configure the YAML file and set ```continue_from_checkpoint``` as ```True``` and ```is_kfold``` as ```True```. You will need to specify the ```checkpoint_path```. Then call ```model.train()```.

### Predictions 
There are several options depending on your needs:
- Prediction on a CSV file: Configure the YAML file and give ```model_path```, ```tokenizer_path``` (if exists), ```in_file_path```, and ```out_file_path```. Then call ```model.predict()```.
```python
model.predict()
```

- Prediction on text: Configure the YAML file and give ```model_path``` and ```tokenizer_path``` (if exists). Then call ```model.predict_text(“some text”)```.
```python
output = model.predict_text("Hello from NERP")
print(output)
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

### PRs
- [@tanmaysurana](https://github.com/tanmaysurana) (Tanmay Surana): add support for testing on multiple files, add additional parameters to maintain consistency across multiple experiments (validation batch size, shuffle, fixed seed), and improve loss computation algorithms [PR #20](https://github.com/Chaarangan/NERP/pull/20)

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