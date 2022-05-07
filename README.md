## NERP - Pipeline for training NER models

### **Config**

The user interface consists of only one file config as a JSON.
Change it to create the desired configuration.

#### Training Parameters
| Parameters | Description | Default |
| ------------- | ------------- | ------------- |
| train_data | path to training csv file | |
| test_data | path to testing csv file | |
| limit | Limit the number of observations to be returned from a given split. Defaults to None, which implies that the entire data split is returned. (it shoud be a ```int```) | 0 (whole data) |
| tag_scheme | All available NER tags for the given data set EXCLUDING the special outside tag, that is handled separately | |
| epochs | number of epochs (int) | 5 |
| warmup_steps | number of learning rate warmup steps (int) | 500 |
| train_batch_size | batch Size for DataLoader (int) | 64 |
| learning_rate | learning rate (float) | 0.0001 |
| max_len | the maximum sentence length (number of tokens after applying the transformer tokenizer) | 128 |
| dropout | dropout probability (float) | 0.1 |
| pretrained_models | list of 'huggingface' transformer models | roberta-base |
| test_size | train/test split ratio | 0.2 |
| is_model_exists | ```True``` for loading existing transformer model's weights otherwise ```False``` | False |
| existing_model_path | model derived from the transformer | |
| output_dir | path to output directory | models/ |
| kfold | number of splits | 0 |
| seed | random state value for k-fold | 42 |

#### Inference Parameters
| Parameters | Description |
| ------------- | ------------- |
| model_path | path to trained model | |
| tokenizer_path | path to saved tokenizer folder | |
| tag_scheme | All available NER tags for the given data set EXCLUDING the special outside tag, that is handled separately | |
| max_len | the maximum sentence length (number of tokens after applying the transformer tokenizer) | 128 |
| pretrained | 'huggingface' transformer model | roberta-base |
| is_bulk | ```True``` if input is a csv file otherwise ```False``` | False |
| in_file_path | if **is_bulk** ```True``` then path to inference file otherwise leave it as empty | |
| out_file_path | if **is_bulk** ```True``` then  path to output file otherwise leave it as empty | |
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
2. Execute the following to install required pip dependencies
```python
pip install NERP
```

#### Initialize NERP
```python
from NERP.models import NERP

model = NERP(
          tag_scheme=[
              'B-PER',
              'I-PER',
              'B-ORG',
              'I-ORG',
              'B-LOC',
              'I-LOC',
              'B-MISC',
              'I-MISC'
          ],
          hyperparameters={'epochs': 1,
                                'warmup_steps': 500,
                                'train_batch_size': 64,
                                'learning_rate': 0.0001},
          tokenizer_parameters={'do_lower_case': True},
          max_len=128,
          dropout=0.1,
          pretrained_models=[
              'roberta-base'
          ]
)
```

#### Training a NER model using NERP

1. Train the base model
```python
message = model.train(train_data="data/train.csv", test_data="data/test.csv",
                       limit=10000, test_size=0.2, is_model_exists=False, existing_model_path="", 
                       output_dir="models/")
print(message)
```

2. Training by using a trained model weights to initialize the base model 
```python
message = model.train(train_data="data/train.csv", test_data="data/test.csv",
                       limit=10000, test_size=0.2, is_model_exists=True, existing_model_path="models/pytorch-model.bin", 
                       output_dir="models/")
print(message)
```

3. Training with KFold Cross Validation
```python
message = model.train(train_data="data/train.csv", test_data="data/test.csv",
                       limit=10000, test_size=0.2, is_model_exists=False, existing_model_path="", 
                       output_dir="models/"
                       kfold = 10,
                       seed = 42)
print(message)
```



#### Inference of a NER model using NERP 

1. Prediction on a sample text
```python
output, message = model.inference(pretrained= "roberta-base", 
                              model_path="models/pytorch-model.bin",
                              tokenizer_path="models/tokenizer/",
                              text = "i was at h.w. holdings pte ltd from tenth august five eleven am to eleventh november one pm")
print(output)
```

2. Prediction on a csv file
```python
message = model.inference(pretrained= "roberta-base", 
                              model_path="models/pytorch-model.bin",
                              tokenizer_path="models/tokenizer/",
                              is_bulk = True,
                              in_file_path= "data/test.csv",
                              out_file_path= "data/output.csv")
print(message)
```

## Shout-outs
- Thanks to [NERDA](https://github.com/ebanalyse/NERDA) package to have initiated us to develop this pipeline.

## Cite this work

```
@inproceedings{nerp,
  title = {NERP},
  author = {Charangan, Vasantharajan},
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