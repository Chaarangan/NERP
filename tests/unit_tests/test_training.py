from NERP.models import NERP

# instantiate a minimal model.
model = NERP(
    tag_scheme=[
        "B-date",
        "I-date",
        "L-date",
        "B-time",
        "I-time",
        "L-time",
        "B-place",
        "I-place",
        "L-place",
        "U-place",
        "B-business",
        "I-business",
        "L-business",
        "U-business",
        "B-street",
        "I-street",
        "L-street",
        "U-street",
        "B-country",
        "I-country",
        "L-country",
        "U-country",
        "B-symptom",
        "I-symptom",
        "L-symptom",
        "U-symptom",
        "B-timeframe",
        "L-timeframe",
        "U-timeframe",
        "B-region",
        "I-region",
        "L-region",
        "U-region",
        "B-first-name",
        "I-first-name",
        "L-first-name",
        "U-first-name",
        "U-last-name",
        "B-buildingno.",
        "I-buildingno.",
        "L-buildingno.",
        "U-buildingno.",
        "B-block",
        "I-block",
        "L-block",
        "B-exclamation",
        "L-exclamation",
        "U-exclamation"
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

train_csv_file = "/Users/charangan/Desktop/Intern/Tools/check/data/train.csv"
test_csv_file = "/Users/charangan/Desktop/Intern/Tools/check/data/test.csv"
limit = 10
test_size = 0.2
output_dir = "models/CovidData/"


def test_training():
    """Test if training runs successfully"""
    message = model.train(train_data=train_csv_file, test_data=test_csv_file,
                                   limit=limit, test_size=test_size,
                                   output_dir=output_dir)
    print(message)


def test_training_with_existing_model():
    """Test if training runs successfully with existing model"""
    isModelExists = True
    existing_model_path = "/Users/charangan/Desktop/Intern/Tools/check/models/CovidData/roberta-base/model.bin"
    message = model.train(train_data=train_csv_file, test_data=test_csv_file,
                          limit=limit, test_size=test_size, is_model_exists=isModelExists, existing_model_path=existing_model_path,
                          output_dir=output_dir)
    print(message)


def test_training_with_kfold():
    """Test if training runs successfully with kfold cross vaidation"""
    kfold = 0
    seed = 42
    message = model.train(train_data=train_csv_file, test_data=test_csv_file,
                          limit=limit, test_size=test_size,
                          output_dir=output_dir,
                          kfold=kfold,
                          seed=seed)
    print(message)
