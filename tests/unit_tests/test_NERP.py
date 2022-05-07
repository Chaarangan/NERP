from NERP.models import NERP

# instantiate a minimal model.
model = NERP(max_len=128,
              transformer='roberta-base',
              hyperparameters={'epochs': 1,
                               'warmup_steps': 10,
                               'train_batch_size': 5,
                               'learning_rate': 0.0001})


def test_instantiate_NERP():
    """Test that model has the correct/expected class"""
    assert isinstance(model, NERP)
