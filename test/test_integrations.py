from main.run import run_app
from unittest.mock import patch
import sys
import os


def test_train(capfd):

    model_path = "test/resources/model"

    testargs = [
        "main/run.py",
        "test/resources/images/training",
        "-m", "train",
        "-d", model_path,
        "-i", "1",
        "-v", "info",
        "-c"
    ]

    with patch.object(sys, 'argv', testargs):
        run_app()

    out, err = capfd.readouterr()

    log_records = err.split('\n')

    assert "WARNING Removing resource: Directory" in log_records[0]
    assert "INFO Loading resource: Images" in log_records[1]
    assert "WARNING Unable to load ConvNet model: " in log_records[2]
    assert "INFO Epoch 0 --- Accuracy: " in log_records[3]

    assert os.path.exists(os.path.join(model_path, 'tensorflow', 'model', 'checkpoint'))
    assert os.path.exists(os.path.join(model_path, 'tensorflow', 'model', 'model.ckpt.data-00000-of-00001'))
    assert os.path.exists(os.path.join(model_path, 'tensorflow', 'model', 'model.ckpt.index'))
    assert os.path.exists(os.path.join(model_path, 'tensorflow', 'model', 'model.ckpt.meta'))

