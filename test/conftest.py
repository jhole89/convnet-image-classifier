import os
import pytest
from main.image_loading import load_data


@pytest.fixture(scope='session')
def image_dir():
    return os.path.abspath(os.path.join('resources', 'images', 'training'))


@pytest.fixture(scope='session')
def prediction_path():
    return os.path.abspath(os.path.join('resources', 'images', 'prediction', '1794225511_0a7ba68969.jpg'))


@pytest.fixture(scope='session')
def model_dir():
    return os.path.abspath(os.path.join('resources', 'model'))


@pytest.fixture(scope='session')
def image_size():
    return 64


@pytest.fixture(scope='session')
def load_image_data(image_dir, image_size):
    return load_data(
        image_dir=image_dir,
        image_size=image_size
    )
