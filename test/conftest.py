import os
import pytest
from main.image_loading import load_data


@pytest.fixture(scope='session')
def image_dir():
    return os.path.abspath(os.path.join('test', 'resources', 'images'))


@pytest.fixture(scope='session')
def image_size():
    return 64


@pytest.fixture(scope='session')
def load_image_data(image_dir, image_size):
    return load_data(
        image_dir=image_dir,
        image_size=image_size
    )
