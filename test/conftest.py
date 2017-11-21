from main.image_loading import load_data
from main.directory import Directory
import numpy as np
import pytest
import os


@pytest.fixture(scope='session')
def image_dir():
    return Directory(os.path.join('test', 'resources', 'images', 'training'))


@pytest.fixture(scope='session')
def prediction_path():
    return os.path.abspath(os.path.join('test', 'resources', 'images', 'prediction', '1794225511_0a7ba68969.jpg'))


@pytest.fixture(scope='session')
def model_dir():
    return Directory(os.path.join('test', 'resources', 'model'))


@pytest.fixture(scope='session')
def image_size():
    return 64


@pytest.fixture(scope='session')
def load_image_data(image_dir, image_size):
    images, labels, ids, cls, binary_cls_map = load_data(image_dir=image_dir, image_size=image_size)
    return np.array(images), np.array(labels), np.array(ids), np.array(cls), binary_cls_map
