import os
import pytest

from main.image_loading import load_data, read_img_sets


@pytest.fixture(scope='module')
def image_dir():
    return os.path.abspath(os.path.join('test', 'resources', 'images'))


@pytest.fixture(scope='module')
def image_size():
    return 64


def test_load_data(image_dir, image_size):

    images, labels, ids, cls, cls_map = load_data(
        image_dir=image_dir,
        image_size=image_size
    )

    assert images.shape == (20, image_size, image_size, 3)
    assert labels.shape == (20, 2)
    assert ids.shape == (20,)
    assert cls.shape == (20,)
    assert cls_map == {0: 'cat', 1: "dog"}


def test_read_img_sets(image_dir, image_size):

    data, category_ref = read_img_sets(
        image_dir=image_dir,
        image_size=image_size,
        validation_size=.2
    )

    assert sorted(list(data.__dict__)) == sorted(['train', 'test'])
    assert data.__dict__['test'].num_examples == 4
    assert data.__dict__['train'].num_examples == 16
    assert category_ref == {0: 'cat', 1: 'dog'}
