from main.image_loading import read_img_sets, load_image
import numpy as np


def test_load_image(prediction_path, image_size):

    images, ids, labels, cls = load_image(
        image_path=prediction_path,
        image_size=image_size,
        file_index=0,
        num_files=1,
    )

    assert np.array(images).shape == (1, image_size, image_size, 3)
    assert ids == ['1794225511_0a7ba68969.jpg']
    assert labels == []
    assert cls == []


def test_load_data(image_size, load_image_data):

    images, labels, ids, cls, cls_map = load_image_data

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
