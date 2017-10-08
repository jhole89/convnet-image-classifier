import os

from main.image_loading import load_data, read_img_sets


def test_load_data():

    images, labels, ids, cls, cls_map = load_data(
        image_dir=os.path.abspath(os.path.join('test', 'resources', 'images')),
        image_size=64
    )

    assert images.shape == (20, 64, 64, 3)
    assert labels.shape == (20, 2)
    assert ids.shape == (20,)
    assert cls.shape == (20,)
    assert cls_map == {0: 'cat', 1: "dog"}
