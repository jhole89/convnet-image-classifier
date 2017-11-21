from main.dataset import DataSet
import numpy as np


def test_dataset(load_image_data, image_size):

    images, labels, ids, cls, _ = load_image_data

    dataset = DataSet(images, labels, ids, cls)

    assert sorted(list(dataset.cls)) == sorted(['cat', 'dog'] * 10)
    assert dataset.cls.shape == (20,)

    assert dataset.epochs_completed == 0

    assert dataset.ids.shape == (20,)

    assert dataset.images.shape == (20, image_size, image_size, 3)
    assert dataset.images.dtype == np.float32
    assert dataset.images.min() == float(0)
    assert dataset.images.max() == float(1)

    assert np.array_equal(dataset.labels, [[1., 0.]] * 10 + [[0., 1.]] * 10)
    assert dataset.labels.shape == (20, 2)
    assert dataset.labels.dtype == np.float64
    assert dataset.labels.min() == float(0)
    assert dataset.labels.max() == float(1)

    assert dataset.num_examples == 20


def test_next_batch(load_image_data, image_size):

    images, labels, ids, cls, _ = load_image_data

    dataset = DataSet(images, labels, ids, cls)

    image_batch, label_batch, id_batch, cls_batch = dataset.next_batch(batch_size=2)

    assert list(cls_batch) == ['cat', 'cat']
    assert cls_batch.shape == (2,)

    assert image_batch.shape == (2, image_size, image_size, 3)
    assert image_batch.dtype == np.float32

    assert np.array_equal(label_batch, [[1., 0.], [1., 0.]])
    assert label_batch.shape == (2, 2)
    assert label_batch.dtype == np.float64
    assert label_batch.min() == float(0)
    assert label_batch.max() == float(1)

    assert id_batch.shape == (2,)
