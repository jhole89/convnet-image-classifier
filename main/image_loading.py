import os
import cv2
import glob
import numpy as np
import logging
from sklearn.utils import shuffle
from main.dataset import DataSet


def load_image(image_path, image_size, file_index=0, num_files=1,
               images=None, ids=None, labels=None, cls=None, category=None):

    if images is None:
        images = []

    if ids is None:
        ids = []

    if labels is None:
        labels = []

    if cls is None:
        cls = []

    image_path = os.path.abspath(image_path)

    logging.debug("Loading resource: Image [%s]", image_path)

    image = cv2.imread(image_path)
    image = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
    images.append(image)

    if category:

        label = np.zeros(num_files)
        label[file_index] = 1.0
        labels.append(label)

        cls.append(category)

    filebase = os.path.basename(image_path)
    ids.append(filebase)

    return images, ids, labels, cls


def load_data(image_dir, image_size):

    images = []
    labels = []
    ids = []
    cls = []

    binary_cls_map = {}

    logging.info("Loading resource: Images [%s]", image_dir.path)

    training_dirs = os.listdir(image_dir.path)
    num_files = len(training_dirs)

    for category in training_dirs:
        index = training_dirs.index(category)
        binary_cls_map[index] = category

        logging.debug("Loading resource: %s images [Index: %s]" % (category, index))

        path = os.path.join(image_dir.path, category, '*g')
        file_list = glob.glob(path)

        for file in file_list:
            images, ids, labels, cls = load_image(
                file, image_size, index,
                num_files, images, ids,
                labels, cls, category
            )

    return images, labels, ids, cls, binary_cls_map


def read_img_sets(image_dir, image_size, validation_size=0):
    class DataSets:
        pass

    data_sets = DataSets()
    cls_map = None

    if type(image_dir).__name__ == 'File':
        images, ids, labels, cls = load_image(image_dir.path, image_size)

    else:
        images, labels, ids, cls, cls_map = load_data(image_dir, image_size)
        images, labels, ids, cls = shuffle(images, labels, ids, cls)

    images = np.array(images)
    labels = np.array(labels)
    ids = np.array(ids)
    cls = np.array(cls)

    if isinstance(validation_size, float):
        validation_size = int(validation_size * images.shape[0])

    test_images = images[:validation_size]
    test_labels = labels[:validation_size]
    test_ids = ids[:validation_size]
    test_cls = cls[:validation_size]

    train_images = images[validation_size:]
    train_labels = labels[validation_size:]
    train_ids = ids[validation_size:]
    train_cls = cls[validation_size:]

    data_sets.train = DataSet(train_images, train_labels, train_ids, train_cls)
    data_sets.test = DataSet(test_images, test_labels, test_ids, test_cls)

    return data_sets, cls_map
