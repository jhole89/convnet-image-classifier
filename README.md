# ConvNet Image Classifier
**Master:** [![Build Status](https://travis-ci.org/jhole89/convet-image-classifier.svg?branch=master)](https://travis-ci.org/jhole89/convet-image-classifier)
[![Coverage Status](https://coveralls.io/repos/github/jhole89/convet-image-classifier/badge.svg?branch=master)](https://coveralls.io/github/jhole89/convet-image-classifier?branch=master)
**Develop:** [![Build Status](https://travis-ci.org/jhole89/convet-image-classifier.svg?branch=develop)](https://travis-ci.org/jhole89/convet-image-classifier)
[![Coverage Status](https://coveralls.io/repos/github/jhole89/convet-image-classifier/badge.svg?branch=develop)](https://coveralls.io/github/jhole89/convet-image-classifier?branch=develop)


## Getting Started

### Prerequisites

* [Python 3.6+](https://www.python.org/downloads/)

### Installation

1. Install Python 3.6 on your Operating System as per the Python Docs.
Continuum's Anaconda distribution is recommended.

2. Checkout the repo:
`git clone https://github.com/jhole89/convnet-image-classifier.git`

3. Setup the project dependencies:
```
$ cd convnet-image-classifier
$ export PYTHONPATH=$PYTHONPATH:$(pwd)
$ pip install -r requirements.txt
```

### Execution

#### Training

To train the convnet to categorise different images we require a directory
of images organised into sub-directories based on the category, ideally with
an equal amount of images in each, e.g.

```
convnet-image-classifier
    │
    └───images
        │
        └───category1
        │   │cat11.jpg
        │   │cat12.jpg
        │   │...
        │
        └───category2
            │cat21.jpg
            │cat22.jpg
            │...
```

With such a layout the most basic convnet can now be trained using the command:

```
$ python path/to/training/images -m train
```

If correct you should see logs similar to the following:
```
2017-11-20 23:14:14,200 INFO Loading resource: Images [/convnet-image-classifier/test/resources/images/training]
2017-11-20 23:14:15,748 WARNING Unable to load ConvNet model: [/convnet-image-classifier/test/resources/model/tensorflow/model/model.ckpt]
2017-11-20 23:14:23,068 INFO Epoch 0 --- Accuracy:   0.0%, Validation Loss: 3.015
2017-11-20 23:14:59,409 INFO Epoch 5 --- Accuracy:   0.0%, Validation Loss: 1.764
2017-11-20 23:15:36,010 INFO Epoch 10 --- Accuracy:   0.0%, Validation Loss: 1.122
2017-11-20 23:16:12,478 INFO Epoch 15 --- Accuracy:   0.0%, Validation Loss: 3.276
2017-11-20 23:16:48,308 INFO Epoch 20 --- Accuracy:   0.0%, Validation Loss: 4.181
2017-11-20 23:17:23,088 INFO Epoch 25 --- Accuracy:   0.0%, Validation Loss: 5.526
2017-11-20 23:17:58,753 INFO Epoch 30 --- Accuracy:   0.0%, Validation Loss: 5.843
2017-11-20 23:18:33,143 INFO Epoch 35 --- Accuracy:   0.0%, Validation Loss: 6.223
2017-11-20 23:19:08,327 INFO Epoch 40 --- Accuracy:   0.0%, Validation Loss: 6.748
2017-11-20 23:19:42,820 INFO Epoch 45 --- Accuracy:   0.0%, Validation Loss: 7.763
```

#### Predicting
```
$ python path/to/images -m predict
```

## Test Coverage and Coding Style

This project uses [Travis-CI](https://travis-ci.org/jhole89/convnet-image-classifier) for our CI/CD
to run test coverage (pytest) and style checks (pycodestyle) against every new commit and against
the nightly CPython build to ensure we are always aligned with the latest CPython dev builds.
Build status is shown at the top of this README.


## Acknowledgments
