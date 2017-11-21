# ConvNet Image Classifier
**Master:** [![Build Status](https://travis-ci.org/jhole89/connvet-image-classifier.svg?branch=master)](https://travis-ci.org/jhole89/convnet-image-classifier)
[![Coverage Status](https://coveralls.io/repos/github/jhole89/convnet-image-classifier/badge.svg?branch=master)](https://coveralls.io/github/jhole89/convnet-image-classifier?branch=master)
**Develop:** [![Build Status](https://travis-ci.org/jhole89/convnet-image-classifier.svg?branch=develop)](https://travis-ci.org/jhole89/convnet-image-classifier)
[![Coverage Status](https://coveralls.io/repos/github/jhole89/convnet-image-classifier/badge.svg?branch=develop)](https://coveralls.io/github/jhole89/convnet-image-classifier?branch=develop)


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

With such a layout the convnet can now be trained using the command:

```
$ python main/run.py path/to/training/images -m train
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

With the convnet trained it is possible to be used for image classification
when the image is unknown. This can be done over a single image file or a
directory of images using the commands:

```
$ python main/run.py path/to/images -m predict
$ python main/run.py path/to/image/file.jpg -m predict
```

Note that if a model directory was specified during the training stage, this same
model directory should also be specified here. For more on optional arguments
see the section below.

If correct you should see logs similar to the following:
```
2017-11-21 21:37:47,049 INFO Loading resource: Images [convet-image-classifier/test/resources/images/prediction]
2017-11-21 21:37:48,566 INFO File: 437202643_e32ce43baa.jpg --- Prediction: dog
2017-11-21 21:37:48,595 INFO File: 1794225511_0a7ba68969.jpg --- Prediction: dog
```

#### Optional Command Line Parameters
There are a number of arguments that can be used when training or predicting.
These all have default values so are optional, and can be passed using either
short or long form:

| short | long | desc | example |
|:---:|:---:|:---|:---|
| -m | --mode | train or predict mode | -m train |
| -d | --model_dir | directory to store model | -d tmp |
| -i | --iterations | num. of iterations (training only) | -i 2000 |
| -b | --batch_size | num. images per batch | -b 100 |
| -v | --verbosity | log verbosity level | -v info |
| -c | --clean | clean run flag (training only) | -c |

### Tensorboard

As convnet training can take a long time depending on local resources of the
CPU/GPU it can be helpful to visualise the training rate, accuracy, and cost.
This can be achieved using TensorBoard, which comes packed with TensorFlow.
To activate TensorBoard run:

```
tensorboard --logdir=<model_dir>/tensorflow/cnn/logs/cnn_with_summaries
```

TensorBoard can now be accessed via a web browser at 127.0.0.1:6006
where a number of metrics can be observed. For more on Tensorboard please
read the Tensorboard [docs.](https://www.tensorflow.org/get_started/summaries_and_tensorboard)

## Test Coverage and Coding Style

This project uses [Travis-CI](https://travis-ci.org/jhole89/convnet-image-classifier) for our CI/CD
to run test coverage (pytest) and style checks (pycodestyle) against every new commit and against
the nightly CPython build to ensure we are always aligned with the latest CPython dev builds.
Build status is shown at the top of this README.
