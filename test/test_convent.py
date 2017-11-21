from main.convnet import ConvNet
import tensorflow as tf
import pytest
import os


@pytest.fixture(scope='module')
def convnet(image_dir, model_dir, image_size):
    return ConvNet(
        model_dir=model_dir,
        image_dir=image_dir,
        img_size=image_size,
        channels=3,
        filter_size=3,
        batch_size=2
    )


@pytest.fixture(scope='module')
def weights(convnet, image_size):
    return convnet._weight_variable(shape=[3, 3, 3, image_size])


@pytest.fixture(scope='module')
def variables(convnet):
    flat_img_size = convnet._flat_img_shape()
    return convnet._variables(flat_img_size, num_classes=2)


@pytest.fixture(scope='module')
def layer(variables, image_size):
    x, _, _ = variables
    return tf.reshape(x, [-1, image_size, image_size, 3])


@pytest.fixture(scope='module')
def logits(convnet, variables):
    x, _, keep_prob = variables
    return convnet._model(x, keep_prob, num_classes=2)


def evaluate_tfmethod(*args, method, dtype_code, op_type):

    tensor = method(*args)
    evaluate_tensor(tensor, dtype_code, op_type)


def evaluate_tensor(tensor, dtype_code, op_type):

    assert type(tensor) == tf.Tensor
    assert tensor.dtype == tf.DType(dtype_code)
    assert tensor.op.type == op_type


def test_convnet(model_dir, image_dir, image_size):

    net = ConvNet(model_dir, image_dir, image_size, channels=3, filter_size=3, batch_size=2)

    assert net.model_dir == model_dir
    assert net.image_dir == image_dir
    assert net.checkpoint_dir == os.path.join(model_dir.path, 'tensorflow', 'model')
    assert net.checkpoint_full_path == os.path.join(model_dir.path, 'tensorflow', 'model', 'model.ckpt')
    assert net.log_dir == os.path.join(model_dir.path, 'tensorflow', 'logs', 'cnn_with_summaries')
    assert net.img_size == image_size
    assert net.neurons == 2 * image_size
    assert net.channels == 3
    assert net.filter_size == 3
    assert net.batch_size == 2


def test_flat_img_shape(convnet, image_size):

    assert convnet._flat_img_shape() == image_size * image_size * 3


def test_weight_variable(convnet):

    tf_weight = convnet._weight_variable(shape=(20, 2))

    assert type(tf_weight) == tf.Variable
    assert tf_weight.dtype == tf.DType(101)
    assert tf_weight.initial_value.op.type == 'Add'


def test_bias_variable(convnet):

    tf_bias = convnet._bias_variable(shape=(20, 1))

    assert type(tf_bias) == tf.Variable
    assert tf_bias.dtype == tf.DType(101)
    assert tf_bias.initial_value.op.type == 'Const'


def test_variables(convnet):

    flat_img_size = convnet._flat_img_shape()
    x, y_true, keep_prob = convnet._variables(flat_img_size, num_classes=2)

    evaluate_tensor(tensor=x, dtype_code=1, op_type='Placeholder')
    evaluate_tensor(tensor=y_true, dtype_code=1, op_type='Placeholder')
    evaluate_tensor(tensor=keep_prob, dtype_code=1, op_type='Placeholder')


def test_conv2d(convnet, weights, layer):

    evaluate_tfmethod(weights, layer, method=convnet._conv2d, dtype_code=1, op_type='Conv2D')


def test_max_pool_2x2(convnet, layer):

    evaluate_tfmethod(layer, method=convnet._max_pool_2x2, dtype_code=1, op_type='MaxPool')


def test_dropout(convnet, variables, layer):

    _, _, keep_prob = variables

    evaluate_tfmethod(layer, keep_prob, method=convnet._dropout, dtype_code=1, op_type='Mul')


def test_new_conv_layer(convnet, layer, image_size):

    evaluate_tfmethod(
        layer, 3, image_size, True, method=convnet._new_conv_layer, dtype_code=1, op_type='Relu')


def test_flatten_layer(convnet, layer):

    layer, num_features = convnet._flatten_layer(layer)

    evaluate_tensor(tensor=layer, dtype_code=1, op_type='Reshape')

    assert num_features == 12288


def test_new_fully_connected_layer(convnet, layer):

    layer, num_features = convnet._flatten_layer(layer)

    evaluate_tfmethod(
        layer, num_features, 1024, method=convnet._new_fully_connected_layer, dtype_code=1, op_type='Relu')


def test_model(convnet, variables):

    x, _, keep_prob = variables

    evaluate_tfmethod(x, keep_prob, 2, method=convnet._model, dtype_code=1, op_type='Add')


def test_softmax(convnet, logits):

    evaluate_tfmethod(logits, method=convnet._softmax, dtype_code=1, op_type='Softmax')


def test_calculate_cost(convnet, logits, variables):

    _, y_true, _ = variables

    evaluate_tfmethod(logits, y_true, method=convnet._calculate_cost, dtype_code=1, op_type='Mean')


def test_optimizer(convnet, logits, variables):

    _, y_true, _ = variables
    cost = convnet._calculate_cost(logits, y_true)

    optimizer = convnet._optimizer(cost)

    assert type(optimizer) == tf.Operation
    assert optimizer.type == 'NoOp'
    assert optimizer.name == 'train/Adam'


def test_calculate_accuracy(convnet, logits, variables):

    _, y_true, _ = variables

    evaluate_tfmethod(logits, y_true, method=convnet._calculate_cost, dtype_code=1, op_type='Mean')
