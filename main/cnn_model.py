import os
import logging
import tensorflow as tf
from main.image_loading import read_img_sets


class ConvNet:

    def __init__(self, model_dir, image_dir, img_size, channels, batch_size):

        self.model_dir = model_dir
        self.image_dir = image_dir
        self.checkpoint_dir = os.path.join(os.path.abspath(model_dir), 'tensorflow', 'cnn', 'model')
        self.checkpoint_full_path = os.path.join(self.checkpoint_dir, 'model.ckpt')
        self.log_dir = os.path.join(os.path.abspath(self.model_dir), 'tensorflow', 'cnn', 'logs', 'cnn_with_summaries')
        self.img_size = img_size
        self.channels = channels
        self.neurons = 2 * img_size
        self.batch_size = batch_size

    def _flat_img_shape(self):
        return self.img_size * self.img_size * self.channels

    @staticmethod
    def _weight_variable(shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

    @staticmethod
    def _bias_variable(shape):
        return tf.Variable(tf.constant(0.05, shape=shape))

    @staticmethod
    def _conv2d(layer, weights):
        return tf.nn.conv2d(input=layer, filter=weights, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def _max_pool_2x2(layer):
        return tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    @staticmethod
    def _dropout(layer, keep_prob):
        return tf.nn.dropout(layer, keep_prob)

    def _new_conv_layer(self, layer, num_input_channels, filter_size, num_filters, use_pooling=True):

        weights = self._weight_variable(shape=[filter_size, filter_size, num_input_channels, num_filters])
        biases = self._bias_variable(shape=[num_filters])

        layer = self._conv2d(layer, weights) + biases

        if use_pooling:
            layer = self._max_pool_2x2(layer)

        layer = tf.nn.relu(layer)

        return layer

    @staticmethod
    def _flatten_layer(layer):

        layer_shape = layer.get_shape()

        num_features = layer_shape[1:4].num_elements()

        layer = tf.reshape(layer, [-1, num_features])

        return layer, num_features

    def _new_fully_connected_layer(self, layer, num_inputs, num_outputs, use_relu=True, layer_id=1, summaries=False):

        weights = self._weight_variable(shape=[num_inputs, num_outputs])
        biases = self._bias_variable(shape=[num_outputs])

        layer = tf.matmul(layer, weights) + biases

        if use_relu:
            layer = tf.nn.relu(layer)

        if summaries:
            tf.summary.histogram("Weight_fc" + str(layer_id), weights)
            tf.summary.histogram("bias_fc" + str(layer_id), biases)

        return layer

    @staticmethod
    def _log_progress(session, saver, cost, accuracy, epoch, test_feed_dict, checkpoint_path):

        val_loss = session.run(cost, feed_dict=test_feed_dict)
        acc = session.run(accuracy, feed_dict=test_feed_dict)

        msg = "Epoch {0} --- Accuracy: {1:>6.1%}, Validation Loss: {2:.3f}"
        logging.info(msg.format(epoch, acc, val_loss))

        save_path = saver.save(session, checkpoint_path)
        logging.debug("Creating resource: CNN Model [%s]", save_path)

    @staticmethod
    def _variables(flat_img_size, num_classes):

        with tf.name_scope('input'):

            x = tf.placeholder(
                tf.float32, shape=[None, flat_img_size], name='x-input')
            y_true = tf.placeholder(
                tf.float32, shape=[None, num_classes], name='y_true')

            keep_prob = tf.placeholder(tf.float32)

        return x, y_true, keep_prob

    def _model(self, x, keep_prob, num_classes):

        with tf.name_scope('reshaping'):
            x_image = tf.reshape(x, [-1, self.img_size, self.img_size, self.channels])
            tf.summary.image('example_images', x_image)

        with tf.name_scope('Conv1'):
            layer_conv1 = self._new_conv_layer(
                x_image,
                num_input_channels=self.channels,
                num_filters=self.img_size
            )

        with tf.name_scope('Conv2'):
            layer_conv2 = self._new_conv_layer(
                layer_conv1,
                num_input_channels=self.img_size,
                num_filters=self.neurons
            )

        with tf.name_scope('Conv3'):
            layer_conv3 = self._new_conv_layer(
                layer_conv2,
                num_input_channels=self.neurons,
                num_filters=self.neurons
            )

        with tf.name_scope('Fully_Connected1'):

            flat_layer, num_features = self._flatten_layer(layer_conv3)

            layer_fc1 = self._new_fully_connected_layer(
                flat_layer,
                num_inputs=num_features,
                num_outputs=1024,
                layer_id=1,
                summaries=True
            )

        with tf.name_scope('Dropout'):
            dropout_layer = self._dropout(layer_fc1, keep_prob)

        with tf.name_scope('Fully_Connected2'):

            layer_fc2 = self._new_fully_connected_layer(
                dropout_layer,
                num_inputs=1024,
                num_outputs=num_classes,
                use_relu=False,
                layer_id=2,
                summaries=True
            )

        return layer_fc2

    @staticmethod
    def _softmax(logits):

        with tf.name_scope('softmax'):
            y_pred = tf.nn.softmax(logits)

        return y_pred

    @staticmethod
    def _calculate_cost(logits, y_true):

        with tf.name_scope('cost'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits)
            cost = tf.reduce_mean(cross_entropy)
            tf.summary.scalar('cost', cost)

        return cost

    @staticmethod
    def _optimizer(cost):

        with tf.name_scope('train'):
            training_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

        return training_op

    def _calculate_accuracy(self, logits, y_true):

        with tf.name_scope('accuracy'):
            y_true_cls = tf.argmax(y_true, dimension=1)
            y_pred_cls = tf.argmax(self._softmax(logits), dimension=1)
            correct_prediction = tf.equal(y_pred_cls, y_true_cls)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', accuracy)

        return accuracy

    def _restore_or_initialize(self, session, saver):

        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)

        if ckpt:
            logging.debug("Loading CNN model: [%s]", self.checkpoint_full_path)
            saver.restore(session, ckpt.model_checkpoint_path)

        else:
            logging.warning("Unable to load CNN model: [%s]", self.checkpoint_full_path)

            os.makedirs(self.checkpoint_dir)
            tf.global_variables_initializer().run()

    def train(self, training_epochs=50):

        data, category_ref = read_img_sets(self.image_dir + '/train', self.img_size, validation_size=.2)

        flat_img_size = self._flat_img_shape()

        num_classes = len(category_ref)

        x, y_true, keep_prob = self._variables(flat_img_size, num_classes)
        logits = self._model(x, keep_prob, num_classes=num_classes)
        cost = self._calculate_cost(logits, y_true)
        training_op = self._optimizer(cost)
        accuracy = self._calculate_accuracy(logits, y_true)

        summary_op = tf.summary.merge_all()
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(self.log_dir, graph=tf.get_default_graph())

        with tf.Session() as sess:

            self._restore_or_initialize(sess, saver)

            for epoch in range(training_epochs):

                batch_count = int(data.train.num_examples / self.batch_size)

                for i in range(batch_count):

                    x_batch, y_true_batch, _, cls_batch = data.train.next_batch(self.batch_size)
                    x_batch = x_batch.reshape(self.batch_size, flat_img_size)

                    x_test_batch, y_test_batch, _, cls_test_batch = data.test.next_batch(self.batch_size)
                    x_test_batch = x_test_batch.reshape(self.batch_size, flat_img_size)

                    _, summary = sess.run([training_op, summary_op],
                                          feed_dict={x: x_batch, y_true: y_true_batch, keep_prob: 0.5})

                    writer.add_summary(summary, epoch * batch_count + i)

                if epoch % 5 == 0:
                    self._log_progress(sess, saver, cost, accuracy, epoch,
                                       test_feed_dict={x: x_test_batch, y_true: y_test_batch, keep_prob: 1.0},
                                       checkpoint_path=self.checkpoint_full_path)

    def predict(self):

        data, category_ref = read_img_sets(self.image_dir + '/predict', self.img_size)

        flat_img_size = self._flat_img_shape()

        num_classes = len(category_ref)

        x, y_true, keep_prob = self._variables(flat_img_size, num_classes)
        logits = self._model(x, keep_prob, num_classes=num_classes)
        predict_op = self._softmax(logits)

        with tf.Session() as sess:

            saver = tf.train.Saver()
            self._restore_or_initialize(sess, saver)

            x_predict_batch, y_predict_batch, _, cls_predict_batch = data.train.next_batch(batch_size=1)
            x_predict_batch = x_predict_batch.reshape(self.batch_size, flat_img_size)

            prediction = sess.run([tf.argmax(predict_op, dimension=1)], feed_dict={x: x_predict_batch, keep_prob: 1.0})

            return category_ref[prediction[0][0]], cls_predict_batch[0]
