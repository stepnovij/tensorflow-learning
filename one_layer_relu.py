import logging

import numpy as np
import tensorflow as tf

from loader import load_test_train_validation_ds, IMAGE_YSIZE, IMAGE_XSIZE, NUM_LABELS
from utils import accuracy, convert_from_one_dim_labels, LOGGING_FORMAT


logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)


def construct_tf_constant(dataset, labels):
    tf_train_data = tf.constant(dataset, dtype=tf.float32)
    tf_train_labels = tf.constant(convert_from_one_dim_labels(labels, NUM_LABELS))
    return tf_train_data, tf_train_labels


def construct_tf_placeholder():
    tf_train_data = tf.placeholder("float", shape=[None, IMAGE_XSIZE*IMAGE_YSIZE])
    tf_train_labels = tf.placeholder("float", shape=[None, NUM_LABELS])
    return tf_train_data, tf_train_labels


n_hidden_1 = 256
n_hidden_2 = 256

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([IMAGE_XSIZE*IMAGE_YSIZE, n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, NUM_LABELS]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'out': tf.Variable(tf.random_normal([NUM_LABELS]))
}


def one_layer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    logits = tf.matmul(x, weights['h1']) + biases['b1']
    relu = tf.nn.relu(logits)
    # Output layer with linear activation
    out_layer = tf.matmul(relu, weights['out']) + biases['out']
    return out_layer


def one_layer_relu(learning_rate, num_steps, train_subset):
    # graph = tf.Graph()
    # with graph.as_default():

    # Variables
    tf_train_data, tf_train_labels = construct_tf_placeholder()

    # Construct architecture and get logits:
    logits = one_layer_perceptron(tf_train_data, weights, biases)

    train_labels = convert_from_one_dim_labels(train_dataset.label, NUM_LABELS)

    # Constants validation set
    tf_valid_data, tf_valid_labels = construct_tf_constant(validation_dataset.data,
                                                           validation_dataset.label)
    valid_labels = convert_from_one_dim_labels(validation_dataset.label, NUM_LABELS)

    # Constants training set
    tf_test_data, tf_test_label = construct_tf_constant(test_dataset.data, test_dataset.label)
    test_labels = convert_from_one_dim_labels(test_dataset.label, NUM_LABELS)


    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                  labels=tf_train_labels))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    train_prediction = tf.nn.softmax(logits)

    # valid_prediction = tf.nn.softmax(one_layer_perceptron(tf_valid_data, weights, biases))
    # test_prediction = tf.nn.softmax(one_layer_perceptron(tf_test_data, weights, biases))

    with tf.Session() as session:
        tf.global_variables_initializer().run()
        logging.info('Initialization')
        for step in range(num_steps):
            random_indx = np.random.randint(1, train_dataset.data.shape[0], size=train_subset)

            batch_data = train_dataset.data[random_indx]
            batch_labels = train_labels[random_indx]

            feed_dict = {tf_train_data: batch_data, tf_train_labels: batch_labels}
            _, l, predictions = session.run([optimizer, loss, train_prediction],
                                            feed_dict=feed_dict)

            if step % 500 == 0:
                logging.info('Loss at step %d: %f' % (step, l))
                logging.info('Training accuracy: %.1f%%' % accuracy(predictions, batch_labels))
        #         logging.info('Validation accuracy: %.1f%%' % accuracy(
        #             valid_prediction.eval(), valid_labels))
        # logging.info('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))


if __name__ == '__main__':
    test_dataset, train_dataset, validation_dataset = load_test_train_validation_ds()
    train_subset = 200
    learning_rate = 0.001
    num_steps = 3001
    logging.info('starting logistic_regression_different_batches')
    one_layer_relu(learning_rate, num_steps, train_subset)
