import logging

import numpy as np
import tensorflow as tf

from loader import load_test_train_validation_ds, IMAGE_YSIZE, IMAGE_XSIZE

NUM_LABELS = 10
LOGGING_FORMAT = '%(asctime)s - %(message)s'

logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)


def accuracy(predictions, labels):
     return 100*np.sum(np.argmax(predictions,1) == np.argmax(labels,1))/predictions.shape[0]


def convert_from_one_dim_labels(labels, num_labels=NUM_LABELS):
    train_dataset = np.zeros([labels.shape[0], num_labels])
    train_dataset[np.arange(labels.shape[0]), labels] = 1
    return train_dataset


def construct_tf_constant(dataset, labels):
    tf_train_data = tf.constant(dataset, dtype=tf.float32)
    tf_train_labels = tf.constant(convert_from_one_dim_labels(labels))
    return tf_train_data, tf_train_labels



def create_graph(learning_rate, num_steps):
    graph = tf.Graph()
    with graph.as_default():
        random_indx = np.random.randint(1,train_dataset.data.shape[0], size=train_subset)
        tf_train_data, tf_train_labels = construct_tf_constant(
            train_dataset.data[random_indx], train_dataset.label[random_indx]
        )
        train_labels = convert_from_one_dim_labels(train_dataset.label[:train_subset])

        tf_valid_data, tf_valid_labels = construct_tf_constant(
            validation_dataset.data, validation_dataset.label
        )
        valid_labels = convert_from_one_dim_labels(validation_dataset.label)

        tf_test_data, tf_test_label = construct_tf_constant(
            test_dataset.data, test_dataset.label
        )
        test_labels = convert_from_one_dim_labels(test_dataset.label)

        weights = tf.Variable(tf.truncated_normal([IMAGE_XSIZE*IMAGE_YSIZE, NUM_LABELS]))
        biases = tf.Variable(tf.zeros([NUM_LABELS]))

        logits = tf.matmul(tf_train_data, weights) + biases
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                      labels=tf_train_labels))

        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_data, weights) + biases)
        test_prediction = tf.nn.softmax(tf.matmul(tf_test_data, weights) + biases)

        with tf.Session(graph=graph) as session:
            tf.global_variables_initializer().run()
            logging.info('Initialization')
            for step in range(num_steps):
                _, l, predictions = session.run([optimizer, loss, train_prediction])
                if (step % 100 == 0):
                    logging.info('Loss at step %d: %f' % (step, l))
                    logging.info('Training accuracy: %.1f%%' % accuracy(
                        predictions, train_labels[:train_subset, :]))
                    # Calling .eval() on valid_prediction is basically like calling run(), but
                    # just to get that one numpy array. Note that it recomputes all its graph
                    # dependencies.
                    logging.info('Validation accuracy: %.1f%%' % accuracy(
                        valid_prediction.eval(), valid_labels))
            logging.info('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))


if __name__ == '__main__':

    test_dataset, train_dataset, validation_dataset = load_test_train_validation_ds()
    train_subset = 1000
    learning_rate = 0.5
    num_steps = 1001
    create_graph(learning_rate, num_steps)