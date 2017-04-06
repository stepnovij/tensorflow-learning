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


def construct_tf_placeholder(batch_size):
    tf_train_data = tf.placeholder(tf.float32, shape=(None, IMAGE_XSIZE*IMAGE_YSIZE))
    tf_train_labels = tf.placeholder(tf.int32, shape=(None, NUM_LABELS))
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


def inference(x):
    # Hidden layer with RELU activation
    logits = tf.matmul(x, weights['h1']) + biases['b1']
    relu = tf.nn.relu(logits)
    # Output layer with linear activation
    out_layer = tf.matmul(relu, weights['out']) + biases['out']
    return out_layer #tf.nn.softmax()


def loss_func(logits, probabilities):
    # sparse_softmax_cross_entropy_with_logits - used when you have labels
    # softmax_cross_entropy_with_logits - used when you have probabilities
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=probabilities, logits=logits)
    regularizers = (tf.nn.l2_loss(weights['h1']) +
                    tf.nn.l2_loss(weights['out']) +
                    tf.nn.l2_loss(biases['b1']) +
                    tf.nn.l2_loss(biases['out']))
    return tf.reduce_mean(cross_entropy) + 0.0001*regularizers


def training(loss, learning_rate):
    tf.scalar_summary('loss', loss)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss, global_step=global_step)
    return training_op


def evaluate(output, y):
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)


def one_layer_relu(learning_rate, num_steps, train_subset):
    # Load dataset:
    test_dataset, train_dataset, validation_dataset = load_test_train_validation_ds()

    # Variables
    tf_train_data, tf_train_labels = construct_tf_placeholder(train_subset)

    train_labels = convert_from_one_dim_labels(train_dataset.label, NUM_LABELS)

    # Constants validation set
    tf_valid_data, tf_valid_labels = construct_tf_constant(validation_dataset.data,
                                                           validation_dataset.label)
    valid_labels = convert_from_one_dim_labels(validation_dataset.label, NUM_LABELS)

    # Constant test set
    tf_test_data, tf_test_label = construct_tf_constant(test_dataset.data, test_dataset.label)
    test_labels = convert_from_one_dim_labels(test_dataset.label, NUM_LABELS)

    # Construct architecture and get predictions:
    logits = inference(tf_train_data)

    # Get loss:
    loss = loss_func(logits, tf_train_labels)
    # loss += 0.01*regularizers
    training_op = training(loss, learning_rate)
    eval_op = evaluate(logits, tf_train_labels)

    # Predictions:
    # test_prediction = inference(tf_test_data)

    summary_op = tf.merge_all_summaries()
    saver = tf.train.Saver()

    test_dict = {tf_train_data: test_dataset.data, tf_train_labels: test_labels}

    with tf.Session() as session:
        summary_writer = tf.train.SummaryWriter('logs/', graph_def=session.graph_def)
        tf.global_variables_initializer().run()
        logging.info('Initialization')
        for step in range(num_steps):
            random_indx = np.random.randint(1, train_dataset.data.shape[0], size=train_subset)

            batch_data = train_dataset.data[random_indx]
            batch_labels = train_labels[random_indx]

            feed_dict = {tf_train_data: batch_data, tf_train_labels: batch_labels}
            _, loss_value = session.run([training_op, loss], feed_dict=feed_dict)
            if step % 500 == 0:

                val_feed_dict = {
                    tf_train_data: validation_dataset.data,
                    tf_train_labels: valid_labels
                }
                simple_accuracy = session.run(eval_op, feed_dict=val_feed_dict)
                logging.info('Loss at step %d: %f, accuracy: %f' % (step, loss_value, simple_accuracy))

                summary_str = session.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                saver.save(session, 'logs/model-checkpoint', global_step=step)
        simple_accuracy = session.run(eval_op, feed_dict=test_dict)
        logging.info('Test accuracy: %.1f%%' % float(100*simple_accuracy))


if __name__ == '__main__':
    train_subset = 150
    learning_rate = 0.001
    num_steps = 3001
    logging.info('starting logistic_regression_different_batches')
    one_layer_relu(learning_rate, num_steps, train_subset)
