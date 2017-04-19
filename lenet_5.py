import logging

import numpy as np
import tensorflow as tf

from loader import load_test_train_validation_ds, IMAGE_YSIZE, NUM_LABELS
from utils import convert_from_one_dim_labels, LOGGING_FORMAT


logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)


batch_size = 128
learning_rate = 0.001
patch_size = 5
depth = 6
num_channels = 1
image_size = IMAGE_YSIZE


# Convolutional layer
layer1_weights = tf.Variable(tf.truncated_normal(
    [patch_size, patch_size, num_channels, depth], stddev=0.1))
layer1_biases = tf.Variable(tf.zeros([depth]))

# MaxPool layer
layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))

# Convolutional layer
layer3_weights = tf.Variable(tf.truncated_normal(
    [patch_size, patch_size, 6, 16], stddev=0.1))
layer3_biases = tf.Variable(tf.zeros([16]))

# MaxPool layer
layer4_biases = tf.Variable(tf.constant(1.0, shape=[16]))

# Fully - connected layer
layer5_weights = tf.Variable(tf.truncated_normal([400, 120], stddev=0.1))
layer5_biases = tf.Variable(tf.constant(1.0, shape=[120]))

# Fully - connected layer
layer6_weights = tf.Variable(tf.truncated_normal([120, 84], stddev=0.1))
layer6_biases = tf.Variable(tf.constant(1.0, shape=[84]))

# Fully - connected layer
layer7_weights = tf.Variable(tf.truncated_normal([84, 10], stddev=0.1))
layer7_biases = tf.Variable(tf.constant(1.0, shape=[10]))


def resize_all_img_in_dataset(dataset, to_size):
    ds_size = dataset.shape[0]
    np_add = np.zeros((ds_size, image_size, to_size[1] - image_size, 1))
    res = np.append(dataset, np_add, 2)
    np_add = np.zeros((ds_size, to_size[0] - image_size, to_size[1], 1), )
    res = np.append(res, np_add, 1)
    return res


def inference(data, keep_prob=None):
    with tf.variable_scope('conv1') as scope:
        # 1
        conv = tf.nn.conv2d(input=data, filter=layer1_weights, strides=[1, 1, 1, 1],
                            padding='VALID')
        hidden = tf.nn.relu(conv + layer1_biases)
        # 2
        conv = tf.nn.max_pool(value=hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer2_biases)

    with tf.variable_scope('conv2') as scope:
        # 3
        conv = tf.nn.conv2d(input=hidden,
                            filter=layer3_weights,
                            strides=[1, 1, 1, 1],
                            padding='VALID')
        hidden = tf.nn.relu(conv + layer3_biases)
        # 4
        conv = tf.nn.max_pool(value=hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer4_biases)

    with tf.variable_scope('flatten') as scope:
        # 5
        flatten_layer = tf.contrib.layers.flatten(hidden)

    with tf.variable_scope('fc1') as scope:
        # 6
        fc1 = tf.nn.relu(tf.matmul(flatten_layer, layer5_weights) + layer5_biases)
        fc2 = tf.nn.dropout(fc1, keep_prob)
    with tf.variable_scope('fc2') as scope:
        # 7
        fc2 = tf.nn.relu(tf.matmul(fc1, layer6_weights) + layer6_biases)
        fc2 = tf.nn.dropout(fc2, keep_prob)
    with tf.variable_scope('fc3') as scope:
        # 8
        fc3 = tf.nn.relu(tf.matmul(fc2, layer7_weights) + layer7_biases)
    return fc3


def loss(y, logit):
    xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=y)
    loss = tf.reduce_mean(xentropy)
    return loss


def training(loss, global_step):
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluate(label, y):
    res = tf.equal(tf.argmax(label, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(res, tf.float32))
    tf.summary.scalar('validation_error', (1.0 - accuracy))
    return accuracy


def accuracy(predictions, labels):
    return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1) / predictions.shape[0])


def chunks(data, label, idx, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(data), n):
        shuffled_index = idx[i:i + n]
        yield data[shuffled_index], label[shuffled_index]


def get_data_and_labels():
    test_dataset, train_dataset, validation_dataset = load_test_train_validation_ds(
        reshape=(-1, image_size, image_size, num_channels)
    )
    train_labels = convert_from_one_dim_labels(train_dataset.label, NUM_LABELS)

    train_labels = train_labels.astype(np.float32)
    train_data_resized = resize_all_img_in_dataset(train_dataset.data.astype(np.float32), (32, 32))

    validation_labels = convert_from_one_dim_labels(
        validation_dataset.label, NUM_LABELS
    ).astype(np.float32)

    validation_data_resized = resize_all_img_in_dataset(
        validation_dataset.data.astype(np.float32), (32, 32)
    )

    test_labels = convert_from_one_dim_labels(test_dataset.label, NUM_LABELS)
    test_data_resized = resize_all_img_in_dataset(test_dataset.data, (32, 32))
    return {
        'test_labels': test_labels,
        'train_labels': train_labels,
        'validation_labels': validation_labels,
        'validation_data_resized': validation_data_resized,
        'test_data_resized': test_data_resized,
        'train_data_resized': train_data_resized
    }


if __name__ == '__main__':
    result = get_data_and_labels()

    sess = tf.Session()
    tf_train_data = tf.placeholder(shape=(None, image_size + 4, image_size + 4, num_channels),
                                   dtype='float32')

    tf_train_label = tf.placeholder(shape=[None, 10], dtype='float32')
    tf_valid_data = tf.constant(result['validation_data_resized'].astype(np.float32))
    tf_test_data = tf.constant(result['test_data_resized'].astype(np.float32))

    keep_prob = tf.placeholder(tf.float32)

    feed_dict = {tf_train_data: result['train_data_resized'],
                 tf_train_label: result['train_labels']}

    global_step = tf.Variable(0, name='global_step', trainable=False)

    inference_op = inference(tf_train_data, keep_prob)
    cost = loss(tf_train_label, inference_op)
    training_op = training(cost, global_step)
    evaluate_op = evaluate(tf_train_label, inference_op)

    valid_prediction = tf.nn.softmax(inference(tf_valid_data, keep_prob))
    test_prediction = tf.nn.softmax(inference(tf_test_data, keep_prob))

    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter('logs/', graph=sess.graph)
    idx = np.arange(len(result['train_labels']))
    np.random.shuffle(idx)
    for epoch in range(1001):
        random_indx = np.random.randint(1, result['train_data_resized'].shape[0], size=batch_size)
        train_data_chunk = result['train_data_resized'][random_indx]
        train_label_chunk = result['train_labels'][random_indx]

        feed_dict = {tf_train_data: train_data_chunk,
                     tf_train_label: train_label_chunk,
                     keep_prob: 0.7}
        gs = sess.run(global_step)
        c, t = sess.run([cost, training_op], feed_dict=feed_dict)
        e_training = sess.run(evaluate_op, feed_dict=feed_dict)
        c, s = sess.run([cost, summary_op], feed_dict=feed_dict)
        if epoch % 100 == 0:
            logging.info('Trainig accuracy: %.1f%%, step: %i', float(100 * e_training), epoch)
        if epoch % 500 == 0:
            evaluate_op_v = evaluate(result['validation_labels'], valid_prediction)
            ev = sess.run(evaluate_op_v, feed_dict={keep_prob: 1.0})
            logging.info('Validation accuracy: %.1f%%', float(100 * ev))
    test_op_v = evaluate(result['test_labels'], test_prediction)
    et = sess.run(test_op_v, feed_dict={keep_prob: 1.0})
    logging.info('Test accuracy: %.1f%%', float(100*et))
