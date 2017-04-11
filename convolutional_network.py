import logging

import numpy as np
import tensorflow as tf

from loader import load_test_train_validation_ds, IMAGE_YSIZE, IMAGE_XSIZE, NUM_LABELS
from utils import convert_from_one_dim_labels, LOGGING_FORMAT


logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)


batch_size = 64
patch_size = 5
depth = 16
num_hidden = 64
num_channels = 1
image_size = IMAGE_YSIZE


# Convolutional layer
layer1_weights = tf.Variable(tf.truncated_normal(
    [patch_size, patch_size, num_channels, depth], stddev=0.1))

layer1_biases = tf.Variable(tf.zeros([depth]))

# Convolutional layer
layer2_weights = tf.Variable(tf.truncated_normal(
    [patch_size, patch_size, depth, depth], stddev=0.1))
layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))

layer3_weights = tf.Variable(tf.truncated_normal(
    [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))

layer4_weights = tf.Variable(tf.truncated_normal(
    [num_hidden, NUM_LABELS], stddev=0.1))
layer4_biases = tf.Variable(tf.constant(1.0, shape=[NUM_LABELS]))


def inference(data, keep_prob=None):
    # 1
    conv = tf.nn.conv2d(input=data,
                        filter=layer1_weights,
                        strides=[1, 2, 2, 1],
                        padding='SAME')
    hidden = tf.nn.relu(conv + layer1_biases)

    # 2
    conv = tf.nn.max_pool(value=hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer2_biases)
    # 3
    shape = hidden.get_shape().as_list()
    reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
    hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
    # 4
    return tf.matmul(hidden, layer4_weights) + layer4_biases


def loss(y, logit):
    xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=y)
    loss = tf.reduce_mean(xentropy)
    vars = tf.trainable_variables()
    # regularizer = 0.001*tf.add_n([tf.nn.l2_loss(x) for x in vars if 'bias' not in x.name])
    return loss #+ regularizer


def training(loss, global_step):
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.AdamOptimizer(0.003)
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


if __name__ == '__main__':
    test_dataset, train_dataset, validation_dataset = load_test_train_validation_ds(
        reshape=(-1, image_size, image_size, num_channels)
    )
    train_labels = convert_from_one_dim_labels(train_dataset.label, NUM_LABELS)

    train_labels = train_labels.astype(np.float32)
    train_data = train_dataset.data.astype(np.float32)

    validation_labels = convert_from_one_dim_labels(validation_dataset.label, NUM_LABELS)

    validation_labels = validation_labels.astype(np.float32)
    validation_data = validation_dataset.data.astype(np.float32)

    test_labels = convert_from_one_dim_labels(test_dataset.label, NUM_LABELS)


    sess = tf.Session()
    tf_train_data = tf.placeholder(shape=(batch_size, image_size, image_size, num_channels),
                                   dtype='float32')

    tf_train_label = tf.placeholder(shape=[None, 10], dtype='float32')

    tf_valid_data = tf.constant(validation_data.astype(np.float32))
    tf_test_data = tf.constant(test_dataset.data.astype(np.float32))

    keep_prob = tf.placeholder(tf.float32)

    feed_dict = {tf_train_data: train_data, tf_train_label: train_labels}

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
    idx = np.arange(len(train_labels))
    np.random.shuffle(idx)
    for epoch in range(5001):
        random_indx = np.random.randint(1, train_dataset.data.shape[0],
                                        size=batch_size)
        train_data_chunk = train_dataset.data[random_indx]
        train_label_chunk = train_labels[random_indx]

        feed_dict = {tf_train_data: train_data_chunk,
                     tf_train_label: train_label_chunk,
                     keep_prob: 1}
        gs = sess.run(global_step)
        c, t = sess.run([cost, training_op], feed_dict=feed_dict)
        e_training = sess.run(evaluate_op, feed_dict=feed_dict)
        c, s = sess.run([cost, summary_op], feed_dict=feed_dict)
        if epoch % 100 == 0:
            logging.info('Trainig accuracy: %.1f%%, step: %i', float(100 * e_training), epoch)
        if epoch % 500 == 0:
            evaluate_op_v = evaluate(validation_labels, valid_prediction)
            ev = sess.run(evaluate_op_v)
            logging.info('Validation accuracy: %.1f%%', float(100 * ev))
    test_op_v = evaluate(test_labels, test_prediction)
    et = sess.run(test_op_v)
    logging.info('Test accuracy: %.1f%%', float(100*et))
