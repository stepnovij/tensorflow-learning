import os
import logging

import numpy as np
import tensorflow as tf

from loader import load_test_train_validation_ds, IMAGE_YSIZE, IMAGE_XSIZE, NUM_LABELS
from utils import convert_from_one_dim_labels, LOGGING_FORMAT


logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)


def layer(x, w_shape, bias_shape):
    weight_init = tf.random_normal_initializer(stddev=(1.0 / w_shape[0]) ** 0.5)
    bias_init = tf.constant_initializer(value=0)
    w = tf.get_variable('w', w_shape, initializer=weight_init, dtype='float32')
    b = tf.get_variable('b', bias_shape, initializer=bias_init, dtype='float32')
    return tf.matmul(x, w) + b


def inference(x, keep_prob):
    with tf.variable_scope('hidden_1'):
        output = tf.nn.relu(layer(x, [784, 256], [256]))
        output_dropout = tf.nn.dropout(output, keep_prob)
    with tf.variable_scope('hidden_2'):
        output_second = tf.nn.relu(layer(output_dropout, [256, 256], [256]))
        output_second = tf.nn.dropout(output_second, keep_prob)
    with tf.variable_scope('hidden_3'):
        output_third = tf.nn.relu(layer(output_second, [256, 256], [256]))
        output_third = tf.nn.dropout(output_third, keep_prob)
    with tf.variable_scope('output'):
        res = layer(output_third, [256, 10], [10])
    return res


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


def chunks(data, label, idx, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(data), n):
        shuffled_index = idx[i:i + n]
        yield data[shuffled_index], label[shuffled_index]


if __name__ == '__main__':
    test_dataset, train_dataset, validation_dataset = load_test_train_validation_ds()
    train_labels = convert_from_one_dim_labels(train_dataset.label, NUM_LABELS)

    train_labels = train_labels.astype(np.float32)
    train_data = train_dataset.data.astype(np.float32)

    validation_labels = convert_from_one_dim_labels(validation_dataset.label, NUM_LABELS)

    validation_labels = validation_labels.astype(np.float32)
    validation_data = validation_dataset.data.astype(np.float32)

    test_labels = convert_from_one_dim_labels(test_dataset.label, NUM_LABELS)

    sess = tf.Session()
    x = tf.placeholder(shape=[None, 784], dtype='float32')
    y = tf.placeholder(shape=[None, 10], dtype='float32')
    keep_prob = tf.placeholder(tf.float32)

    feed_dict = {x: train_data, y: train_labels}

    global_step = tf.Variable(0, name='global_step', trainable=False)
    inference_op = inference(x, keep_prob)
    cost = loss(y, inference_op)
    training_op = training(cost, global_step)
    evaluate_op = evaluate(y, inference_op)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter('logs/', graph=sess.graph)
    idx = np.arange(len(train_labels))
    np.random.shuffle(idx)
    for epoch in range(25):
        for train_data_chunk, train_label_chunk in chunks(train_data, train_labels, idx, 200):
            # random_indx = np.random.randint(1, train_labels.shape[0], size=100)
            # feed_dict = {x: train_data[random_indx], y: train_labels[random_indx]}
            feed_dict = {x: train_data_chunk, y: train_label_chunk, keep_prob: 0.8}
            c, t = sess.run([cost, training_op], feed_dict=feed_dict)
        if epoch % 2 == 0:

            e_training = sess.run(evaluate_op, feed_dict=feed_dict)

            e_validation = sess.run(evaluate_op, feed_dict={
                x: validation_data, y: validation_labels, keep_prob: 0.8})

            c, s = sess.run([cost, summary_op], feed_dict=feed_dict)
            saver.save(sess, 'logs/goodfellow', global_step=global_step)
            logging.info(
                'Cost : %s Training accuracy: %s Validation accuracy: %s' % (
                    c, 100*e_training, 100*e_validation)
            )
            summary_writer.add_summary(s, epoch)
    e_test = sess.run(evaluate_op, feed_dict={x: test_dataset.data, y: test_labels, keep_prob: 1.0})
    # 96.6700017452
    logging.info('Testing accuracy: %s' % (100 * e_test))
