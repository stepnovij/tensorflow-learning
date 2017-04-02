import os
import logging

import matplotlib.pyplot as plt
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


def inference(x):
    with tf.variable_scope('hidden_1'):
        output = tf.nn.relu(layer(x, [784, 256], [256]))
    with tf.variable_scope('hidden_2'):
        output_second = tf.nn.relu(layer(output, [256, 256], [256]))
    with tf.variable_scope('hidden_3'):
        output_third = tf.nn.relu(layer(output_second, [256, 256], [256]))
    with tf.variable_scope('output'):
        res = layer(output_third, [256, 10], [10])
    return res


def loss(y, logit):
    xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=y)
    loss = tf.reduce_mean(xentropy)
    return loss


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


def prepare_opt_weights():
    with tf.variable_scope('preparing_variables') as scope:
        global_step = tf.Variable(0, name='global_step', trainable=False)
        inference_op = inference(x)
        cost = loss(y, inference_op)
        training_op = training(cost, global_step)
        evaluate_op = evaluate(y, inference_op)
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        summary_op = tf.summary.merge_all()
        saver = tf.train.Saver()
        summary_writer = tf.summary.FileWriter('logs/', graph=sess.graph)
        if not os.path.exists('logs/goodfellow-3838.meta'):
            idx = np.arange(len(train_labels))
            np.random.shuffle(idx)

            for epoch in range(20):
                for train_data_chunk, train_label_chunk in chunks(train_data, train_labels, idx, 180):
                    # random_indx = np.random.randint(1, train_labels.shape[0], size=100)
                    # feed_dict = {x: train_data[random_indx], y: train_labels[random_indx]}
                    feed_dict = {x: train_data_chunk, y: train_label_chunk}
                    c, t = sess.run([cost, training_op], feed_dict=feed_dict)
                if epoch % 2 == 0:

                    e_training = sess.run(evaluate_op, feed_dict=feed_dict)

                    e_validation = sess.run(evaluate_op, feed_dict={
                        x: validation_data, y: validation_labels})

                    c, s = sess.run([cost, summary_op], feed_dict=feed_dict)
                    saver.save(sess, 'logs/goodfellow', global_step=global_step)
                    logging.info(
                        'Cost : %s Training accuracy: %s Validation accuracy: %s' % (
                            c, 100*e_training, 100*e_validation)
                    )
                    summary_writer.add_summary(s, epoch)
            e_test = sess.run(evaluate_op, feed_dict={x: test_dataset.data, y: test_labels})
            # 96.6700017452
            logging.info('Testing accuracy: %s' % (100 * e_test))

        scope.reuse_variables()
        var_list_opt = ["hidden_1/w", "hidden_1/b", "hidden_2/w", "hidden_2/b", "output/w",
                        "output/b"]
        var_list_opt = [tf.get_variable(v) for v in var_list_opt]
        saver.restore(sess, 'logs/goodfellow-3838')
    return var_list_opt


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
    feed_dict = {x: train_data, y: train_labels}
    var_list_opt = prepare_opt_weights()

    with tf.variable_scope('mlp_init') as scope:
        output_rand = inference(x)
        cost_rand = loss(y, output_rand)
        scope.reuse_variables()
        var_list_rand = ["hidden_1/w", "hidden_1/b", "hidden_2/w", "hidden_2/b", "output/w",
                         "output/b"]
        var_list_rand = [tf.get_variable(v) for v in var_list_rand]
        init_op = tf.variables_initializer(var_list_rand)
        sess.run(init_op)

    with tf.variable_scope("mlp_inter") as scope:
        alpha = tf.placeholder("float", [1, 1])

        h1_W_inter = var_list_opt[0] * (1 - alpha) + var_list_rand[0] * (alpha)
        h1_b_inter = var_list_opt[1] * (1 - alpha) + var_list_rand[1] * (alpha)
        h2_W_inter = var_list_opt[2] * (1 - alpha) + var_list_rand[2] * (alpha)
        h2_b_inter = var_list_opt[3] * (1 - alpha) + var_list_rand[3] * (alpha)
        o_W_inter = var_list_opt[4] * (1 - alpha) + var_list_rand[4] * (alpha)
        o_b_inter = var_list_opt[5] * (1 - alpha) + var_list_rand[5] * (alpha)

        h1_inter = tf.nn.relu(tf.matmul(x, h1_W_inter) + h1_b_inter)
        h2_inter = tf.nn.relu(tf.matmul(h1_inter, h2_W_inter) + h2_b_inter)
        o_inter = tf.nn.relu(tf.matmul(h2_inter, o_W_inter) + o_b_inter)

        cost_inter = loss(o_inter, y)
        tf.summary.scalar("interpolated_cost", cost_inter)

    summary_writer = tf.summary.FileWriter("linear_interp_logs/", graph=sess.graph)
    summary_op = tf.summary.merge_all()
    results = []
    for a in np.arange(-2, 2, 0.01):
        feed_dict = {
            x: validation_data,
            y: validation_labels,
            alpha: [[a]],
        }

        cost, summary_str = sess.run([cost_inter, summary_op], feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, (a + 2) / 0.01)
        results.append(cost)

    plt.plot(np.arange(-2, 2, 0.01), results, 'ro')
    plt.ylabel('Incurred Error')
    plt.xlabel('Alpha')
    plt.show()

