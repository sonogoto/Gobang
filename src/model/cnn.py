#!/usr/bin/env python3

import numpy as np
import tensorflow as tf


l2 = tf.keras.regularizers.l2
relu = tf.keras.activations.relu
tanh = tf.keras.activations.tanh
sigmoid = tf.keras.activations.sigmoid
Conv2D = tf.keras.layers.Conv2D
BN = tf.keras.layers.BatchNormalization
add = tf.keras.layers.add
Dense = tf.keras.layers.Dense
Flatten = tf.keras.layers.Flatten
mse = tf.keras.losses.mse
cross_entropy = tf.keras.losses.categorical_crossentropy


def conv_block(x, **kwargs):
    with tf.name_scope('conv_block'):
        conv1 = Conv2D(
            filters=kwargs.get('filters', 32),
            kernel_size=kwargs.get('kernel_size', 3),
            padding='same',
            kernel_regularizer=l2(kwargs.get('l2_reg', .001))
        )(x)
        return relu(BN()(conv1))


def res_block(idx, x, **kwargs):
    with tf.name_scope('res_block_%d' % idx):
        conv1 = Conv2D(
            filters=kwargs.get('filters', 32),
            kernel_size=kwargs.get('kernel_size', 3),
            padding='same',
            kernel_regularizer=l2(kwargs.get('l2_reg', .001))
        )(x)
        bn1 = BN()(conv1)
        act1 = relu(bn1)
        conv2 = Conv2D(
            filters=kwargs.get('filters', 32),
            kernel_size=kwargs.get('kernel_size', 3),
            padding='same',
            kernel_regularizer=l2(kwargs.get('l2_reg', .001))
        )(act1)
        bn1 = BN()(conv2)
        return relu(add([x, bn1]))


def dual_header(features, board_size, **kwargs):
    def header_v():
        with tf.name_scope('value_header'):
            conv = Conv2D(
                filters=kwargs.get('filters', 1),
                kernel_size=kwargs.get('kernel_size', 1),
                padding='same',
                kernel_regularizer=l2(kwargs.get('l2_reg', .001))
            )(features)
            bn = BN()(conv)
            act = relu(bn)
            fc1 = Dense(
                units=kwargs.get('fc_units', 256),
                kernel_initializer='glorot_normal',
                activation=relu
            )(Flatten()(act))
            return Dense(
                units=1,
                kernel_initializer='glorot_normal',
                activation=tanh
            )(fc1)

    def header_p():
        with tf.name_scope('policy_header'):
            conv = Conv2D(
                filters=kwargs.get('filters', 2),
                kernel_size=kwargs.get('kernel_size', 1),
                padding='same',
                kernel_regularizer=l2(kwargs.get('l2_reg', .001))
            )(features)
            bn = BN()(conv)
            act = relu(bn)
            return Dense(
                units=board_size**2,
                kernel_initializer='glorot_normal',
                activation=sigmoid
            )(Flatten()(act))

    return header_v(), header_p()


def create_network(features, num_res_blocks, board_size, **kwargs):
    x = conv_block(features, **kwargs)
    for i in range(num_res_blocks):
        x = res_block(i+1, x, **kwargs)
    return dual_header(x, board_size, **kwargs)


def model_fn(features, labels, mode, params, config):
    v, p = create_network(features, **params)
    predictions = {'value': v, 'policy': p}
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions
        )
    mse_loss = tf.reduce_mean(mse(labels['value'], v))
    cross_entropy_loss = tf.reduce_mean(
        cross_entropy(labels['policy'], p)
    )
    loss = (params.get('v_weight', .5) * mse_loss
            + params.get('p_weight', .5) * cross_entropy_loss)
    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = tf.train.AdamOptimizer(
            learning_rate=params.get('learning_rate', .001)
        ).minimize(
            loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op
        )
    eval_metric_ops = {
        'mse_loss': mse_loss,
        'cross_entropy_loss': cross_entropy_loss,
        'accuracy': tf.metrics.accuracy(
            tf.argmax(labels, axis=1),
            tf.argmax(p, axis=1)
        )
    }
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        eval_metric_ops=eval_metric_ops
    )


if __name__ == '__main__':
    x_in = tf.constant(
        np.random.randint(10, size=(10, 3, 3, 2)),
        dtype=tf.float32
    )
    y_out = create_network(x_in, 3, size=3)
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(logdir='../../log', graph=sess.graph)
        _ = sess.run(tf.global_variables_initializer())
        print(sess.run(y_out))
        writer.close()

