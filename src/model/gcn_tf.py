#!/usr/bin/env python3

import tensorflow as tf
from .utils import build_edge_idx
import numpy as np
from tensorflow.keras import activations


def create_gcn(features, feature_dim, board_size, gcn_layers, **kwargs):
    adj_matrix = tf.constant(
        _build_adj_matrix(board_size),
        name='adj_matrix',
        dtype=tf.float32
    )
    activation = activations.get(kwargs.get('activation', 'relu'))
    with tf.name_scope('gcn_layers'):
        x = _gcn_layer(
            index=0,
            input_=tf.reshape(features, (-1, board_size**2, feature_dim)),
            input_shape=feature_dim,
            output_shape=gcn_layers[0],
            adj_matrix=adj_matrix,
            activation=activation
        )
        for idx in range(gcn_layers.__len__()-1):
            x = _gcn_layer(
                index=idx+1,
                input_=x,
                input_shape=gcn_layers[idx],
                output_shape=gcn_layers[idx+1],
                adj_matrix=adj_matrix,
                activation=activation
            )
    with tf.name_scope('policy_header'):
        weight_p = tf.get_variable(
            shape=(gcn_layers[-1], 1),
            initializer=tf.initializers.glorot_normal(),
            name='policy_weight',
        )
        b_p = tf.get_variable(
            shape=(1,),
            initializer=tf.initializers.zeros(),
            name='policy_bias'
        )
        p = tf.reshape(
            tf.tensordot(x, weight_p, [[2, ], [0, ]]) + b_p,
            shape=(-1, board_size**2)
        )
    pooling = kwargs.get('pooling', 'avg')
    with tf.name_scope('value_header'):
        if pooling == 'avg':
            x_value = tf.reduce_mean(x, axis=1)
        else:
            x_value = tf.reduce_max(x, axis=1)
        weight_v = tf.get_variable(
            shape=(gcn_layers[-1], 1),
            initializer=tf.initializers.glorot_normal(),
            name='value_weight'
        )
        b_v = tf.get_variable(
            shape=(1,),
            initializer=tf.initializers.zeros(),
            name='value_bias'
        )
        # v = tf.tensordot(x_value, weight_v, [[2, ], [0, ]]) + b_v
        v = tf.matmul(x_value, weight_v) + b_v
    return v, p


def _build_adj_matrix(board_size):
    edge_idx = build_edge_idx(board_size)
    adj_matrix = np.eye(N=board_size**2)
    for i, j in zip(edge_idx[0], edge_idx[1]):
        adj_matrix[i, j] = 1
    return _normalize(adj_matrix)


def _normalize(adj_matrix):
    degree_inv = np.reshape(
        np.power(adj_matrix.sum(axis=0), -.5),
        newshape=(-1, 1)
    )
    return np.dot(degree_inv, degree_inv.T) * adj_matrix


def _gcn_layer(index, input_, input_shape, output_shape, adj_matrix, activation=None):
    with tf.name_scope('gcn_%d' % index):
        weight = tf.get_variable(
            shape=(input_shape, output_shape),
            initializer=tf.initializers.truncated_normal(stddev=.01),
            name='gcn_%d_weight' % index
        )
        b = tf.get_variable(
            shape=(output_shape,),
            initializer=tf.initializers.zeros(),
            name='gcn_%d_bias' % index
        )
        out = tf.tensordot(adj_matrix, input_, axes=[[1, ], [1, ]])
        out = tf.tensordot(tf.transpose(out, [1, 0, 2]), weight, axes=[[2, ], [0, ]]) + b
        if activation is None:
            return out
        else:
            return activation(out)


class GCNLayer(tf.Module):

    def __init__(self, input_features, output_features, name=None, activation=None):
        super(GCNLayer, self).__init__(name)
        self.weight = tf.Variable(tf.initializers.truncated_normal(
            stddev=.01)(shape=(input_features, output_features))
        )
        self.bias = tf.Variable(
            tf.initializers.zeros()(shape=(output_features,))
        )
        self.activation = activations.get(activation)

    def __call__(self, x, adj_matrix):
        y = tf.tensordot(adj_matrix, x, axes=[[1, ], [1, ]])
        y = tf.tensordot(
            tf.transpose(y, [1, 0, 2]),
            self.weight,
            axes=[[2, ], [0, ]]
        ) + self.bias
        return self.activation(y)


class PolicyHeader(tf.Module):

    def __init__(self, input_features, board_size, name=None):
        super(PolicyHeader, self).__init__(name)
        self._board_size = board_size
        self.weight = tf.Variable(tf.initializers.glorot_normal()(
            shape=(input_features, 1)
        ))
        self.bias = tf.Variable(
            tf.initializers.zeros()(shape=(1,))
        )

    def __call__(self, x):
        return tf.reshape(
            tf.tensordot(x, self.weight, [[2, ], [0, ]]) + self.bias,
            shape=(-1, self._board_size ** 2)
        )


class ValueHeader(tf.Module):
    def __init__(self, input_features, name=None):
        super(ValueHeader, self).__init__(name)
        self.weight = tf.Variable(tf.initializers.glorot_normal()(
            shape=(input_features, 1)
        ))
        self.bias = tf.Variable(
            tf.initializers.zeros()(shape=(1,))
        )

    def __call__(self, x, pooling):
        if pooling == 'avg':
            x = tf.reduce_mean(x, axis=1)
        elif pooling == 'max':
            x = tf.reduce_max(x, axis=1)
        else:
            raise RuntimeError('unrecognized pooling')
        return tf.tanh(tf.matmul(x, self.weight) + self.bias)


class GCNNet(tf.Module):

    def __init__(self,
                 board_size,
                 feature_dim,
                 gcn_layers,
                 name=None,
                 activation=None,
                 **kwargs):
        super(GCNNet, self).__init__(name)
        self._feature_dim = feature_dim
        self._board_size = board_size
        self._adj_matrix = tf.constant(
            _build_adj_matrix(board_size),
            dtype=tf.float32
        )
        self._gcn_layers = []
        dims = [feature_dim, ] + gcn_layers
        for idx in range(gcn_layers.__len__()):
            self._gcn_layers.append(
                GCNLayer(
                    input_features=dims[idx],
                    output_features=dims[idx+1],
                    name='gcn_%d' % idx,
                    activation=activation
                )
            )
        self._policy_header = PolicyHeader(
            input_features=gcn_layers[-1],
            board_size=board_size,
            name='policy_header'
        )
        self._value_header = ValueHeader(
            input_features=gcn_layers[-1],
            name='value_header'
        )

    def __call__(self, x, pooling='max'):
        for layer in self._gcn_layers:
            x = layer(x, self._adj_matrix)
        return self._value_header(x, pooling), self._policy_header(x)


if __name__ == '__main__':
    gcn = GCNNet(
        board_size=3,
        feature_dim=8,
        gcn_layers=[16, 16],
        name='gcn_model'
    )
    x = tf.constant(
        np.random.randint(10, size=(20, 9, 8)),
        dtype=tf.float32
    )
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 从保存的变量中恢复时，不需要再初始化
        # saver.restore(sess, "/tmp/model.ckpt")
        # print("Model restored.")
        v, p = sess.run(gcn(x))
        print(v.shape, p.shape)
        save_path = saver.save(sess, "/tmp/model.ckpt")
        print("Model saved in path: %s" % save_path)
