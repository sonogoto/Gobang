#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from .gcn_torch import GCNNet
from .utils import softmax
mse = tf.keras.losses.mse
cross_entropy = tf.nn.softmax_cross_entropy_with_logits


class BaseEvaluator(object):

    def predict(self, X):
        raise NotImplementedError(
            'method `predict` not implemented in class [%s]'
            % self.__class__.__name__
        )


class EstimatorEvaluator(BaseEvaluator):

    def __init__(self, est):
        self._est = est

    def predict(self, X):
        # Input graph does not use tf.data.Dataset
        # or contain a QueueRunner. That means predict
        # yields forever. This is probably a mistake.
        pred = self._est.predict(
            input_fn=lambda: tf.cast(X, dtype=tf.float32),
            yield_single_examples=False
        )
        # return to ensure only predict once
        for rst in pred:
            return rst['value'], rst['policy']


class RandomEvaluator(BaseEvaluator):

    def __init__(self, **kwargs):
        pass

    def predict(self, X):
        v = np.zeros(shape=(X.shape[0], 1))
        p = np.random.random(size=(X.shape[0], X.shape[1]**2))
        return v, p


class TFEvaluator(BaseEvaluator):

    def __init__(self, model_func, **kwargs):
        G = tf.Graph()
        with G.as_default():
            with tf.device(kwargs['device']):
                self._sess = tf.Session(
                    config=tf.ConfigProto(
                        gpu_options=tf.GPUOptions(allow_growth=True)
                    )
                )
                self._x = tf.placeholder(
                    shape=(None,
                           kwargs['board_size']**2,
                           kwargs['feature_dim']),
                    dtype=tf.float32
                )
                self._y = {
                    'policy': tf.placeholder(
                        shape=(None, kwargs['board_size']**2),
                        dtype=tf.float32,
                        name='policy'
                    ),
                    'value': tf.placeholder(
                        shape=(None, 1),
                        dtype=tf.float32,
                        name='value'
                    )
                }
                self._vout, self._pout = model_func(**kwargs)(self._x)
                self._v_loss, self._p_loss = self._compute_loss()
                self._loss = kwargs.get('value_loss_weight', .5) * self._v_loss \
                             + kwargs.get('policy_loss_weight', .5) * self._p_loss
                self._train_op = tf.train.AdamOptimizer(
                ).minimize(
                    self._loss,
                    global_step=tf.train.get_global_step()
                )
            self._saver = tf.train.Saver()
            ckpt = kwargs.get('ckpt', None)
            if ckpt:
                self._saver.restore(self._sess, ckpt)
            else:
                self._sess.run(tf.global_variables_initializer())

    def _compute_loss(self):
        value_loss = tf.reduce_mean(mse(self._y['value'], self._vout))
        policy_loss = tf.reduce_mean(
            cross_entropy(labels=self._y['policy'], logits=self._pout)
        )
        return value_loss, policy_loss

    def predict(self, x):
        assert x.shape[1] == x.shape[2], 'axis-1 and axis-2 must be equal'
        x = np.asarray(x, dtype=np.float32).reshape((-1, x.shape[1]**2, x.shape[-1]))
        v_out, p_out = self._sess.run(
            (self._vout, self._pout),
            feed_dict={self._x: x}
        )
        p_out = softmax(p_out)
        # set illegal position to 0
        # idx = np.where(x != 0)
        # p_out[idx[0], idx[1]] = 0.
        return v_out, p_out

    def train_(self, x, value, policy):
        assert x.shape[1] == x.shape[2], 'axis-1 and axis-2 must be equal'
        if isinstance(x, np.ndarray):
            x = np.asarray(x, dtype=np.float32).reshape((-1, x.shape[1] ** 2, x.shape[-1]))
        elif isinstance(x, tf.Tensor):
            x = tf.reshape(x, shape=(-1, x.shape[1]*x.shape[1], x.shape[-1]))
        else:
            raise RuntimeError('unsupported type of x: %s' % x.__class__.__name__)
        return self._sess.run(
            fetches=(self._v_loss, self._p_loss, self._loss, self._train_op),
            feed_dict={
                self._x: x,
                self._y['policy']: policy,
                self._y['value']: value
            }
        )

    def save(self, ckpt):
        return self._saver.save(self._sess, ckpt)

    def __del__(self):
        self._sess.close()


class TFFunctionalEvaluator(BaseEvaluator):
    def __init__(self, model_func, **kwargs):
        device = kwargs.get('device', '/cpu:0')
        board_size = kwargs['board_size']
        with tf.device(device):
            self._input = tf.placeholder(
                shape=(None, board_size, board_size, 1),
                dtype=tf.float32
            )
            self._out = model_func(self._input, **kwargs)
        self._sess = tf.Session(
            config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        )
        self._sess.run(tf.global_variables_initializer())

    def predict(self, X):
        return self._sess.run(
            self._out,
            feed_dict={self._input: X}
         )

    def __del__(self):
        self._sess.close()


class TorchEvaluator(GCNNet):
    pass


if __name__ == '__main__':
    pass
