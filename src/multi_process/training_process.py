#!/usr/bin/env python3

from multiprocessing import Process
import numpy as np
from ..common.log import *
import _pickle as pk


class TrainProcess(Process):

    def __init__(self,
                 model,
                 batches,
                 batch_size,
                 task_queue,
                 training_queue,
                 ckpt_path,
                 dump_data=True,
                 **kwargs):
        super(TrainProcess, self).__init__(name='TrainProcess', **kwargs)
        self._model = model
        self._batches = batches
        self._batch_size = batch_size
        self._task_queue = task_queue
        self._training_queue = training_queue
        self._ckpt_path = ckpt_path
        self._dump_data = dump_data

    def run(self):
        for batch_no in range(self._batches):
            info('start the %d-th batch' % batch_no)
            for sample_no in range(self._batch_size):
                self._task_queue.put((batch_no, sample_no))
            states, values, policies = [], [], []
            while len(values) < self._batch_size:
                s_t, p_t, v_t = self._training_queue.get()
                states.append(s_t)
                values.append(v_t)
                policies.append(p_t)
            features = np.expand_dims(states, axis=-1)
            values = np.expand_dims(values, axis=-1)
            policies = np.array(policies)
            if self._dump_data:
                with open('./train_data/features-%d.pkl' % batch_no, 'wb') as fw:
                    pk.dump(features, fw)
                with open('./train_data/values-%d.pkl' % batch_no, 'wb') as fw:
                    pk.dump(values, fw)
                with open('./train_data/policies-%d.pkl' % batch_no, 'wb') as fw:
                    pk.dump(policies, fw)
            v_loss, p_loss, loss, _ = self._model.train_(features, values, policies)
            info('the %d-th batch finished, loss: %f, value loss: %f, policy loss: %f'
                 % (batch_no, loss, v_loss, p_loss))
        self._model.save(self._ckpt_path)
