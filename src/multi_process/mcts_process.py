#!/usr/bin/env python3

from multiprocessing import Process
from ..common.log import *
from ..model.train_model import self_play
import numpy as np
from ..model.rotation import ROTATION


class MCTSProcess(Process):

    def __init__(self,
                 game,
                 task_queue,
                 resource_queue,
                 request_queue,
                 result_queues,
                 training_queue,
                 id_,
                 rotate=False,
                 sample_range=0,
                 **kwargs):
        '''
        :param game: the game to execute mcts.
        :param task_queue: task queue where process will fetch task.
        :param resource_queue: resource queue to indicate which result queue is available.
        :param request_queue: request queue which process will put request into.
        :param result_queues: result queue where process will get result.
        :param training_queue: train data queue into which process will put train data, including
                                states, values and policies.
        :param id_: the process ID.
        :param rotate: whether to add rotation, for GCN, there is no need to add rotation, since
                        GCN is NATURALLY rotate/flip-invariant. For value estimation, the input
                        of value header is max/avg pooling from node embeddings, rotation/flip do
                        not change it. For policy estimation, the prior probability of each valid
                        position is the function of the corresponding node embedding which specified
                        by local structure which is invariant to rotation/flip.
        :param sample_range: int, sample a record from the last `sample_range` records, default
                            is 0, which sample randomly from all game records.
        :param kwargs: for compatibility.
        '''
        name = 'MCTSProcess-%d' % id_
        super(MCTSProcess, self).__init__(name=name, **kwargs)
        self._game = game
        self._task_queue = task_queue
        self._res_queue = resource_queue
        self._req_queue = request_queue
        self._rst_queues = result_queues
        self._training_queue = training_queue
        self._rotate = rotate
        self._sample_range = sample_range
        self._rst_queue_idx = None

    def run(self):
        while True:
            batch_no, sample_no = self._task_queue.get()
            info('got task, batch NO.: [%d], sample NO.: [%d]' % (batch_no, sample_no))
            game_records, winner = self._self_play()
            if len(game_records) < 1:
                info('game records is empty, put task back to queue')
                self._task_queue.put((batch_no, sample_no))
                continue
            if self._sample_range > 0:
                idx = np.random.randint(self._sample_range)
            else:
                idx = np.random.randint(game_records.__len__())
            record = game_records[-self._sample_range:][idx]
            if self._rotate:
                self._add_rotation(record, winner)
            else:
                self._training_queue.put(
                    (record[0], record[2], -1 * winner * record[1])
                )
            info('task finished, batch NO.: [%d], sample NO.: [%d]' % (batch_no, sample_no))

    def predict(self, X):
        self._rst_queue_idx = self._res_queue.get()
        # debug('resource acquired, queue index: [%d]' % self._rst_queue_idx)
        assert self._rst_queue_idx is not None, fatal('No resource')
        self._req_queue.put(
            (self._rst_queue_idx, X)
        )
        v, p = self._rst_queues[self._rst_queue_idx].get()
        self._res_queue.put(self._rst_queue_idx)
        # debug('resource released, queue index: [%d]' % self._rst_queue_idx)
        self._rst_queue_idx = None
        return v, p

    def _self_play(self, **kwargs):
        game_status = self._game.get_game_status()
        game_records, winner = self_play(self._game, evaluator=self)
        self._game.reset(game_status)
        return game_records, winner

    def _add_rotation(self, record, winner):
        _shape = record[0].shape[0]
        policy = np.reshape(record[2], newshape=(_shape, _shape))
        for rotate in ROTATION:
            self._training_queue.put(
                (rotate(record[0]),
                 np.reshape(rotate(policy), newshape=record[2].shape),
                 -1 * winner * record[1])
            )
