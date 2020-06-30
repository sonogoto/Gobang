#!/usr/bin/env python3

from multiprocessing import Process, Queue
from ..common.log import *


class EvaluatorProcess(Process):

    def __init__(self,
                 evaluator,
                 request_queue,
                 result_queues,
                 **kwargs):
        super(EvaluatorProcess, self).__init__(name='EvaluatorProcess', **kwargs)
        self._evaluator = evaluator
        self._req_queue = request_queue
        self._rst_queues = result_queues

    def run(self):
        while True:
            # debug('request queue length [%d]' % self._req_queue.qsize())
            req = self._req_queue.get()
            rst_queue_idx = req[0]
            self._rst_queues[rst_queue_idx].put(
                self._evaluator.predict(req[1])
            )


if __name__ == '__main__':
    pass
