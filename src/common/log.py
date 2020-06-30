#!/usr/bin/env python3


import logging
import sys
import time
fmt = '[%(levelname)s]%(asctime)s [%(processName)s] %(filename)s:%(lineno)d %(message)s'
logging.basicConfig(format=fmt)
logger = logging.getLogger(__name__)
LOGGER_LEVEL = 'INFO'
logger.setLevel(LOGGER_LEVEL)

info = logger.info
debug = logger.debug
warning = logger.warning
error = logger.error
fatal = logger.fatal


def test_log():
    debug('this is a debug')
    time.sleep(.1)
    info('this is an info')
    time.sleep(.1)
    warning('this is a warning')
    time.sleep(.1)
    error('this is an ERROR')
    time.sleep(.1)
    fatal('this is a fatal')
    time.sleep(.1)


if __name__ == '__main__':
    from multiprocessing import Process
    p = []
    for i in range(10):
        p.append(Process(target=test_log, name='test_log_process_%d' % i))
    for p_ in p:
        p_.start()
    for p_ in p:
        p_.join()
