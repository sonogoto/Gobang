#!/usr/bin/env python3

import numpy as np
from .exceptions import NoSpaceLeft


class Board(object):

    def __init__(self, board_size=15, init_state=None):
        if init_state is None:
            self._board_state = np.zeros(
                shape=(board_size, board_size),
                dtype=np.int
            )
        else:
            assert init_state.shape.__len__() == 2, \
                '`init_state` must be a 2-d numpy array'
            assert board_size == init_state.shape[0] == init_state.shape[1], \
                'the size of 2 axes of `init_state` must equal to `board_size`'
            self._board_state = init_state.copy()

    def place(self, row_idx, col_idx, color):
        assert self._board_state[row_idx, col_idx] == 0, \
            'can not place at a position occupied'
        self._board_state[row_idx, col_idx] = color

    def check_board_state(self):
        if abs(self._board_state).sum() >= self._board_state.shape[0] ** 2:
            raise NoSpaceLeft('there is no space left on the board to play')

    @property
    def board_state(self):
        return self._board_state

    @property
    def board_size(self):
        return self._board_state.shape[0]

    def reset(self, board_state):
        self._board_state = board_state
