#!/usr/bin/env python3

from random import random


class BasePlayer(object):

    WHITE = 1
    BLACK = -1
    COLORS = (WHITE, BLACK)

    def __init__(self, id_, color):
        self._id = id_
        assert color in BasePlayer.COLORS, \
            '`color` accept only 1 or -1'
        self._color = color

    def play(self, board, **kwargs):
        raise NotImplementedError(
            'Method `play` not implemented in class [%s]' % self.__class__.__name__
        )

    @property
    def id(self):
        return self._id

    @property
    def color(self):
        return self._color


class ActingPlayer(BasePlayer):

    def play(self, board, **kwargs):
        board.check_board_state()
        row_idx = kwargs['idx'] // board.board_size
        col_idx = kwargs['idx'] % board.board_size
        return row_idx, col_idx


class RandomPlayer(BasePlayer):

    def __init__(self, id_, color):
        super(RandomPlayer, self).__init__(id_, color)

    def play(self, board, **kwargs):
        board.check_board_state()
        board_size = kwargs.get('board_size', None)
        if not board_size:
            board_size = board.board_size
        row_idx = int(board_size * random())
        col_idx = int(board_size * random())
        while board.board_state[row_idx, col_idx] != 0:
            row_idx = int(board_size * random())
            col_idx = int(board_size * random())
        return row_idx, col_idx


class ManualPlayer(RandomPlayer):

    DEFAULT_UNIT_LENGTH = 50

    def play(self, board, **kwargs):
        board.check_board_state()
        event = kwargs.get('event', None)
        if event is None:
            return super(ManualPlayer, self).play(board, **kwargs)
        unit_length = kwargs.get('unit_length',
                                 ManualPlayer.DEFAULT_UNIT_LENGTH)
        return self.__class__._get_nearest_position(board, event, unit_length)

    @staticmethod
    def _get_nearest_position(board, event, unit_length):

        def _look_around(row_idx_, col_idx_):
            min_dist = 1e6
            nearest_position = (row_idx_, col_idx_)
            if board.board_state[row_idx_, col_idx_] == 0:
                return nearest_position
            for ridx in range(board.board_size):
                for cidx in range(board.board_size):
                    if board.board_state[ridx, cidx] != 0:
                        continue
                    dist = (row_idx_-ridx)**2 + (col_idx_-cidx)**2
                    if dist < min_dist:
                        min_dist = dist
                        nearest_position = (ridx, cidx)
            return nearest_position

        row_idx = (event.y-unit_length)//unit_length
        # closer to next vertical position
        if event.y % unit_length > unit_length/2:
            row_idx += 1
        if row_idx < 0:
            row_idx = 0
        col_idx = (event.x-unit_length)//unit_length
        # closer to next horizontal position
        if event.x % unit_length > unit_length/2:
            col_idx += 1
        if col_idx < 0:
            col_idx = 0
        return _look_around(row_idx, col_idx)
