#!/usr/bin/env python3

import numpy as np


def build_edge_idx(board_size):
    edge_idx = [[], []]
    for row_idx in range(board_size):
        for col_idx in range(board_size):
            if row_idx == 0:
                if col_idx == 0:
                    # left-up corner
                    edge_idx[0].extend([row_idx * board_size + col_idx, ] * 3)
                    targets = [
                        row_idx * board_size + col_idx + 1,
                        (row_idx + 1) * board_size + col_idx,
                        (row_idx + 1) * board_size + col_idx + 1,
                    ]
                    edge_idx[1].extend(targets)
                elif col_idx == board_size - 1:
                    # right-up corner
                    edge_idx[0].extend([row_idx * board_size + col_idx, ] * 3)
                    targets = [
                        row_idx * board_size + col_idx - 1,
                        (row_idx + 1) * board_size + col_idx,
                        (row_idx + 1) * board_size + col_idx - 1,
                    ]
                    edge_idx[1].extend(targets)
                else:
                    # first row, but not corner
                    edge_idx[0].extend([row_idx * board_size + col_idx, ] * 5)
                    targets = [
                        row_idx * board_size + col_idx - 1,
                        row_idx * board_size + col_idx + 1,
                        (row_idx + 1) * board_size + col_idx - 1,
                        (row_idx + 1) * board_size + col_idx,
                        (row_idx + 1) * board_size + col_idx + 1,
                    ]
                    edge_idx[1].extend(targets)
            elif row_idx == board_size - 1:
                if col_idx == 0:
                    # left-down corner
                    edge_idx[0].extend([row_idx * board_size + col_idx, ] * 3)
                    targets = [
                        (row_idx - 1) * board_size + col_idx,
                        (row_idx - 1) * board_size + col_idx + 1,
                        row_idx * board_size + col_idx + 1,
                    ]
                    edge_idx[1].extend(targets)
                elif col_idx == board_size - 1:
                    # right-down corner
                    edge_idx[0].extend([row_idx * board_size + col_idx, ] * 3)
                    targets = [
                        (row_idx - 1) * board_size + col_idx,
                        (row_idx - 1) * board_size + col_idx - 1,
                        row_idx * board_size + col_idx - 1,
                    ]
                    edge_idx[1].extend(targets)
                else:
                    # bottom row, but not corner
                    edge_idx[0].extend([row_idx * board_size + col_idx, ] * 5)
                    targets = [
                        (row_idx - 1) * board_size + col_idx - 1,
                        (row_idx - 1) * board_size + col_idx,
                        (row_idx - 1) * board_size + col_idx + 1,
                        row_idx * board_size + col_idx - 1,
                        row_idx * board_size + col_idx + 1,
                    ]
                    edge_idx[1].extend(targets)
            elif col_idx == 0:
                # left most column, but not corner
                edge_idx[0].extend([row_idx * board_size + col_idx, ] * 5)
                targets = [
                    (row_idx - 1) * board_size + col_idx,
                    (row_idx - 1) * board_size + col_idx + 1,
                    row_idx * board_size + col_idx + 1,
                    (row_idx + 1) * board_size + col_idx,
                    (row_idx + 1) * board_size + col_idx + 1,
                ]
                edge_idx[1].extend(targets)
            elif col_idx == board_size - 1:
                # right most column, but not corner
                edge_idx[0].extend([row_idx * board_size + col_idx, ] * 5)
                targets = [
                    (row_idx - 1) * board_size + col_idx - 1,
                    (row_idx - 1) * board_size + col_idx,
                    row_idx * board_size + col_idx - 1,
                    (row_idx + 1) * board_size + col_idx - 1,
                    (row_idx + 1) * board_size + col_idx,
                ]
                edge_idx[1].extend(targets)
            else:
                # position not on bound
                edge_idx[0].extend([row_idx * board_size + col_idx, ] * 8)
                targets = [
                    (row_idx - 1) * board_size + col_idx - 1,
                    (row_idx - 1) * board_size + col_idx,
                    (row_idx - 1) * board_size + col_idx + 1,
                    row_idx * board_size + col_idx - 1,
                    row_idx * board_size + col_idx + 1,
                    (row_idx + 1) * board_size + col_idx - 1,
                    (row_idx + 1) * board_size + col_idx,
                    (row_idx + 1) * board_size + col_idx + 1,
                ]
                edge_idx[1].extend(targets)
    return edge_idx


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def softmax(x):
    tmp = np.exp(x)
    norm = np.sum(tmp, axis=1, keepdims=True)
    return tmp / norm


class FakeModel(object):

    def __init__(self):
        pass

    def train_(self, *args, **kwargs):
        pass

    def save(self, *args, **kwargs):
        pass

    def __getattr__(self, item):
        return item

