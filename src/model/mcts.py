#!/usr/bin/env python3


from math import sqrt
import numpy as np
import time
from scipy.stats import dirichlet
from ..common.log import *


C_PUCT = .5
DIRICHLET_WEIGHT = .25
DIRICHLET_ALPHA = .03


class Node(object):

    def __init__(self, parent=None, value=None, policy=None):
        self._parent = parent
        self._num_visit = 0
        self._value = value
        self._tot_value = .0
        self._avg_reward = .0
        self._policy = policy
        self._children = dict()

    @property
    def parent(self):
        return self._parent

    @property
    def num_visit(self):
        return self._num_visit

    @property
    def tot_value(self):
        return self._tot_value

    @property
    def avg_reward(self):
        return self._avg_reward

    @property
    def policy(self):
        return self._policy

    @property
    def children(self):
        return self._children

    @property
    def value(self):
        return self._value

    def update_num_visit(self, cnt=1):
        self._num_visit += cnt

    def update_tot_value(self, delta):
        self._tot_value += delta

    def update_avg_reward(self):
        self._avg_reward = self._tot_value / (self._num_visit - 1)

    def set_child(self, child_idx, child):
        self._children[child_idx] = child


def expand_node(node, game, evaluator):
    child_indexes = []
    child_states = []
    current_state = game.game_state
    color = game.current_player.color
    for row_idx, col_idx in zip(*np.where(game.game_state == 0)):
        child_indexes.append(row_idx * game.board.board_size + col_idx)
        child_states.append(current_state.copy())
        child_states[-1][row_idx, col_idx] = color
    assert child_states.__len__() >= 1, fatal('No space to expand node')
    X = np.expand_dims(child_states, axis=-1)
    v, p = evaluator.predict(color * X)
    for idx, child_idx in enumerate(child_indexes):
        node.set_child(
            child_idx,
            Node(node, value=v[idx][0], policy=p[idx].tolist())
        )


def backup(leaf_node):
    current_node = leaf_node.parent
    sign = 1.
    while current_node is not None:
        current_node.update_tot_value(
            leaf_node.value * sign
        )
        current_node.update_avg_reward()
        current_node = current_node.parent
        sign *= -1.


def sample_move(node):
    prob = dict()
    # parent.num_visit = sum(num_visit of children)+1
    # since search will return when visiting a node first
    # time.
    coef = sqrt(node.num_visit-1)
    for idx, p in enumerate(node.policy):
        if idx not in node.children:
            # prob[idx] = 0.
            continue
        elif node.children[idx].num_visit == 0:
            prob[idx] = node.children[idx].value + \
                C_PUCT * p * coef / (1 + node.children[idx].num_visit)
        else:
            prob[idx] = node.children[idx].avg_reward + \
                C_PUCT * p * coef / (1 + node.children[idx].num_visit)
    return max(prob.items(), key=lambda x: x[1])[0]


def simulation(node, game, evaluator):
    node.update_num_visit()
    if node.num_visit == 1 or game.gameover or game.gametied:
        # evaluator
        backup(node)
        if game.gameover or game.gametied:
            return
        expand_node(node, game, evaluator)
    else:
        idx = sample_move(node)
        game.play(idx=idx)
        simulation(node.children[idx], game, evaluator)


def mcts(game,
         evaluator,
         num_simulations=32,
         add_dir_noise=True,
         dir_noise_weight=DIRICHLET_WEIGHT,
         dir_alpha=DIRICHLET_ALPHA,
         verbose=False):

    def _gen_pi(node):
        pi_t = np.zeros_like(policy)
        for child_id in node.children:
            pi_t[child_id] = 1. * node.children[child_id].num_visit
        return 1./(node.num_visit-1) * pi_t

    v, p = evaluator.predict(
        np.reshape(
            game.game_state * game.current_player.color,
            newshape=(1, game.board.board_size, game.board.board_size, 1)
        )
    )
    policy = p[0]
    if add_dir_noise:
        policy = (1-dir_noise_weight)*policy + \
                 dir_noise_weight*dirichlet.rvs([dir_alpha]*(p.shape[1]))[0]
    root = Node(
        value=v[0, 0],
        policy=policy
    )
    for i in range(num_simulations):
        # debug('start %d-th simulation' % (i+1))
        game_status = game.get_game_status()
        simulation(root, game, evaluator)
        game.reset(game_status)
    return _gen_pi(root)

