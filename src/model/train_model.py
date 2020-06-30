#!/usr/bin/env python3

from .mcts import mcts
from scipy.stats import multinomial
import time
import numpy as np
import random
import tensorflow as tf
from .rotation import ROTATION, NUM_OF_ROTATION
from ..common.log import *


TIED = 0
PLAYER1 = 1
PLAYER2 = 2
NONE = 0


def self_play(game, evaluator, steps=0, explore_steps=20, verbose=False):

    game_records = []

    def _append_records():
        # to sample more records near end.
        # if random.random() > prob*(steps+1):
        #     return
        color_t = game.current_player.color
        s_t = game.game_state * color_t
        game_records.append((s_t, color_t, pi_t))

    while not game.gameover and not game.gametied:
        # debug('start %d-th move' % (steps+1))
        pi_t = mcts(game, evaluator)
        _append_records()
        if steps <= explore_steps:
            move = np.argmax(multinomial.rvs(1, pi_t))
        else:
            move = np.argmax(pi_t)
        game.play(idx=move)
        steps += 1
    # print(steps)
    return game_records, game.get_winner().color if game.gameover else TIED


def gen_train_data(game, evaluator, batch_size=512, verbose=False):

    def _add_rotation():
        _shape = record[0].shape[0]
        policy = np.reshape(record[2], newshape=(_shape, _shape))
        for rotate in ROTATION:
            states.append(rotate(record[0]))

            policies.append(
                np.reshape(rotate(policy), newshape=record[2].shape)
            )
            values.append(-1 * winner * record[1])

    states, values, policies = [], [], []
    cnt = 0
    game_status = game.get_game_status()
    while cnt < batch_size:
        game_records, winner = self_play(game, evaluator)
        game.reset(game_status)
        if game_records.__len__() < 1:
            continue
        # record = random.sample(game_records, 1)[0]
        record = game_records[-1]
        _add_rotation()
        cnt += NUM_OF_ROTATION
        debug('%d-th sample in batch' % cnt)
    features = np.asarray(
        np.expand_dims(states, axis=-1),
        dtype=np.float32
    )
    values = np.asarray(np.expand_dims(values, axis=-1), dtype=np.float32)
    policies = np.asarray(np.array(policies), dtype=np.float32)
    return features, {'value': values, 'policy': policies}


def play_against(game, evaluator1, evaluator2, verbose=False):

    def _winner():
        if game.gameover:
            return PLAYER1 if current_player is evaluator2 else PLAYER2
        else:
            return NONE

    current_player = evaluator1
    steps = 1
    while not game.gameover and not game.gametied:
        debug('%d-th move in play against' % steps)
        pi_t = mcts(game, current_player, add_dir_noise=False)
        move = np.argmax(pi_t)
        game.play(idx=move)
        current_player = evaluator2 if current_player is evaluator1 else evaluator1
        steps += 1
    return _winner()


def batch_play_against(game, evaluator1, evaluator2, times=100, verbose=False):
    game_status = game.get_game_status()
    game_stat = {PLAYER1: 0, PLAYER2: 0, NONE: 0}
    for i in range(times):
        winner = play_against(game, evaluator1, evaluator2)
        debug('%d-th game in first half, winner is %d' % (i, winner))
        game_stat[winner] += 1
        game.reset(game_status)
    for i in range(times):
        winner = play_against(game, evaluator2, evaluator1)
        debug('%d-th game in second half, winner is %d' % (i, winner))
        if winner == PLAYER1:
            game_stat[PLAYER2] += 1
        elif winner == PLAYER2:
            game_stat[PLAYER1] += 1
        else:
            game_stat[NONE] += 1
        game.reset(game_status)
    return game_stat
