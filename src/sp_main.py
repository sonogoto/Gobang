#!/usr/bin/env python3

from .model import gen_train_data
from .game import Board
from .game import ActingPlayer
from .game import Game
from .model import TFEvaluator
import time
from .model import TF_GCNNet
from .model.train_model import batch_play_against


BOARD_SIZE = 15
BATCH_SIZE = 1024


def eval_model(ckpt1=None, ckpt2=None, rounds=10):
    params['ckpt'] = ckpt1
    evaluator1 = TFEvaluator(TF_GCNNet, **params)
    params['ckpt'] = ckpt2
    evaluator2 = TFEvaluator(TF_GCNNet, **params)
    print(batch_play_against(game, evaluator1, evaluator2, rounds))


def train_model(ckpt_train=None, ckpt_eval=None, epochs=50):
    params['ckpt'] = ckpt_train
    model_train = TFEvaluator(TF_GCNNet, **params)
    params['ckpt'] = ckpt_eval
    evaluator = TFEvaluator(TF_GCNNet, **params)
    for epoch in range(epochs):
        print(time.ctime(), '%d-th epoch' % (epoch+1))
        x, y = gen_train_data(game, evaluator, BATCH_SIZE, verbose=True)
        print(model_train.train_(x, y['value'], y['policy']))
    model_train.save("/home/duser/model/gcn/%d.ckpt" % int(time.time()))


if __name__ == '__main__':

    player1 = ActingPlayer(1, ActingPlayer.BLACK)
    player2 = ActingPlayer(2, ActingPlayer.WHITE)
    params = {'board_size': BOARD_SIZE,
              'feature_dim': 1,
              'gcn_layers': [16, 32, 32, 32, 16],
              'device': '/device:GPU:0',
              'num_res_blocks': 5}
    board = Board(board_size=BOARD_SIZE)
    game = Game(board, player1, player2)
    eval_model(
        ckpt1='/home/user/beta5s/model/gcn/1568881758.ckpt',
        ckpt2='/home/user/beta5s/model/gcn/1568891787.ckpt',
        rounds=32
    )
    # train_model(
    #     # ckpt_train='/home/duser/model/gcn/1568817952.ckpt',
    #     # ckpt_eval='/home/duser/model/gcn/1568817952.ckpt'
    # )

