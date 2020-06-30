#!/usr/bin/env python3

import tensorflow as tf
from .game import ActingPlayer
from .game import Board, Game
from .model.utils import FakeModel
from .model.evaluator import TFEvaluator
from .model.evaluator import TFFunctionalEvaluator
from .model.cnn import create_network
from .model import gen_train_data
from .model import NUM_OF_ROTATION
from .model import TorchGCNNet, TF_GCNNet
from .model.gcn_tf import create_gcn
from .mp_main import mp_train as _mp
from multiprocessing.managers import BaseManager


BOARD_SIZE = 15
BATCH_SIZE = 128
BATCHES = 2
CKPT_PATH = '/tmp/benchmark.ckpt'
player1 = ActingPlayer(1, ActingPlayer.BLACK)
player2 = ActingPlayer(2, ActingPlayer.WHITE)
board = Board(board_size=BOARD_SIZE)
game = Game(board, player1, player2)
params = {
    'board_size': BOARD_SIZE,
    'feature_dim': 1,
    'gcn_layers': [16, 32, 32, 32, 16],
    'device': '/device:GPU:0',
    'num_res_blocks': 5
}


def new_game():
    return Game(
        board=Board(BOARD_SIZE),
        player1=ActingPlayer(1, ActingPlayer.BLACK),
        player2=ActingPlayer(1, ActingPlayer.WHITE)
    )


def new_eval_CNN():
    return TFFunctionalEvaluator(create_network, **params)


def new_eval_fGCN_tf():
    return TFFunctionalEvaluator(create_gcn, **params)


def new_eval_mGCN_tf():
    return TFEvaluator(TF_GCNNet, **params)


def new_eval_mGCN_torch():
    return TorchGCNNet(**params)


class Manager(BaseManager):
    pass
# Manager.register('Game', new_game)
Manager.register('Model', FakeModel)


def _sp(evaluator, model=None):
    if not model:
        model = FakeModel()
    for _ in range(BATCHES):
        x, y = gen_train_data(
            game,
            evaluator,
            BATCH_SIZE * NUM_OF_ROTATION,
            verbose=True
        )
        model.train_(x, y)


#########################################################
# 1. single process CNN
def _spCNN():
    _sp(TFFunctionalEvaluator(create_network, **params))


#########################################################
def _mpCNN():
    Manager.register('Evaluator', new_eval_CNN)
    manager = Manager()
    manager.start()
    return manager


# 2. multi process CNN with 1 MCTS process
def _mpCNN_1MCTS():
    mgr = _mpCNN()
    _mp(mgr)


# 3. multi process CNN with 4 MCTS process
def _mpCNN_4MCTS():
    mgr = _mpCNN()
    _mp(mgr, 4)


# 4. multi process CNN with 8 MCTS process
def _mpCNN_8MCTS():
    mgr = _mpCNN()
    _mp(mgr, 8)


#########################################################
# 5. single process tf functional GCN
def _sp_fGCN():
    _sp(TFFunctionalEvaluator(create_gcn, **params))


#########################################################
def _mp_fGCN_tf():
    Manager.register('Evaluator', new_eval_fGCN_tf)
    manager = Manager()
    manager.start()
    return manager


# 6. multi process tf functional GCN with 1 MCTS process
def _mp_fGCN_tf_1MCTS():
    mgr = _mp_fGCN_tf()
    _mp(mgr)


# 7. multi process tf functional GCN with 4 MCTS process
def _mp_fGCN_tf_4MCTS():
    mgr = _mp_fGCN_tf()
    _mp(mgr, 4)


# 8. multi process tf functional GCN with 8 MCTS process
def _mp_fGCN_tf_8MCTS():
    mgr = _mp_fGCN_tf()
    _mp(mgr, 8)


#########################################################
# 9. single process module torch GCN
def _sp_mGCN_torch():
    _sp(TorchGCNNet(**params))


#########################################################
def _mp_mGCN_torch():
    Manager.register('Evaluator', new_eval_mGCN_torch)
    manager = Manager()
    manager.start()
    return manager


# 10. multi process module torch GCN with 1 MCTS process
def _mp_mGCN_torch_1MCTS():
    mgr = _mp_mGCN_torch()
    _mp(mgr)


# 11. multi process module torch GCN with 4 MCTS process
def _mp_mGCN_torch_4MCTS():
    mgr = _mp_mGCN_torch()
    _mp(mgr, 4)


# 12. multi process module torch GCN with 8 MCTS process
def _mp_mGCN_torch_8MCTS():
    mgr = _mp_mGCN_torch()
    _mp(mgr, 8)


#########################################################
# 13. single process module tf GCN
def _sp_mGCN_tf():
    _sp(TFEvaluator(TF_GCNNet, **params))


#########################################################
def _mp_mGCN_tf():
    Manager.register('Evaluator', new_eval_mGCN_tf)
    manager = Manager()
    manager.start()
    return manager


# 14. multi process module tf GCN with 1 MCTS process
def _mp_mGCN_tf_1MCTS():
    mgr = _mp_mGCN_tf()
    _mp(mgr)


# 15. multi process module tf GCN with 4 MCTS process
def _mp_mGCN_tf_4MCTS():
    mgr = _mp_mGCN_tf()
    _mp(mgr, 4)


# 16. multi process module tf GCN with 8 MCTS process
def _mp_mGCN_tf_8MCTS():
    mgr = _mp_mGCN_tf()
    _mp(mgr, 8)


id2method_map = {
    '1': _spCNN,
    '2': _mpCNN_1MCTS,
    '3': _mpCNN_4MCTS,
    '4': _mpCNN_8MCTS,
    '5': _sp_fGCN,
    '6': _mp_fGCN_tf_1MCTS,
    '7': _mp_fGCN_tf_4MCTS,
    '8': _mp_fGCN_tf_8MCTS,
    '9': _sp_mGCN_torch,
    '10': _mp_mGCN_torch_1MCTS,
    '11': _mp_mGCN_torch_4MCTS,
    '12': _mp_mGCN_torch_8MCTS,
    '13': _sp_mGCN_tf,
    '14': _mp_mGCN_tf_1MCTS,
    '15': _mp_mGCN_tf_4MCTS,
    '16': _mp_mGCN_tf_8MCTS
}


if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    for key_ in id2method_map:
        print(key_, ': ', id2method_map[key_].__name__)
    key_ = input('input id: \n')
    if not key_ or key_ not in id2method_map:
        print('invalid id: %s' % key_)
        exit(-1)
    id2method_map[key_]()
