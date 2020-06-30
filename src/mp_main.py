#!/usr/bin/env python3


import multiprocessing as mp
from .multi_process import *
from multiprocessing import Queue, Process
from multiprocessing import Manager as MGR
from multiprocessing.managers import BaseManager
from .game import Game, ActingPlayer, Board
import torch
import time
from .common.log import *
from .model import TF_GCNNet, TorchGCNNet
from .model.evaluator import TFEvaluator
from .model.train_model import play_against


BOARD_SIZE = 15
INPUT_SIZE = 1
HIDDEN_SIZE = [32, 32, 32, 32, 32]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT_PATH = "/home/user/beta5s/model/gcn/%d.ckpt" % int(time.time())
EVALUATOR_CKPT_PATH = ''
BATCHES = 50
BATCH_SIZE = 128
RESOURCE_QUEUE_LEN = RESULT_QUEUE_NO = 4
MCTS_PROCESS_NO = 4
EVALUATOR_PROCESS_NO = 1
NUM_OF_GPU = 4


params = {'board_size': BOARD_SIZE,
          'feature_dim': 1,
          'gcn_layers': [16, 32, 32, 32, 16],
          'device': '/device:GPU:0',
          'value_loss_weight': .5,
          'policy_loss_weight': .5,
          'num_res_blocks': 5}


def new_game():
    return Game(
        board=Board(BOARD_SIZE),
        player1=ActingPlayer(1, ActingPlayer.BLACK),
        player2=ActingPlayer(1, ActingPlayer.WHITE)
    )


def new_gcn(ckpt=None, device_id=0):
    params['ckpt'] = ckpt
    params['device'] = '/device:GPU:%d' % device_id
    return TFEvaluator(TF_GCNNet, **params)


class Manager(BaseManager):
    pass
Manager.register('Model', new_gcn)
Manager.register('Evaluator', new_gcn)


def mp_train(mgr,
             model_ckpt=None,
             eval_ckpt=None,
             mcts_process_no=8,
             eval_process_no=4):
    model = mgr.Model(model_ckpt)
    task_queue = Queue()
    training_queue = Queue()
    train_p = TrainProcess(
        model=model,
        batches=BATCHES,
        batch_size=BATCH_SIZE,
        task_queue=task_queue,
        training_queue=training_queue,
        ckpt_path=CKPT_PATH,
        dump_data=False
    )
    request_queue = Queue()
    result_queues = [Queue() for _ in range(mcts_process_no)]
    evaluator_processes = []
    for i in range(eval_process_no):
        evaluator = mgr.Evaluator(eval_ckpt, i % NUM_OF_GPU)
        evaluator_processes.append(
            EvaluatorProcess(
                evaluator=evaluator,
                request_queue=request_queue,
                result_queues=result_queues,
            )
        )
    for i, p in enumerate(evaluator_processes):
        info('create %d-th evaluator process' % i)
        p.start()

    resource_queue = Queue(mcts_process_no)
    for i in range(mcts_process_no):
        resource_queue.put(i)

    mcts_processes = []
    for i in range(mcts_process_no):
        mcts_processes.append(
            MCTSProcess(
                # game=mgr.Game(),
                game=new_game(),
                task_queue=task_queue,
                resource_queue=resource_queue,
                request_queue=request_queue,
                result_queues=result_queues,
                training_queue=training_queue,
                id_=i,
                sample_range=1
            )
        )
    for i, p in enumerate(mcts_processes):
        info('create the %d-th mcts process' % i)
        p.start()
    info('create training process')
    train_p.start()
    train_p.join()
    for p in evaluator_processes:
        p.terminate()
    for p in mcts_processes:
        p.terminate()


def _target(ckpt1, ckpt2, task_queue, rst_stat, half_idx=1):
    game = new_game()
    game_status = game.get_game_status()
    evaluator1 = new_gcn(ckpt1)
    evaluator2 = new_gcn(ckpt2)
    while not task_queue.empty():
        round_no = task_queue.get_nowait()
        winner = play_against(game, evaluator1, evaluator2)
        info('play against, %d-th half, round %d, winner is %d' % (half_idx, round_no, winner))
        rst_stat[half_idx*3+winner] = rst_stat.get(half_idx*3+winner, 0) + 1
        game.reset(game_status)


def mp_eval(mgr,
            ckpt1=None,
            ckpt2=None,
            rounds=100,
            process_no=4
            ):
    task_queue = Queue()
    rst_stat = mgr.dict()
    info('first half begin ......')
    for i in range(rounds):
        task_queue.put(i)
    p_list = []
    for i in range(process_no):
        _args = (
            ckpt1,
            ckpt2,
            task_queue,
            rst_stat
        )
        p_list.append(Process(target=_target, args=_args))
    for i, p in enumerate(p_list):
        info('start %d-th process ......' % i)
        p.start()
    for p in p_list:
        p.join()
    info('second half begin ......')
    for i in range(rounds):
        task_queue.put(i)
    p_list = []
    for i in range(process_no):
        _args = (
            ckpt2,
            ckpt1,
            task_queue,
            rst_stat,
            2
        )
        p_list.append(Process(target=_target, args=_args))
    for i, p in enumerate(p_list):
        info('start %d-th process ......' % i)
        p.start()
    for p in p_list:
        p.join()
    print('player #1 won %d games' % (rst_stat.get(4, 0)+rst_stat.get(8, 0)))
    print('player #2 won %d games' % (rst_stat.get(5, 0)+rst_stat.get(7, 0)))
    print('tied: %d, total: %d' % (rst_stat.get(3, 0)+rst_stat.get(6, 0),
                                   sum(rst_stat.values())))


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    if sys.argv[1] == 'train':
        manager = Manager()
        manager.start()
        mp_train(
            manager,
            *sys.argv[2:],
            mcts_process_no=8,
            eval_process_no=4
        )
    elif sys.argv[1] == 'eval':
        with MGR() as manager:
            mp_eval(
                manager,
                *sys.argv[2:],
                rounds=16)
    else:
        info('unrecognized arg [%s], do nothing' % sys.argv[1])

