#!/usr/bin/env python3


from .evaluator import TFEvaluator
from .gcn_tf import GCNNet
import sys
import pickle as pk
import numpy as np


params = {'board_size': 15,
          'feature_dim': 1,
          'gcn_layers': [16, 32, 32, 32, 16],
          'device': '/device:GPU:0',
          'value_loss_weight': .5,
          'policy_loss_weight': .5
          }

if __name__ == '__main__':
    data_file = sys.argv[1]
    ckpt = sys.argv[2] if len(sys.argv) >= 3 else None

    if ckpt:
        params['ckpt'] = ckpt

    evaluator = TFEvaluator(model_func=GCNNet, **params)

    x = pk.load(open(data_file, 'rb'))
    if x.shape[-1] != 1:
        x = np.expand_dims(x, -1)

    v, p = evaluator.predict(x)

    print(
        'value:, %f, ' % v[0],
        '\npolicy:\n',
        np.reshape(p[0], (-1, params['board_size']))
    )
