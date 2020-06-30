#!/usr/bin/env python3


from .model import TFEvaluator
from .model import TF_GCNNet
import os
import pickle as pk
import time
import sys


params = {
    'board_size': 15,
    'feature_dim': 1,
    'gcn_layers': [16, 32, 32, 32, 16],
    'device': '/device:GPU:0',
    'value_loss_weight': .5,
    'policy_loss_weight': .5,
}


def train_from_file(model, path, save_model=False):
    for fname in os.listdir(path):
        if not fname.startswith('features'):
            continue
        idx = fname.split('.')[0].split('-')[-1]
        if not os.path.exists(path+'/policies-'+idx+'.pkl'):
            continue
        if not os.path.exists(path+'/values-'+idx+'.pkl'):
            continue
        features, policies, values = (
            pk.load(open(path+'/'+fname, 'rb')),
            pk.load(open(path+'policies-'+idx+'.pkl', 'rb')),
            pk.load(open(path+'values-'+idx+'.pkl', 'rb')),
        )
        v_loss, p_loss, loss, _ = model.train_(features, values, policies)
        print('%s-th batch, value loss: %s, policy: %s, total: %s'
              % (idx, v_loss, p_loss, loss))
    if save_model:
        model.save("/home/user/beta5s/model/gcn/%d.ckpt" % int(time.time()))


if __name__ == '__main__':
    epochs = int(sys.argv[2]) if len(sys.argv) >= 3 else 1
    model = TFEvaluator(TF_GCNNet, **params)
    for n in range(epochs-1):
        print(time.ctime(), '****** %d-th epoch ******' % (n+1))
        train_from_file(model, sys.argv[1])
    print(time.ctime(), '****** %d-th epoch ******' % epochs)
    train_from_file(model, sys.argv[1], True)
