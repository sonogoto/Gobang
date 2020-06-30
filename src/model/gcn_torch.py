#!/usr/bin/env python3

from torch.nn import ModuleList
from torch_geometric.nn.conv import GCNConv
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
import torch
from torch.nn import functional as F
from ..common.log import *
from .utils import build_edge_idx


class GCNNet(torch.nn.Module):

    def __init__(self,
                 board_size,
                 feature_dim,
                 gcn_layers,
                 device='cpu',
                 optimizer=torch.optim.Adam,
                 **kwargs):
        super(GCNNet, self).__init__()
        self._input_size = feature_dim
        self._board_size = board_size
        self._hidden_size = gcn_layers
        self._device = device if device == 'cpu' else 'cuda'
        self._edge_idx = torch.tensor(
            build_edge_idx(board_size),
            dtype=torch.int64,
            # device=self._device
        )
        self._gcn_layers = ModuleList(
            [GCNConv(in_channels=feature_dim, out_channels=gcn_layers[0])]
        )
        for layer_idx, size in enumerate(gcn_layers[:-1]):
            self._gcn_layers.append(
                GCNConv(in_channels=size, out_channels=gcn_layers[layer_idx + 1])
            )
        self._policy_fc = torch.nn.Linear(in_features=gcn_layers[-1], out_features=1)
        self._value_fc = torch.nn.Linear(in_features=gcn_layers[-1], out_features=1)
        self._weight_init()
        self._optimizer = optimizer(
            self.parameters(),
            lr=kwargs.get('lr', .03),
            weight_decay=kwargs.get('weight_decay', .001)
        )
        self.to(self._device)

    def _weight_init(self):
        torch.nn.init.xavier_normal_(self._policy_fc.weight)
        torch.nn.init.xavier_normal_(self._value_fc.weight)

    def forward(self, X, **kwargs):
        graph = self._cvt2batch(
            X,
            kwargs.get('batch_size', sys.maxsize)
        ).__iter__().__next__()
        features, edge_idx = graph.x.to(self._device), graph.edge_index.to(self._device)
        for gcn_layer in self._gcn_layers:
            features = torch.relu(gcn_layer(features, edge_idx))
        policies = self._policy_header(features)
        values = self._value_header(features, kwargs.get('pooling', 'avg'))
        return values, policies

    def _policy_header(self, features):
        return self._policy_fc(features).reshape(shape=(-1, self._board_size**2))

    def _value_header(self, features, pooling):
        features = features.reshape(
            shape=(-1, self._board_size**2, features.shape[-1])
        )
        if pooling == 'avg':
            features = torch.mean(features, dim=1)
        else:
            features = torch.max(features, dim=1)[0]
        return self._value_fc(features)

    def _cvt2batch(self, X, batch_size=1):
        data_list = []
        for attr in self._gen_node_attrs(X):
            data_list.append(
                Data(
                    x=attr,
                    edge_index=self._edge_idx,
                    num_nodes=self._board_size**2
                )
            )
        return DataLoader(data_list, batch_size=batch_size, shuffle=False)

    def _gen_node_attrs(self, X):
        return torch.tensor(
            X.reshape((-1, self._board_size ** 2, 1)),
            dtype=torch.float32
        )

    def predict(self, X, **kwargs):
        self.eval()
        v, p = self.forward(X, **kwargs)
        return v.cpu().detach().numpy(), p.cpu().detach().numpy()

    def train_(self, features, values, policies):
        self.train()
        self._optimizer.zero_grad()
        v = torch.tensor(
            values,
            dtype=torch.float,
            device=self._device
        )
        p = torch.tensor(
            policies,
            dtype=torch.float,
            device=self._device
        )
        # TODO 把不合法的位置mask掉，避免让模型自己去学，增加难度
        v_out, p_out = self.forward(features)
        loss = self.__class__._loss(
            v_out, v, p_out, p
        )
        info('loss: [%f]' % loss.item())
        loss.backward()
        self._optimizer.step()

    @staticmethod
    def _loss(v_out, v, p_out, p, v_weight=.5, p_weight=.5):
        v_loss = F.mse_loss(v_out, v, reduction='mean')
        # TODO kl_div的取值范围
        p_loss = F.kl_div(p_out, p, reduction='mean')
        return v_weight*v_loss + p_weight*p_loss

    def save(self, ckpt_path):
        torch.save(
            {'model_state': self.state_dict(),
             'input_size': self._input_size,
             'board_size': self._board_size,
             'hidden_size': self._hidden_size},
            ckpt_path
        )

    @classmethod
    def load(cls, ckpt_path, device='cpu'):
        ckpt = torch.load(ckpt_path)
        model = cls(
            board_size=ckpt['board_size'],
            input_size=ckpt['input_size'],
            hidden_size=ckpt['hidden_size'],
            device=device
        )
        model.load_state_dict(ckpt['model_state'])
        return model


if __name__ == '__main__':
    import numpy as np
    gcn = GCNNet(
        board_size=5,
        input_size=1,
        hidden_size=[4, 4],
    )
    x = np.random.randint(3, size=(10, 10)) - 1
    y = gcn.predict(x)
    print(y.__class__.__name__, y.shape)
    pass

