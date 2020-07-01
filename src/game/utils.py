#!/usr/bin/env python3

import torch


class Judge(torch.nn.Module):

    def __init__(self, n_continuous):
        super(Judge, self).__init__()
        self._n_continuous = n_continuous
        conv2d = torch.nn.Conv2d(
            in_channels=1,
            out_channels=2*n_continuous + 2,
            kernel_size=(n_continuous, n_continuous),
            bias=False
        )
        conv2d.weight.data = self._gen_kernels()
        self._conv2d = conv2d.eval()

    def _gen_kernels(self, fill=1):
        kernels = torch.zeros(
            size=(self._n_continuous * 2 + 2, 1, self._n_continuous, self._n_continuous),
            dtype=torch.float32
        )
        for i in range(self._n_continuous):
            kernels[i, 0, i, :] = fill
            kernels[self._n_continuous + i, 0, :, i] = fill
        for i in range(self._n_continuous):
            kernels[self._n_continuous * 2, 0, i, i] = fill
            kernels[self._n_continuous * 2 + 1, 0, i, self._n_continuous - 1 - i] = fill
        return kernels

    def forward(self, game_state, device="cpu"):
        x = torch.from_numpy(
            game_state
        ).reshape(
            1, 1, game_state.shape[0], game_state.shape[1]
        ).to(device)
        x = self._conv2d(x)
        return x.max().item(), x.min().item()
