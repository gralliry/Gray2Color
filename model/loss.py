#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description:
import numpy as np
import tensorlayerx as tlx
from tensorlayerx.nn import Module
from tensorlayerx.losses import mean_squared_error, absolute_difference_error


class Loss(Module):
    def __init__(self, g):
        super(self.__class__, self).__init__()
        self.g = g
        self.weights = np.array([0.299, 0.587, 0.114], dtype=np.float32).reshape([1, 1, 1, 3])

    def forward(self, gr, cr):
        sr = self.g(gr)
        loss1 = mean_squared_error(sr, cr, "mean")
        weights = tlx.convert_to_tensor(self.weights)
        gray = tlx.reduce_sum(sr * weights, axis=-1, keepdims=True)
        loss2 = 3.0 * absolute_difference_error(gr, gray, "mean")
        # print(loss1.numpy(), loss2.numpy())
        return loss1 + loss2
