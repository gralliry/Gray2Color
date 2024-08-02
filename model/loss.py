#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description:
from tensorlayerx.nn import Module
from tensorlayerx.losses import mean_squared_error


class Loss(Module):
    def __init__(self, g):
        super(self.__class__, self).__init__()
        self.g = g
        self.loss = mean_squared_error

    def forward(self, gr, cr):
        sr = self.g(gr)
        return self.loss(sr, cr, "mean")
