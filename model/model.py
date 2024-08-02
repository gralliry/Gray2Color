#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description:
from tensorlayerx.nn import (Module, Sequential, Conv2d, BatchNorm2d, ReLU, Tanh)


class ResdualBlock(Module):
    def __init__(self, in_channels, out_channels):
        super(ResdualBlock, self).__init__()
        self.conv1 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                            b_init=None)
        self.bn1 = BatchNorm2d(num_features=out_channels)
        self.act = ReLU()
        self.conv2 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                            b_init=None)
        self.bn2 = BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + identity
        return x


class ResNet(Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.head = Sequential([
            Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            ReLU(),
        ])
        self.body = Sequential([
            ResdualBlock(in_channels=32, out_channels=32) for _ in range(16)
        ])

        self.tail = Sequential([
            Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            ReLU(),
            Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1),
            Tanh()
        ])

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x
