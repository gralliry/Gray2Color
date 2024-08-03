#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description:
import numpy as np
import tensorlayerx as tlx
from tensorlayerx.vision.transforms import Compose, RandomCrop, FlipVertical, FlipHorizontal, Normalize


class Dataset(tlx.dataflow.Dataset):
    def __init__(self, path):
        super(self.__class__, self).__init__()
        self.color_imgs = tlx.vision.load_images(path=path, n_threads=30)
        self.color_tran = Compose([
            RandomCrop(size=(128, 128)),
            FlipVertical(),
            FlipHorizontal(),
            Normalize(mean=(0.0, 0.0, 0.0), std=(255.0, 255.0, 255.0))
        ])
        self.weights = np.array([0.299, 0.587, 0.114], dtype=np.float32).reshape([1, 1, 3])

    def __getitem__(self, index):
        color_img = self.color_imgs[index]
        color_img = self.color_tran(color_img)

        gray_img = np.sum(color_img * self.weights, axis=-1, keepdims=True)

        return gray_img, color_img

    def __len__(self):
        return len(self.color_imgs)
