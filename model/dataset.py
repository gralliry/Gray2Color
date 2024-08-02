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
            Normalize(mean=(127.5, 127.5, 127.5), std=(127.5, 127.5, 127.5))
        ])

    def __getitem__(self, index):
        color_img = self.color_imgs[index]
        color_img = self.color_tran(color_img)

        gray_img = 0.299 * color_img[..., 0] + 0.587 * color_img[..., 1] + 0.114 * color_img[..., 2]

        return gray_img[..., np.newaxis], color_img

    def __len__(self):
        return len(self.color_imgs)
