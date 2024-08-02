#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description:
import argparse
import os

from PIL import Image

import numpy as np
import tensorflow as tf
import tensorlayerx as tlx

from model.model import ResNet

tlx.logging.set_verbosity(tlx.logging.ERROR)
os.environ['TL_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

tlx.files.exists_or_mkdir("./samples", verbose=False)

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", type=str, required=True, help="image of path")
args = parser.parse_args()


def evaluate():
    #
    physical_gpus = tf.config.list_physical_devices("GPU")
    print("[-] All GPUs:", physical_gpus)
    print("[-] Using:", physical_gpus[0].name)
    tlx.set_device('GPU', id=0)
    #
    image = tlx.vision.load_image(path=args.path)
    gray_image = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    out_gray_image = gray_image.astype(np.uint8)
    Image.fromarray(out_gray_image, mode='L').save("./samples/gray_img.png")
    model = ResNet()
    model.set_eval()
    model.load_weights("./checkpoint/g.npz", format="npz_dict")
    color_img = model(gray_image[np.newaxis, ..., np.newaxis] / 127.5 - 1)
    color_img = tlx.ops.convert_to_numpy(color_img[0])
    out_color_img = ((color_img + 1) * 127.5).astype(np.uint8)
    Image.fromarray(out_color_img, mode='RGB').save("./samples/color_img.png")


if __name__ == "__main__":
    evaluate()
