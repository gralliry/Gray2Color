#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description:
import argparse
import os
import time

import tensorflow as tf
import tensorlayerx as tlx
from tensorlayerx.dataflow import DataLoader
from tensorlayerx.model import TrainOneStep

from model.model import ResNet
from model.dataset import Dataset
from model.loss import Loss

tlx.logging.set_verbosity(tlx.logging.ERROR)
os.environ['TL_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

tlx.files.exists_or_mkdir("./checkpoint", verbose=False)

parser = argparse.ArgumentParser()
parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="learning rate")
parser.add_argument("-e", "--epoch", type=int, default=3000, help="nums of epoch")
parser.add_argument("-bs", "--batch_size", type=int, default=16, help="batch size")
parser.add_argument("-d", "--dir", type=str, required=True, help="dir of dataset")
args = parser.parse_args()

lr = args.learning_rate
epochs = args.epoch
batch_size = args.batch_size
imgs_path = args.dir


def train():
    physical_gpus = tf.config.list_physical_devices("GPU")
    print("[-] All GPUs:", physical_gpus)
    print("[-] Using:", physical_gpus[0].name)
    tlx.set_device('GPU', id=0)
    g = ResNet()
    if os.path.exists("./checkpoint/g.npz"):
        g.load_weights("./checkpoint/g.npz", format="npz_dict")
    g.set_train()
    dataset = Dataset(path=imgs_path)
    img_nums = len(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
    scheduler = tlx.optimizers.lr.CosineAnnealingDecay(learning_rate=lr, T_max=epochs, eta_min=1e-6, verbose=True)
    optimizer = tlx.optimizers.Adam(scheduler)
    trainner = TrainOneStep(Loss(g), optimizer=optimizer, train_weights=g.trainable_weights)

    steps = round(img_nums // batch_size)
    for epoch in range(1, epochs + 1):
        for step, (gr_patch, cr_patch) in enumerate(dataloader, start=1):
            step_time = time.time()
            loss = trainner(gr_patch, cr_patch)
            print("Epoch: [{}/{}] Step: [{}/{}] Time: {:.3f}s, MSE: {:.5f} ".format(
                epoch, epochs, step, steps, time.time() - step_time, float(loss)
            ))
        scheduler.step()
        g.save_weights("./checkpoint/g.npz", format="npz_dict")
        g.save_weights("./checkpoint/g_{}.npz".format(epoch), format="npz_dict")
    print("[*] Finished!")


if __name__ == "__main__":
    train()
