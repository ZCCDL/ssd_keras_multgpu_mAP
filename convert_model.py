"""
学習に使ったモデルはGPUを複数利用している。
モデルの重みはGPU複数利用するモデルとして保存されている。
予測を行う際にGPU１枚だけを利用するために、重みを変換する。
マルチモデルで読み込んでシングルモデルで保存
"""

import os
import sys

import cv2
import keras
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.preprocessing import image
from keras.utils import multi_gpu_model
import matplotlib.pyplot as plt
import numpy as np
import pickle
from random import shuffle
from scipy.misc import imread
from scipy.misc import imresize
import tensorflow as tf

from ssd import SSD300
from ssd_training import MultiboxLoss
from ssd_utils import BBoxUtility


GPU_COUNT = 4
NUM_CLASSES = 6 #4
INPUT_SHAPE = (300, 300, 3)
EPOCHS = 200
BATCH_SIZE = 6 * GPU_COUNT
priors = pickle.load(open('prior_boxes_ssd300.pkl', 'rb'))

with tf.device('/cpu:0'):
    model = SSD300(INPUT_SHAPE, num_classes=NUM_CLASSES)

def schedule(epoch, decay=0.9):
    return base_lr * decay**(epoch)

if not os.path.isdir('./checkpoints'):
    os.mkdir('./checkpoints')
callbacks = [keras.callbacks.ModelCheckpoint('./checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                             verbose=1,
                                             save_weights_only=True),
             keras.callbacks.LearningRateScheduler(schedule)]

parallel_model = multi_gpu_model(model, GPU_COUNT)
parallel_model.load_weights('./weights.200-11.18.hdf5')
base_lr = 3e-4
optim = keras.optimizers.Adam(lr=base_lr)
parallel_model.compile(optimizer=optim,
              loss=MultiboxLoss(NUM_CLASSES, neg_pos_ratio=2.0).compute_loss)

model.save_weights('my_model_weights.h5')
