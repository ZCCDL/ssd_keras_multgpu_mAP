import os
import sys
import glob

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

plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams['image.interpolation'] = 'nearest'

np.set_printoptions(suppress=True)

classes_name = ['crack', 'carbonation', 'free_lime_c', 'free_lime_d', 'free_lime_e']
mAP_text_path = './mAP/predicted/'
img_dir = './datasets/dataset/'
save_dir = './predict/'
NUM_CLASSES = len(classes_name) + 1
GPU_COUNT = 4
input_shape = (300, 300, 3)
EPOCHS = 200
priors = pickle.load(open('prior_boxes_ssd300.pkl', 'rb'))
bbox_util = BBoxUtility(NUM_CLASSES, priors)


gt = pickle.load(open('datasets.pkl', 'rb'))
keys = sorted(gt.keys())
num_train = int(round(0.8 * len(keys)))
train_keys = keys[:num_train]
val_keys = keys[num_train:]
num_val = len(val_keys)

model = SSD300(input_shape, num_classes=NUM_CLASSES)

model.load_weights('./param.hdf5')
base_lr = 3e-4
optim = keras.optimizers.Adam(lr=base_lr)
model.compile(optimizer=optim,
              loss=MultiboxLoss(NUM_CLASSES, neg_pos_ratio=2.0).compute_loss)

#image_list = glob.glob('./datasets/dataset/image*')


# For mAP
def output_mAP_text(text_mAP, name, mAP_text_path, is_mAP=False):
    if is_mAP:
        with open(mAP_text_path+name+'.txt', 'wt') as f:
            for line in text_mAP:
                f.write(' '.join(line)+'\n')


xml_list = os.listdir('./datasets/annotation/')
image_dir = './datasets/dataset/'
image_list = []
name_list = []
for file_name in xml_list:
    name, _ = os.path.splitext(file_name)
    image_list.append(image_dir+name+'.jpg')
    name_list.append(name)


for idx, name in enumerate(image_list):
    text_mAP = []
    name = os.path.basename(name)
    print(name)
    inputs = []
    images = []
    img_path = img_dir + name
    img = image.load_img(img_path, target_size=(300, 300))
    img = image.img_to_array(img)
    images.append(imread(img_path))
    inputs.append(img.copy())
    inputs = preprocess_input(np.array(inputs))

    preds = model.predict(inputs)
    results = bbox_util.detection_out(preds)
    # 予測が空リストを返した場合に対処
    if results==[[]]:
        continue
    for i, img in enumerate(images):
        # Parse the outputs.
        det_label = results[i][:, 0]
        det_conf = results[i][:, 1]
        det_xmin = results[i][:, 2]
        det_ymin = results[i][:, 3]
        det_xmax = results[i][:, 4]
        det_ymax = results[i][:, 5]

        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        colors = plt.cm.hsv(np.linspace(0, 1, NUM_CLASSES)).tolist()

        plt.imshow(img / 255.)
        currentAxis = plt.gca()

        for i in range(top_conf.shape[0]):
            xmin = int(round(top_xmin[i] * img.shape[1]))
            ymin = int(round(top_ymin[i] * img.shape[0]))
            xmax = int(round(top_xmax[i] * img.shape[1]))
            ymax = int(round(top_ymax[i] * img.shape[0]))
            score = top_conf[i]
            label = int(top_label_indices[i])
            text_mAP.append([classes_name[label-1], str(score), str(xmin), str(ymin), str(xmax), str(ymax)])
            display_txt = '{:0.2f}, {}'.format(score, classes_name[label-1])
            coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
            color = colors[label]
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})
        currentAxis.tick_params(labelbottom="off",bottom="off") # x軸の削除
        currentAxis.tick_params(labelleft="off",left="off") # y軸の削除
        plt.box("off") #枠線の削除
        plt.savefig(save_dir+name)
        plt.clf()
    output_mAP_text(text_mAP, name_list[idx], mAP_text_path, is_mAP=True)
