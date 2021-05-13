# -*- coding: utf-8 -*-

# In[1]:

import keras as K
import keras.layers as L
import tensorflow as tf
import scipy.io as sio
import argparse
import os
import numpy as np
import h5py
import time
import sys
import matplotlib.pyplot as plt
from config import variables
from data_util import *
from model__ import *
from train import*
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import plot_model
# from resnet import *
from keras import backend as KB


from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
# In[3]:
os.environ['CUDA_VISIBLE_DEVICES'] = '"/device:GPU:0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
KB.set_session(session)
parser = argparse.ArgumentParser()
parser.add_argument('--NUM_EPOCH',
                    type=int,
                    default=500,
                    help='number of epoch')
parser.add_argument('--regular1',
                    type=float,
                    default=0.01,
                    help='parameter')
parser.add_argument('--regular2',
                    type=float,
                    default=1.,
                    help='parameter')
parser.add_argument('--mode',
                    type=int,
                    default=1,
                    help='train or test mode')
parser.add_argument('--BATCH_SIZE',
                    type=int,
                    default=128,
                    help='train or test mode')
parser.add_argument('--NET',
                    type=int,
                    default=3,
                    help='achitecture of net')
parser.add_argument('--ksize',
                    type=int,
                    default=13,
                    help='window size')
parser.add_argument('--strategy',
                    type=str,
                    default='Adam',
                    help='learning strategy')
parser.add_argument('--data',
                    type=str,
                    default='houston',
                    help='data tuning')
parser.add_argument('--avgTIME',
                    type=str,
                    default='9',
                    help='data tuning')
args = parser.parse_args()

#----------ABOUT DATASET---------------------------#
if args.data=='italy':
    variables.dataset='italy'
    variables.ALL_H = 'italy6.mat'
    variables.ALL_L = 'italy6_Lidar.mat'
    variables.NUM_CHN = 63
    variables.NUM_DIM = 2
    variables.HEIGHT = 166
    variables.WIDTH = 200
    variables.NUM_CLASS = 6
    variables.train_label_name = 'italy6_mask_train'
    variables.test_label_name = 'italy6_mask_test'
if args.data=='MUUFL':
    variables.dataset='MUUFL'
    variables.ALL_H = 'hsi_data.mat'
    variables.ALL_L = 'lidar_data2.mat'
    variables.NUM_CHN = 64
    variables.NUM_DIM = 2
    variables.HEIGHT = 325
    variables.WIDTH = 220
    variables.NUM_CLASS = 11
    variables.train_label_name = 'mask_train_150'
    variables.test_label_name = 'mask_test_1501'
if args.data=='houston':
    variables.dataset='houston'
    variables.ALL_H = 'houston15.mat'
    variables.ALL_L = 'houston_Lidar15.mat'
    variables.NUM_CHN = 144
    variables.NUM_DIM = 1
    variables.HEIGHT = 349
    variables.WIDTH = 1905
    variables.NUM_CLASS = 15
    variables.train_label_name = 'houston15_mask_train'
    variables.test_label_name = 'houston15_mask_test1'
#----------ABOUT TRAINING--------------------------#
variables.TRAIN_H = 'train_data_H.npy'
variables.TEST_H = 'test_data_H.npy'
variables.TEST_L = 'test_data_L.npy'
variables.LBL = 'test_label_H.npy'
variables.PATH = os.path.join('../file/' + args.data+'/')
variables.PATH_ALL = os.path.join('../data/' + variables.dataset+'/')
variables.weights_path = os.path.join('weights/' + args.data+'/') 
variables.regular1=args.regular1
variables.regular2=args.regular2
variables.NUM_EPOCH = args.NUM_EPOCH
variables.BATCH_SIZE = args.BATCH_SIZE
variables.ksize = args.ksize
variables.r = args.ksize // 2
variables.stride = 2
variables.avgTIME=args.avgTIME
variables.model_name_mul_P2P_H = os.path.join(variables.weights_path + args.strategy+str(args.ksize)+str(args.regular1)+str(args.regular2)+variables.ALL_L.split('.mat')[0]+'.h5')
variables.model_name_mul_HL = os.path.join(variables.weights_path +str(args.ksize)+str(args.regular1)+str(args.regular2)+'finetuneCLS_NET.h5')
variables.NETWORK1 = variables.model_name_mul_HL+'NET1'
variables.NETWORK2 =  os.path.join(variables.weights_path +str(args.ksize)+'CLS_NET2.h5')
variables.model_final = os.path.join(variables.weights_path +str(args.ksize)+str(args.regular1)+str(args.regular2)+'CLS_FINAL.h5')

variables.new_model_name = os.path.join(
    variables.weights_path + 'ALL' + str(args.BATCH_SIZE) + '.h5')
#--------------------------------------------------#
if not os.path.exists('log/'):
    os.makedirs('log/')
# In[4]:


def main(mode=1, show=False):
    H_shape = (args.ksize, args.ksize, variables.NUM_CHN)
    L_shape = (args.ksize, args.ksize, variables.NUM_DIM)
    print(H_shape)
    if args.mode == 0:
        if args.NET == 0:
            model = mul_P2P_net(H_shape,L_shape,variables.NUM_DIM,args.strategy).model
            start_time = time.time()
            train_P2P(model, variables.model_name_mul_P2P_H,'H_L')
            duration = time.time() - start_time
            print(duration)
        if args.NET == 1:
            model = cls_mul_P2P_net(H_shape,L_shape, args.strategy, P2P_weight=variables.model_name_mul_P2P_H).model
            start_time = time.time()
            train_cls_P2P(model, args.NET, variables.model_name_mul_HL+'NET1','H_L')
            duration = time.time() - start_time
            print (duration)
        if args.NET == 2:
            model = cls_mul_P2P_net1(H_shape,L_shape,args.strategy,variables.NETWORK1).model
            start_time = time.time()
            train_cls_P2P(model, args.NET, variables.NETWORK2,'H_L')
            duration = time.time() - start_time
            print (duration)
        if args.NET == 3:
            model = cls_mul_merge(H_shape,L_shape,args.strategy,variables.NETWORK1,variables.NETWORK2).model
            start_time = time.time()
            train_cls_P2P(model, args.NET, variables.model_final,'H_L')
            duration = time.time() - start_time
            print (duration)
    else:

        test_H = np.load(os.path.join(variables.PATH + variables.TEST_H))
        test_L = np.load(os.path.join(variables.PATH + variables.TEST_L))
        label_data =  np.load(os.path.join(variables.PATH + variables.LBL))
        label_data=np.expand_dims(label_data,0)
        label_data = np.reshape(label_data.T, (label_data.shape[1]))
        if args.NET == 3: 
            model = cls_mul_merge(H_shape,L_shape,args.strategy,variables.NETWORK1,variables.NETWORK2).model
            if len(test_L.shape)==3:
                prediction = test_cls_P2P__(model,variables.model_final, test_H, np.expand_dims(test_L,-1))
            else:
                prediction = test_cls_P2P__(model,variables.model_final, test_H, test_L)
        print('OA: {}%'.format(eval(prediction, label_data)))
        prediction = np.asarray(prediction)
        pred = np.argmax(prediction, axis=1)
        pred = np.asarray(pred, dtype=np.int8)
        print (confusion_matrix(label_data, pred))
        print (classification_report(label_data, pred,digits=4))
        print (cohen_kappa_score(label_data, pred))

if __name__ == '__main__':
    main()
