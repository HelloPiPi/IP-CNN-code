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
from model import *
from train import*
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import plot_model
from resnet import *
# In[3]:

parser = argparse.ArgumentParser()
parser.add_argument('--NUM_EPOCH',
                    type=int,
                    default=500,
                    help='number of epoch')
parser.add_argument('--mode',
                    type=int,
                    default=0,
                    help='train or test mode')
parser.add_argument('--BATCH_SIZE',
                    type=int,
                    default=64,
                    help='train or test mode')
parser.add_argument('--NET',
                    type=int,
                    default=0,
                    help='achitecture of net')
parser.add_argument('--ksize',
                    type=int,
                    default=11,
                    help='window size')
parser.add_argument('--path',
                    type=int,
                    default=1,
                    help='path')
parser.add_argument('--strategy',
                    type=str,
                    default='Adam',
                    help='learning strategy')
args = parser.parse_args()

#----------ABOUT DATASET---------------------------#

variables.dataset='salinas'




# variables.ALL_H = 'pavia9.mat'
# variables.ALL_L = 'pavia9_Lidar.mat'
# variables.NUM_CHN = 103
# variables.NUM_DIM = 3
# variables.HEIGHT = 610
# variables.WIDTH = 340
# variables.NUM_CLASS = 9

variables.ALL_H = 'salinas16.mat'
variables.ALL_L = 'salinas16_Lidar.mat'
variables.NUM_CHN = 224
variables.NUM_DIM = 3
variables.HEIGHT = 512
variables.WIDTH = 217
variables.NUM_CLASS = 16
#--------------------------------------------------#
#----------ABOUT TRAINING--------------------------#
variables.TRAIN_H = 'train_data_H'
variables.TEST_H = 'test_data_H.txt'
variables.PATH = os.path.join('../file/' + variables.dataset+'/')
variables.PATH_ALL = os.path.join('../data/' + variables.dataset+'/')
variables.weights_path = os.path.join('weights/' + variables.dataset+'/') 

variables.NUM_EPOCH = args.NUM_EPOCH
variables.BATCH_SIZE = args.BATCH_SIZE
variables.ksize = args.ksize
variables.r = args.ksize // 2
variables.stride = 2
variables.model_name_mul_P2P_H = os.path.join(variables.weights_path + args.strategy+'mul_HL.h5')
variables.model_name_mul_HL = os.path.join(variables.weights_path + 'mul_HL_cls.h5')


variables.new_model_name = os.path.join(
    variables.weights_path + 'ALL' + str(args.BATCH_SIZE) + '.h5')
#--------------------------------------------------#
if not os.path.exists('log/'):
    os.makedirs('log/')
# In[4]:


def main(mode=1, show=False):
    # f = open(os.path.join(variables.PATH, variables.TEST_H), 'r')  
    # a = f.read()  
    # print type(a)
    # mdata = eval(a)  
    # f.close()
    # print test_H.shape
    # import pprint, pickle

    # pkl_file = open(os.path.join(variables.PATH, variables.TEST_H), 'rb')

    # data1 = pickle.load(pkl_file)
    # aa1=pprint.pprint(data1['data_H'])
    
    # print aa1.shape
 
    # pkl_file.close()
    
 

if __name__ == '__main__':
    main()
