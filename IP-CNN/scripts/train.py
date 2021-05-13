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
from data_util import *
from call_util import *
from model__ import *
from config import variables
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard
# In[3]:
def train_P2P(model, model_name,mode):
    model_ckt = ModelCheckpoint(
        filepath=model_name, verbose=1, save_best_only=False)
    tensorbd = TensorBoard(log_dir='./log', histogram_freq=0,
                           write_graph=True, write_images=True)
    train_data = read_data(variables.PATH_ALL, variables.ALL_H, 'data', 'mat')
    label_data = read_data(variables.PATH_ALL, variables.ALL_L, 'data', 'mat')
    GT_trainDA = read_data(variables.PATH_ALL,variables.train_label_name ,'mask_train', 'mat')
    GT_testDA = read_data(variables.PATH_ALL,variables.test_label_name ,'mask_test', 'mat')
    train_data = np.asarray(train_data, dtype=np.float32)
    label_data = np.asarray(label_data, dtype=np.float32)
    GT_trainDA = np.asarray(GT_trainDA, dtype=np.float32)
    GT_testDA = np.asarray(GT_testDA, dtype=np.float32)

    train_data = sample_wise_standardization(train_data)
    label_data = sample_wise_standardization(label_data)

    train_data = image_pad(train_data, variables.r + variables.r - 1)
    label_data = image_pad(label_data, variables.r + variables.r - 1)
    GT_trainDA = image_pad(GT_trainDA, variables.r + variables.r - 1)
    GT_testDA = image_pad(GT_testDA, variables.r + variables.r - 1)
    if len(label_data.shape)==2:
        label_data=np.expand_dims(label_data,-1)
    else:
        label_data=label_data
    print(GT_trainDA.shape,label_data.shape)
    print(GT_trainDA.shape,label_data.shape)
    train_gen = image_data_generator(mode,
        train_data, label_data, variables.r, variables.stride, batch_size=variables.BATCH_SIZE, seed=0)
    model.fit_generator(train_gen, steps_per_epoch=variables.HEIGHT*variables.WIDTH/(variables.BATCH_SIZE*variables.stride*variables.stride), epochs=variables.NUM_EPOCH, verbose=1,
                        callbacks=[model_ckt, tensorbd])
    model.save(model_name)

def train_cls_P2P(model, net, model_name,mode):
    model_ckt = ModelCheckpoint(
        filepath=model_name, verbose=1, save_best_only=True)
    tensorbd = TensorBoard(log_dir='./log', histogram_freq=0,
                           write_graph=True, write_images=True)
    train_H = read_data(variables.PATH, 'train_data_H', 'train_data_H', 'npy')
    train_L = read_data(variables.PATH, 'train_data_L', 'train_data_L', 'npy')
    label_train = read_data(variables.PATH, 'train_label_H', 'train_label_H', 'npy')
    validation_H = read_data(variables.PATH, 'validation_data_H', 'validation_data_H', 'npy')
    validation_L = read_data(variables.PATH, 'validation_data_L', 'validation_data_L', 'npy')
    label_validation = read_data(variables.PATH, 'validation_label_H', 'validation_label_H', 'npy')

    print('train label data shape:{}'.format(label_train.shape))
    label_train=np.expand_dims(label_train,0)

    train_H = np.asarray(train_H, dtype=np.float32)
    train_L = np.asarray(train_L, dtype=np.float32)
    label_train = np.asarray(label_train, dtype=np.float32)
    validation_H = np.asarray(validation_H, dtype=np.float32)
    validation_L = np.asarray(validation_L, dtype=np.float32)
    label_validation = np.asarray(label_validation, dtype=np.float32)

    test_H = np.load(os.path.join(variables.PATH + 'test_data_H.npy'))
    test_L = np.load(os.path.join(variables.PATH + 'test_data_L.npy'))
    if len(test_L.shape)==3:
        test_L = np.expand_dims(test_L, 3)
    
    label_test =  np.load(os.path.join(variables.PATH + 'test_label_H.npy'))
    label_test=np.expand_dims(label_test,0)
    label_test = np.reshape(label_test.T, (label_test.shape[1]))

    train_labels = np.squeeze(K.utils.np_utils.to_categorical(
        label_train, variables.NUM_CLASS))
    validation_labels = np.squeeze(K.utils.np_utils.to_categorical(
        label_validation, variables.NUM_CLASS))
    test_labels = np.squeeze(K.utils.np_utils.to_categorical(
        label_test, variables.NUM_CLASS))
    print('train hsi data shape:{}'.format(train_H.shape))
    print('{} train sample'.format(train_H.shape[0]))
    if net==1:
        model.fit([train_H,train_L], [train_labels],
                batch_size=variables.BATCH_SIZE,
                epochs=variables.NUM_EPOCH,
                verbose=1,
                validation_data=([validation_H,validation_L], [validation_labels]),
                shuffle=True,
                callbacks=[model_ckt, tensorbd])
    if net==2:
        model.fit([train_H], [train_labels],
                    batch_size=variables.BATCH_SIZE,
                    epochs=variables.NUM_EPOCH,
                    verbose=1,
                    validation_data=([validation_H], [validation_labels]),
                    shuffle=True,
                    callbacks=[model_ckt, tensorbd])
    if net==3:
        model.fit([train_H,train_L,train_H], [train_labels],
                    batch_size=variables.BATCH_SIZE,
                    epochs=variables.NUM_EPOCH,
                    verbose=1,
                    validation_data=([validation_H,validation_L,validation_H], [validation_labels]),
                    shuffle=True,
                    callbacks=[model_ckt, tensorbd])
    model.save(os.path.join(model_name+'final') )


def test_P2P(model, model_name, H_data, L_data):
    #-----cannot do testing-----#
    model.load_weights(model_name)
    pred = model.predict([H_data], batch_size=variables.BATCH_SIZE)
    return pred


def test_cls_P2P(model, model_name, input_data,input_data2):
    model.load_weights(model_name)
    pred = model.predict([input_data,input_data2], batch_size=variables.BATCH_SIZE)
    return pred

def test_cls_P2P_(model, model_name, input_data,input_data1):
    model.load_weights(model_name)
    pred = model.predict([input_data], batch_size=variables.BATCH_SIZE)
    return pred

def test_cls_P2P__(model, model_name, input_data,input_data1):
    model.load_weights(model_name)
    pred = model.predict([input_data,input_data1,input_data], batch_size=variables.BATCH_SIZE)
    return pred