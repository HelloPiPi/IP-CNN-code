# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import os
import time
import cv2
from data_util import *
import numpy as np
import scipy.io as sio
import keras as K
import keras.layers as L
import tensorflow as tf
import tifffile as tiff

 
parser = argparse.ArgumentParser()
parser.add_argument('--save_path',
                    type=str,
                    default='../file/MUUFL/',
                    help='save path')
parser.add_argument('--PATH',
                    type=str,
                    default='../data/MUUFL/',
                    help='load path')
parser.add_argument('--data',
                    type=str,
                    default='MUUFL',
                    help='load path')
parser.add_argument('--ksize',
                    type=int,
                    default=13,
                    help='window size')
args = parser.parse_args()

r = args.ksize // 2

if args.data=='houston':
    NUM_CLASS = 15
    mdata_name = 'houston15.mat'
    ldata_name = 'houston_Lidar15'
    train_label_name = 'houston15_mask_train'
    test_label_name = 'houston15_mask_test1'
    data_type = 'mat'
if args.data=='italy':
    NUM_CLASS = 6
    mdata_name = 'italy6.mat'
    ldata_name = 'italy6_Lidar.mat'
    train_label_name = 'italy6_mask_train.mat'
    test_label_name='italy6_mask_test.mat'
    data_type = 'mat'
if args.data=='MUUFL':
    NUM_CLASS = 11
    mdata_name = 'hsi_data.mat'
    ldata_name = 'lidar_data2.mat'
    train_label_name = 'mask_train_150.mat'
    test_label_name='mask_test_1501.mat'
    data_type = 'mat'
# In[6]:


def main():
    mdata = read_data(args.PATH, mdata_name, 'data', data_type)
    ldata = read_data(args.PATH, ldata_name, 'data', data_type)
    label_train = read_data(args.PATH, train_label_name,
                            'mask_train', data_type)
    mdata = np.asarray(mdata, dtype=np.float32)
    print(mdata.shape)
    ldata = np.asarray(ldata, dtype=np.float32)
    mdata = image_pad(mdata, r + r + 1)
    ldata = image_pad(ldata, r + r + 1)
    label_train = image_pad(label_train, r + r + 1)
    mdata = sample_wise_standardization(mdata)
    ldata = sample_wise_standardization(ldata)
    train_data_H, train_data_L, train_label_H = creat_train(mdata,ldata,label_train,r,validation=False)

    print('HSI train data shape:{}'.format(train_data_H.shape))
    print('Lidar train data shape:{}'.format(train_data_L.shape)) 
    print('Lidar train label shape:{}'.format(train_label_H.shape)) 
    # SAVE TRAIN_DATA TO MAT FILE
    path_train = os.path.join(args.save_path + 'train_data_H.npy')
    np.save(path_train, train_data_H)
    path_train = os.path.join(args.save_path + 'train_data_L.npy')
    np.save(path_train, train_data_L)
    path_train = os.path.join(args.save_path + 'train_label_H.npy')
    np.save(path_train, train_label_H)

    validation_data_H, validation_data_L, validation_label_H = creat_train(mdata,ldata,label_train,r,validation=False)
    print('HSI vlidation data shape:{}'.format(validation_data_H.shape))
    print('Lidar vlidation data shape:{}'.format(validation_data_L.shape)) 
    print('Lidar vlidation label shape:{}'.format(validation_label_H.shape)) 
    # SAVE vlidation_DATA TO MAT FILE
    path_validation = os.path.join(args.save_path + 'validation_data_H.npy')
    np.save(path_validation, validation_data_H)
    path_validation = os.path.join(args.save_path + 'validation_data_L.npy')
    np.save(path_validation, validation_data_L)
    path_validation = os.path.join(args.save_path + 'validation_label_H.npy')
    np.save(path_validation, validation_label_H)

    # SAVE TEST_DATA TO MAT FILE
    label_test = read_data(args.PATH, test_label_name, 'mask_test', data_type)
    label_test = image_pad(label_test, r + r + 1)
    # test mode!!!!!!!!!
    test_data_H, test_data_L, test_label_H, idx, idy = construct_spatial_patch(
        mdata, ldata, label_test, r, 'test')
    print('HSI test data shape:{}'.format(test_data_H.shape))
    print('Lidar test data shape:{}'.format(test_data_L.shape))
    print('Saving test data...')
    path_test=os.path.join(args.save_path,'test_data_H.npy')
    np.save(path_test, test_data_H)
    path_test=os.path.join(args.save_path,'test_data_L.npy')
    np.save(path_test, test_data_L)
    path_test=os.path.join(args.save_path,'test_label_H.npy')
    np.save(path_test, test_label_H)
    np.save(os.path.join(args.save_path,'index.npy'), [idx,idy])
 

    print('Done')

if __name__ == '__main__':
    main()