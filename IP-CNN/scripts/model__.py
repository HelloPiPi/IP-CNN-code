# coding: utf-8

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
import cv2
import tifffile as tiff
import tensorflow as tf
import numpy as np
import scipy.io as sio
import keras as K
import keras.layers as L
from sklearn.preprocessing import MultiLabelBinarizer
from config import variables
from keras.utils import plot_model
from keras import backend as KB
from keras import regularizers
from keras_util import *
from non_local import *

def gram_matrix(x):
    if KB.image_data_format() == 'channels_first':
        features = KB.batch_flatten(x)
    else:
        features = L.Lambda(lambda x:KB.reshape(x,(-1,x.shape[3],x.shape[1]*x.shape[2])))(x)
        print(features.shape)
    gram = L.Lambda(lambda x:KB.batch_dot(KB.permute_dimensions(x,(0,2,1)),x))(features)
    print(gram.shape)
    return gram

def spatial_loss(y_true1, y_pred1):
    S = gram_matrix(y_true1)
    C = gram_matrix(y_pred1)
    channels = y_true1.shape[3].value
    size = y_true1.shape[1].value * y_true1.shape[2].value
    return L.Lambda(lambda x:KB.sum(KB.sum(KB.square(x[0] - x[1]),1,keepdims=True),2,keepdims=False) / (4. * (channels**2) * (size**2)))([S,C])
 
def gram_matrix1(x):
    if KB.image_data_format() == 'channels_first':
        features = KB.batch_flatten(x)
    else:
        features = L.Lambda(lambda x:KB.reshape(x,(-1,x.shape[3],x.shape[1]*x.shape[2])))(x)
        print(features.shape)
    gram = L.Lambda(lambda x:KB.batch_dot(x, KB.permute_dimensions(x,(0,2,1))))(features)
    print(gram.shape)
    return gram

def spectral_loss(y_true1, y_pred1):
    S = gram_matrix1(y_true1)
    C = gram_matrix1(y_pred1)
    channels = y_true1.shape[3].value
    size = y_true1.shape[1].value * y_true1.shape[2].value
    return L.Lambda(lambda x:KB.sum(KB.sum(KB.square(x[0] - x[1]),1,keepdims=True),2,keepdims=False) / (4. * (channels**2) * (size**2)))([S,C])

def mean_squared_error(y_true, y_pred):
    if not KB.is_tensor(y_pred):
        y_pred = KB.constant(y_pred)
    y_true = KB.cast(y_true, y_pred.dtype)
    return KB.mean(KB.square(y_pred - y_true), axis=-1)
class mul_P2P_net(object):
    def __init__(self, H_shape,L_shape, c,strategy):
        filters = [32, 64, 128, 256]
        dilations = [1, 3, 3, 7]
        self.c = c
        self.input_spat1 = L.Input(H_shape)
        self.input_spat2 = L.Input(L_shape)

        self.conv0 = L.Conv2D(variables.NUM_CHN, (3,3), padding='same')(self.input_spat2)
        self.conv1 = L.BatchNormalization(axis=-1)(self.conv0)
        self.conv2 = L.Activation('relu')(self.conv1)
        self.conv3 = L.Conv2D(variables.NUM_CHN, (1, 1), padding='same')(self.conv2)
        self.conv4 = L.Activation('relu')(self.conv3)

        self.conv0_0 = L.Conv2D(variables.NUM_CHN, (3, 3), padding='same')(self.input_spat1)
        self.conv0_1 = L.BatchNormalization(axis=-1)(self.conv0_0)
        self.conv0_2 = L.advanced_activations.LeakyReLU(alpha=0.2)(self.conv0_1)
        self.conv0_3 = L.Conv2D(variables.NUM_CHN, (1,1), padding='same')(self.conv0_2)
        self.conv0_4 = L.advanced_activations.LeakyReLU(alpha=0.2)(self.conv0_3)
 
        self.merge0=L.concatenate([self.conv4,self.conv0_4],axis=-1)
        self.merge1 = L.BatchNormalization(axis=-1)(self.merge0)
        self.merge2 = L.Activation('relu')(self.merge1)
        self.merge3 = L.Conv2D(variables.NUM_CHN, (1,1), padding='same')(self.merge2)

        self.conv5 = L.BatchNormalization(axis=-1)(self.merge3)
        self.conv6 = L.Activation('relu')(self.conv5)
        self.HSI = L.Conv2D(variables.NUM_CHN, (dilations[2], dilations[2]), padding='same')(self.conv6)

        self.conv0_5 = L.BatchNormalization(axis=-1)(self.merge3)
        self.conv0_6 = L.Activation('relu')(self.conv0_5)
        self.LiDAR = L.Conv2D(variables.NUM_DIM, (dilations[2], dilations[2]), padding='same')(self.conv0_6)

        a=spatial_loss(self.merge3,self.input_spat2)
        b=spectral_loss(self.merge3,self.input_spat1)
        a=L.Lambda(lambda x:tf.add(x[0],x[1]))([a,b])

        self.model = K.models.Model([self.input_spat1,self.input_spat2], [a,self.HSI,self.LiDAR])
 
        if strategy=='Adam':
            opti = K.optimizers.Adam(lr=0.001)
        if strategy=='SGD_001_95_1e-5':
            opti=K.optimizers.SGD(lr=0.001, momentum=0.95,decay=1e-5)
        if strategy=='SGD_001_99_1e-3':
            opti=K.optimizers.SGD(lr=0.001, momentum=0.99,decay=1e-3)
        self.model.compile(optimizer=opti, loss=[lambda y_true,y_pred: y_pred,'mse','mse'], loss_weights=[variables.regular1,variables.regular1,variables.regular2], metrics=['acc'])

class cls_mul_P2P_net(object):
    def __init__(self, input_shape,input_shape2,strategy='adam', P2P_weight=None):
        """
        input:
            input_shape: input shape of HSI or LIDAR
        """
        NUM_DIM=variables.NUM_DIM
        HL9=mul_P2P_net(input_shape,input_shape2,NUM_DIM ,strategy)
        HL9.model.load_weights(P2P_weight)
        # HL9.model.trainable = False
        # for layer in HL9.model.layers:
        #     layer.trainable = False
        # HL9.model.compile(optimizer=K.optimizers.Adam(lr=0.001), loss=[lambda y_true,y_pred: y_pred,'mse','mse'], loss_weights=[variables.regular1,variables.regular1,variables.regular2], metrics=['acc'])    
        p2p_out9 = HL9.merge3
        p2p_in9=HL9.model.input
        h_simple = simple_cnn_branch(p2p_out9, small_mode=False)
        # hsi_pxin=L.Lambda(lambda x:KB.permute_dimensions(x[:,5,5,:],(1,0)))(merge0)
        hsi_pxin=L.Lambda(lambda x:KB.expand_dims(x[:,variables.ksize//2,variables.ksize//2,:],axis=2))(p2p_out9)
        px_out = pixel_branch(hsi_pxin)
        # px_out=pixel_branch_2d(hsi_pxin)
        merge=L.concatenate([h_simple,px_out])
        merge = L.Dropout(0.1)(merge)
        logits = L.Dense(variables.NUM_CLASS, 
        activation='softmax')(merge)
        # activation='linear',kernel_regularizer=regularizers.l2(0.5))(merge)
        self.model = K.models.Model([p2p_in9[0],p2p_in9[1]], [logits])
        # opti = K.optimizers.Adam(lr=0.0001)
        opti=K.optimizers.SGD(lr=0.001,momentum=0.99,decay=1e-4)
        self.model.compile(optimizer=opti,
        loss='categorical_crossentropy', metrics=['acc'])
        # loss='categorical_squared_hinge', metrics=['acc'])
        
class cls_mul_P2P_net1(object):
    def __init__(self, input_shape,input_shape2, strategy,net1):
        """
        input:
            input_shape: input shape of HSI or LIDAR
        """
        filters = [64, 128, 256, 512]
        lidar_in = L.Input(input_shape)
        L_cas=cascade_Net(lidar_in)
        merge = L.Dropout(0.5)(L_cas)
        logits = L.Dense(variables.NUM_CLASS, activation='softmax')(merge)
        self.model = K.models.Model([lidar_in], logits)
        adam = K.optimizers.Adam(lr=0.0001)
        self.model.compile(optimizer=adam,
                    loss='categorical_crossentropy', metrics=['acc'])
 
class cls_mul_merge(object):
    def __init__(self, input_shape,input_shape2, strategy,net1,net2):
        """
        input:
            input_shape: input shape of HSI or LIDAR
        """
        model_h=cls_mul_P2P_net(input_shape,input_shape2, 'Adam',variables.model_name_mul_P2P_H).model
        model_l=cls_mul_P2P_net1(input_shape,input_shape2, 'Adam',variables.NETWORK1).model
        if not net1 is None: 
            model_h.load_weights(net1)
        if not net2 is None:
            model_l.load_weights(net2)
        for i in range(2):
            model_h.layers.pop()
            model_l.layers.pop()
        # model_h.trainable = False
        # model_l.trainable = False
        # for layer in model_h.layers:
        #     layer.trainable = False
        # for layer in model_l.layers:
        #     layer.trainable = False
        # model_h.compile(optimizer=K.optimizers.Adam(lr=0.0001),
        #             loss='categorical_crossentropy', metrics=['acc'])
        # model_l.compile(optimizer=K.optimizers.Adam(lr=0.0001),
        #             loss='categorical_crossentropy', metrics=['acc'])
        hsi_in, hsi_px = model_h.input
        hsi_out = model_h.layers[-1].output
        hsi_in, hsi_pxin = model_h.input
        lidar_out = model_l.layers[-1].output
        lidar_in = model_l.input

        merge = L.concatenate([hsi_out, lidar_out], axis=-1)
        merge = L.BatchNormalization(axis=-1)(merge)
        merge=L.Dropout(0.25)(merge)
        merge = L.Dense(128)(merge)
        merge = L.advanced_activations.LeakyReLU(alpha=0.2)(merge)
        logits = L.Dense(variables.NUM_CLASS, activation='softmax')(merge)
        self.model = K.models.Model([hsi_in, hsi_pxin, lidar_in], logits)
        if not net1 is None or net2 is None:
            optm = K.optimizers.SGD(lr=0.005,momentum=1e-6)
        else:
            optm=K.optimizers.Adam(lr=0.001)
        self.model.compile(optimizer=optm,
                    loss='categorical_crossentropy', metrics=['acc'])
      