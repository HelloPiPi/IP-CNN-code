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
from functools import partial
from keras.regularizers import l2

def gram_matrix(x):
    # assert KB.ndim(x) == 3
    if KB.ndim(x)==2:
        # features = L.Lambda(lambda x:KB.reshape(x,(-1,x.shape[3],x.shape[1]*x.shape[2])))(x)
        gram = L.Lambda(lambda x:KB.batch_dot(KB.expand_dims(x,axis=-1),KB.expand_dims(x,axis=1)))(x)
    else:
        if KB.image_data_format() == 'channels_first':
            features = KB.batch_flatten(x)
        else:
            features = L.Lambda(lambda x:KB.reshape(x,(-1,x.shape[1]*x.shape[2],x.shape[3])))(x)
            print(features.shape)
        gram = L.Lambda(lambda x:KB.batch_dot(x,KB.permute_dimensions(x,(0,2,1))))(features)
    print(gram.shape)
    return gram

def gram_spec_matrix(x):
    if KB.image_data_format() == 'channels_first':
        features = KB.batch_flatten(x)
    else:
        features = L.Lambda(lambda x:KB.reshape(x,(-1,x.shape[1]*x.shape[2],x.shape[3])))(x)
        print(features.shape)
    gram = L.Lambda(lambda x:KB.batch_dot(KB.permute_dimensions(x,(0,2,1)),x))(features)
    print(gram.shape)
    return gram

def gram(x,y): 
    features1 = L.Lambda(lambda x:KB.reshape(x,(-1,x.shape[1]*x.shape[2],x.shape[3])))(y)
    print(features1.shape)
    gram_x = gram_matrix(x)
    gram = L.Lambda(lambda x:KB.permute_dimensions(x,(0,2,1)))(features1)
    gram = L.Lambda(lambda x:KB.batch_dot(x[0],x[1]))([gram,gram_x])
    gram = L.Lambda(lambda x:KB.permute_dimensions(x,(0,2,1)))(gram)
    gram = L.Lambda(lambda x:KB.reshape(x[1],(-1,x[0].shape[1],x[0].shape[1],x[1].shape[2])))([y,gram])
    return gram


def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    if KB.ndim(style)==4:
        channels = style.shape[3].value
        size = style.shape[1].value * style.shape[2].value
    else:
        channels = 1
        size = style.shape[1].value
    return L.Lambda(lambda x:KB.sum(KB.sum(KB.square(x[0] - x[1]),1,keepdims=True),2,keepdims=False) / (4. * (channels**2) * (size**2)))([S,C])

def spect_loss(style, combination):
    S = gram_spec_matrix(style)
    C = gram_spec_matrix(combination)
    channels = style.shape[3].value
    size = style.shape[1].value * style.shape[2].value
    return L.Lambda(lambda x:KB.sum(KB.sum(KB.square(x[0] - x[1]),1,keepdims=True),2,keepdims=False) / (4. * (channels**2) * (size**2)))([S,C])

def compute_pairwise_distances(x, y):
    if x.get_shape().as_list()[1] != y.get_shape().as_list()[1]:
        raise ValueError('The number of features should be the same.')
    return L.Lambda(lambda x:KB.sum(KB.sum(KB.square(x[0] - x[1]),1,keepdims=True),2,keepdims=False))([x,y])
    # norm = lambda x: tf.reduce_sum(tf.square(x), 1)

def gaussian_kernel_matrix(x, y, sigmas):
    # beta = L.Lambda(lambda x:1. / (2. * (tf.expand_dims(x, 1))))(sigmas)
    dist = compute_pairwise_distances(x, y)
    # s = L.Lambda(lambda x:tf.matmul(x[0], tf.reshape(x[1], (1, -1))))([sigmas,dist])
    # print(KB.is_keras_tensor(beta))
    # return L.Lambda(lambda x:tf.reshape(tf.reduce_sum(tf.exp(tf.negative(x[0])), 0), tf.shape(x[1])))([s,dist])
    return L.Lambda(lambda x:KB.exp(tf.negative(x[0])*x[1]))([sigmas,dist])

# sigmas = [
#     1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
#     1e3, 1e4, 1e5, 1e6
# ]
# gaussian_kernel = partial(gaussian_kernel_matrix, sigmas=tf.constant(sigmas))

def MMD_cost(source, target):
    xx=gaussian_kernel_matrix(source,source,1)
    xy=gaussian_kernel_matrix(source,target,1)
    yy=gaussian_kernel_matrix(target,target,1)
    MMD = L.Lambda(lambda x:KB.mean(x[0]) - 2 * KB.mean(x[1]) + KB.mean(x[2]))([xx,xy,yy])
    #return the square root of the MMD because it optimizes better
    return L.Lambda(lambda x:KB.sqrt(x))(MMD)
# content loss
# 内容图片与风格图片“内容”部分的差异
def content_loss(base, combination):
    return KB.sum(KB.square(combination - base),keepdims=True)

# total variation loss：
# 第三个loss函数，用来表示生成图片的局部相干性
def total_variation_loss(x):
    assert KB.ndim(x) == 4
    if KB.image_data_format() == 'channels_first':
        a = KB.square(x[:, :, :img_nrows-1, :img_ncols-1] - x[:, :, 1:, :img_ncols-1])
        b = KB.square(x[:, :, :img_nrows-1, :img_ncols-1] - x[:, :, :img_nrows-1, 1:])
    else:
        a = KB.square(x[:, :img_nrows-1, :img_ncols-1, :] - x[:, 1:, :img_ncols-1, :])
        b = KB.square(x[:, :img_nrows-1, :img_ncols-1, :] - x[:, :img_nrows-1, 1:, :])
    return K.sum(K.pow(a+b, 1.25))

 
def simple_cnn_branch(input_tensor, small_mode=True):
    filters = 128 if small_mode else 384
    # conv0 = L.Conv2D(128, (1, 1), padding='same')(input_tensor)
    conv0 = L.Conv2D(256, (3, 3), padding='same')(input_tensor)
    conv0 = L.BatchNormalization(axis=-1)(conv0)
    conv0 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv0)
    conv2 = L.Conv2D(512, (1,1), padding='same')(conv0)
    conv2 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv2)
   # conv3=L.Conv2D(256, (3,3), padding='same',activation='relu')(conv3)
    conv2=L.MaxPool2D(pool_size=(2, 2),padding='same')(conv2)
    conv2 = L.Flatten()(conv2)
    # conv2 = L.Dense(1536)(conv2)
    return conv2
    
def cascade_block(input, nb_filter, kernel_size=3):
    conv1_1 = L.Conv2D(nb_filter * 2, (kernel_size, kernel_size), padding='same')(input)  # nb_filters*2
    conv1_2 = L.Conv2D(nb_filter, (1, 1),padding='same')(conv1_1)  # nb_filters
    relu1 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv1_2)

    conv2_1 = L.Conv2D(nb_filter * 2, (kernel_size, kernel_size),padding='same')(relu1)  # nb_filters*2
    conv2_1 = L.add([conv1_1, conv2_1])
    conv2_1 = L.BatchNormalization(axis=-1)(conv2_1)

    conv2_2 = L.Conv2D(nb_filter, (1, 1), padding='same')(conv2_1)  # nb_filters
    relu2 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv2_2)
    relu2 = L.add([relu1, relu2])
    
    conv3_1 = L.Conv2D(nb_filter , (1, 1),padding='same')(relu2)  # nb_filters*2
    relu3 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv3_1)
    return relu3

def cascade_Net(input_tensor):
    filters = [16, 32, 64, 96, 128,192, 256, 512]
    conv0 = L.Conv2D(filters[2], (3, 3), padding='same')(input_tensor)
    # conv0 = L.BatchNormalization(axis=-1)(conv0)
    conv0 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv0)
    conv0 = cascade_block(conv0, filters[2])
    conv0 = L.MaxPool2D(pool_size=(2, 2), padding='same')(conv0)

    # conv1 = L.Conv2D(filters[4], (1, 1), padding='same')(conv0)
    # conv1 = L.BatchNormalization(axis=-1)(conv1)
    conv1 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv0)
    # conv1 = L.GaussianNoise(stddev=0.05)(conv1)
    conv1 = cascade_block(conv1, nb_filter=filters[4])
    # conv2 = L.Conv2D(filters[4], (1,1), padding='same')(conv1)
    conv_flt = L.Flatten()(conv1)
    # conv_flt=L.Dense(512,activation='relu')(conv_flt)
    return conv_flt

def pixel_branch(input_tensor):
    filters = [8, 16, 32, 64, 96, 128]
    # input_tensor=L.Permute((2,1))(input_tensor)
    conv0 = L.Conv1D(filters[3], 11, padding='valid')(input_tensor) 
    conv0 = L.BatchNormalization(axis=-1)(conv0)
    conv0 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv0)
    # conv0 = L.MaxPool1D(padding='valid')(conv0)

    # conv1 = L.Conv1D(filters[2], 7, padding='valid')(conv0)  
    # # conv1 = L.BatchNormalization(axis=-1)(conv1)
    # conv1 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv1)
    # conv2 = L.Conv1D(filters[3], 5, padding='valid')(conv1)  
    # # conv2 = L.BatchNormalization(axis=-1)(conv2)
    # conv2 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv2)
    # # conv2 = L.MaxPool1D(padding='valid')(conv2) 
    conv3 = L.Conv1D(filters[5], 3, padding='valid')(conv0)  
    # conv3 = L.BatchNormalization(axis=-1)(conv3)
    conv3 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv3)
    conv3 = L.MaxPool1D(pool_size=2, padding='valid')(conv3)
    conv3 = L.Flatten()(conv3)
    # conv3 = L.Dense(192)(conv3)
    return conv3

def hsi_branch(hchn,ksize=11):
    ksize = 2 * r + 1
    filters = [64, 128, 256, 512]
    hsi_in = L.Input((ksize, ksize, hchn))
    hsi_pxin = L.Input((hchn, 1))

    h_simple = simple_cnn_branch(hsi_in, small_mode=False)
    px_out = pixel_branch(hsi_pxin)
    # px_out=pixel_branch_2d(hsi_pxin)
    merge=L.concatenate([h_simple,px_out])
    merge = L.Dropout(0.5)(merge)
    logits = L.Dense(NUM_CLASS, activation='softmax')(merge)

    model = K.models.Model([hsi_in,hsi_pxin], logits)
    adam = K.optimizers.Adam(lr=0.0001)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy', metrics=['acc'])
    return model


def _bn_relu(x, bn_name=None, relu_name=None):
    """Helper to build a BN -> relu block
    """
    norm = L.BatchNormalization(axis=-1, name=bn_name)(x)
    return L.Activation("relu", name=relu_name)(norm)


def conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu residual unit activation function.
        This is the original ResNet v1 scheme in https://arxiv.org/abs/1512.03385
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    dilation_rate = conv_params.setdefault("dilation_rate", (1, 1))
    conv_name = conv_params.setdefault("conv_name", None)
    bn_name = conv_params.setdefault("bn_name", None)
    relu_name = conv_params.setdefault("relu_name", None)
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(x):
        x = L.Conv2D(filters=filters, kernel_size=kernel_size,
                   strides=strides, padding=padding,
                   dilation_rate=dilation_rate,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer,
                   name=conv_name)(x)
        return _bn_relu(x, bn_name=bn_name, relu_name=relu_name)

    return f

    
def _bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv residual unit with full pre-activation function.
    This is the ResNet v2 scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    dilation_rate = conv_params.setdefault("dilation_rate", (1, 1))
    conv_name = conv_params.setdefault("conv_name", None)
    bn_name = conv_params.setdefault("bn_name", None)
    relu_name = conv_params.setdefault("relu_name", None)
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(x):
        activation = _bn_relu(x, bn_name=bn_name, relu_name=relu_name)
        return L.Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      dilation_rate=dilation_rate,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer,
                      name=conv_name)(activation)

    return f


def _shortcut(input_feature, residual, conv_name_base=None, bn_name_base=None):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(input_feature)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input_feature
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        print('reshaping via a convolution...')
        if conv_name_base is not None:
            conv_name_base = conv_name_base + '1'
        shortcut = L.Conv2D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001),
                          name=conv_name_base)(input_feature)
        if bn_name_base is not None:
            bn_name_base = bn_name_base + '1'
        shortcut = L.BatchNormalization(axis=CHANNEL_AXIS, name=bn_name_base)(shortcut)

    return L.add([shortcut, residual])
 