# coding: utf-8

import os
import cv2
import random
import tifffile as tiff
import numpy as np
import scipy.io as sio
import keras as K
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import MultiLabelBinarizer
from config import variables
import itertools

colors = ['#000000', '#901010', '#EC0505', '#FF00BF', '#BEABC0', '#00FF00','#00CECF','#00CCCD','#0F26FB','#4DBFED','#6FAF55','#7B4D77','#EDB021','#D9541A','#0073BD','#E6E6E6']
bounds = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
cmap = mpl.colors.ListedColormap(colors)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)


class DataSet(object):
    def __init__(self, hsi, labels):
        self._hsi = hsi
        self._labels = labels

    @property
    def hsi(self):
        return self._hsi

    @property
    def labels(self):
        return self._labels


def read_data(path, file_name, data_name, data_type):
    mdata = []
    if data_type == 'tif':
        mdata = tiff.imread(os.path.join(path, file_name))
        return mdata
    if data_type == 'mat':
        mdata = sio.loadmat(os.path.join(path, file_name))
        mdata = np.array(mdata[data_name])
        return mdata
    if data_type == 'npy':
        mdata=np.load(os.path.join(path + file_name+'.npy'))
        return mdata



def image_pad(data, r):
    if len(data.shape) == 3:
        data_new = np.lib.pad(data, ((r, r), (r, r), (0, 0)), 'symmetric')
        return data_new
    if len(data.shape) == 2:
        data_new = np.lib.pad(data, r, 'constant', constant_values=0)
        return data_new


def normalization(data, style):
    if style == 0:
        mi = np.min(data)
        ma = np.max(data)
        data_new = (data - mi) / (ma - mi)
    else:
        data_new = ldata / np.max(data)
    return data_new


def samele_wise_normalization(data):
    """
    normalize each sample to 0-1
    Input:
        sample
    Output:
        Normalized sample
    y=1.0*(x-np.min(x))/(np.max(x)-np.min(x))
    """
    if np.max(data) == np.min(data):
        return np.ones_like(data, dtype=np.float32) * 1e-6
    else:
        return 1.0 * (data - np.min(data)) / (np.max(data) - np.min(data))


def sample_wise_standardization(data):
    import math
    _mean = np.mean(data)
    _std = np.std(data)
    npixel = np.size(data) * 1.0
    min_stddev = 1.0 / math.sqrt(npixel)
    return (data - _mean) / max(_std, min_stddev)

def construct_spatial_patch(mdata, ldata, label, r, patch_type):
    # 根据label逐标签点构建HSI空间块(半径为r)和其标签
    # 使用该函数需要预先做好map_train,map_test,分别调用一次本函数
    if patch_type == 'test':
        patch_part_H = []
        patch_part_L = []
        result_part_H = []
        result_part_L = []
        labels = []
        result_labels = []
        idx, idy = np.nonzero(label)
        for i in range(len(idx)):
            tmpl=ldata[idx[i] - r:idx[i] + r + 1, idy[i] - r:idy[i] + r + 1, ...]
            patch_part_L.append(tmpl)
            patch_part_H.append(
                mdata[idx[i] - r:idx[i] + r + 1, idy[i] - r:idy[i] + r + 1, ...])
            labels.append(label[idx[i], idy[i]] - 1)
        result_part_H = np.asarray(patch_part_H, dtype=np.float32)
        result_part_L = np.asarray(patch_part_L, dtype=np.float32)
        result_labels = np.asarray(labels, dtype=np.int8)
        idx = idx - 2 * r - 1
        idy = idy - 2 * r - 1
        return result_part_H, result_part_L, result_labels, idx, idy

def cvt_map(pred, show=False):
    """
    convert prediction percent to map
    """
    # gth = tiff.imread(os.path.join('../data/univ/', 'mask_test_all.mat'))
    mdata = sio.loadmat(os.path.join(variables.PATH_ALL, variables.test_label_name+'.mat'))
    gth = np.array(mdata['mask_test'])
    pred = np.argmax(pred, axis=1)
    pred = np.asarray(pred, dtype=np.int8) + 1
    print (pred)
    index = np.load(os.path.join(variables.PATH, 'index.npy'))
    pred_map = np.zeros_like(gth)
    cls = []
    for i in range(index.shape[1]):
        pred_map[index[0, i], index[1, i]] = pred[i]
        cls.append(gth[index[0, i], index[1, i]])
    cls = np.asarray(cls, dtype=np.int8)
    if show:
        plt.axis('off')
        plt.imshow(pred_map,cmap=cmap)
        plt.figure()
        plt.axis('off')
        plt.imshow(gth,cmap=cmap)
        plt.show()
    sio.savemat('Houston_lidar.mat',{'a':pred_map})
    count = np.sum(pred == cls)
    mx = confusion(pred - 1, cls - 1)
    print (mx)
    acc = 100.0 * count / np.sum(gth != 0)
    kappa = compute_Kappa(mx)
    return acc, kappa

def confusion(pred, labels):
    """
    make confusion matrix 
    """
    mx = np.zeros((variables.NUM_CLASS, variables.NUM_CLASS))
    if len(pred.shape) == 2:
        pred = np.asarray(np.argmax(pred, axis=1))

    for i in range(labels.shape[0]):
        mx[pred[i], labels[i]] += 1
    mx = np.asarray(mx, dtype=np.int16)
    np.savetxt('confusion.txt', mx, delimiter=" ", fmt="%s")
    return mx


def compute_Kappa(confusion_matrix):
    """
    TODO =_= 
    """
    N = np.sum(confusion_matrix)
    N_observed = np.trace(confusion_matrix)
    Po = 1.0 * N_observed / N
    h_sum = np.sum(confusion_matrix, axis=0)
    v_sum = np.sum(confusion_matrix, axis=1)
    Pe = np.sum(np.multiply(1.0 * h_sum / N, 1.0 * v_sum / N))
    kappa = (Po - Pe) / (1.0 - Pe)
    return kappa

def split_image_to_patches(data, r, stride):
    if len(data.shape) == 2:
        h, w = data.shape
    if len(data.shape) == 3:
        h, w, c = data.shape
    patch = []
    result_patchs = []
    for i in xrange(r, h - r, stride):
        for j in xrange(r, w - r, stride):
            patch.append(data[i - r:i + r + 1, j - r:j + r + 1])
    result_patchs = np.asarray(patch, dtype=np.float32)
    return result_patchs


def random_flip(mdata, ldata, label, label_type, seed=0):
    if label_type == True:
        num = mdata.shape[0]
        mdatas = []
        ldatas = []
        labels = []
        for i in xrange(num):
            mdatas.append(mdata[i])
            ldatas.append(ldata[i])
            if len(mdata[i].shape) == 3:
                mdatas.append(np.flip(mdata, axis=0))
                noise = np.random.normal(0.0, 0.01, size=mdata.shape)
                mdatas.append(np.flip(mdata + noise, axis=1))
                k = np.random.randint(4)
                mdatas.append(np.rot90(mdata, k=k))

                ldatas.append(np.flip(ldata, axis=0))
                noise = np.random.normal(0.0, 0.03, size=ldata.shape)
                ldatas.append(np.flip(ldata + noise, axis=1))
                ldatas.append(np.rot90(ldata, k=k))
            labels.append(label[i])
            labels.append(label[i])
            labels.append(label[i])
            labels.append(label[i])
        mdatas = np.asarray(mdatas, dtype=np.float32)
        ldatas = np.asarray(ldatas, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.float32)
        np.random.seed(seed)
        index = np.random.permutation(mdatas.shape[0])
        return mdatas[index], ldatas[index], labels[index]
    if label_type == False:
        num = mdata.shape[0]
        mdatas = []
        ldatas = []
        for i in xrange(num):
            mdatas.append(mdata[i])
            ldatas.append(ldata[i])
            if len(mdata[i].shape) == 3:
                noise = np.random.normal(0.0, 0.01, size=(mdata[i].shape))
                mdatas.append(np.fliplr(mdata[i]) + noise)
                noise = np.random.normal(0.0, 0.01, size=(ldata[i].shape))
                ldatas.append(np.fliplr(ldata[i]) + noise)
        mdatas = np.asarray(mdatas, dtype=np.float32)
        ldatas = np.asarray(ldatas, dtype=np.float32)
        np.random.seed(seed)
        index = np.random.permutation(mdatas.shape[0])
        return mdatas[index], ldatas[index]

def creat_train(hsi,lidar,gth,r,validation=False):
    per=1
    Xh = []
    Xl = []
    Y = []
    num_class = np.max(gth)
    count=1
    for c in range(1, num_class + 1):
        idx, idy = np.where(gth == c)
        if not validation:
            idx = idx[:int(per * len(idx))]
            idy = idy[:int(per * len(idy))]
        else:
            idx = idx[int(per * len(idx)):]
            idy = idy[int(per * len(idy)):]
        np.random.seed(820)
        ID = np.random.permutation(len(idx))
        idx = idx[ID]
        idy = idy[ID]
        for i in range(len(idx)):
            count+=1
            tmph = hsi[idx[i] - r:idx[i] + r + 1, idy[i] - r:idy[i] + r + 1, :]
            tmpl = lidar[idx[i] - r:idx[i] + r + 1, idy[i] - r:idy[i] + r + 1]
            Xh.append(tmph)
            Xh.append(np.flip(tmph, axis=0))
            noise = np.random.normal(0.0, 0.01, size=tmph.shape)
            Xh.append(np.flip(tmph + noise, axis=1))
            k = np.random.randint(4)
            Xh.append(np.rot90(tmph, k=k))

            Xl.append(tmpl)
            Xl.append(np.flip(tmpl, axis=0))
            noise = np.random.normal(0.0, 0.01, size=tmpl.shape)
            Xl.append(np.flip(tmpl + noise, axis=1))
            Xl.append(np.rot90(tmpl, k=k)) 
           
            tmpy = gth[idx[i], idy[i]] - 1
            Y.append(tmpy)
            Y.append(tmpy)
            Y.append(tmpy)
            Y.append(tmpy) 
    index = np.random.permutation(len(Xh))
    Xh = np.asarray(Xh, dtype=np.float32)
    Xl = np.asarray(Xl, dtype=np.float32)
    print(Xl.shape)
    Y = np.asarray(Y, dtype=np.int8)
    Xh = Xh[index, ...]
    if len(Xl.shape)==3:
        Xl = Xl[index, ..., np.newaxis]
    elif len(Xl.shape)==4:
        Xl = Xl[index, ...]
    Y = Y[index]
    print('train hsi data shape:{},train lidar data shape:{}'.format(Xh.shape,Xl.shape))
    return Xh, Xl, Y
 
    
def image_data_generator(mode,data, label, r, stride, batch_size=50, seed=0):
    if len(data.shape) == 2:
        h, w = data.shape
    if len(data.shape) == 3:
        h, w, c = data.shape
    mdatas = []
    ldatas = []
    count = 0
    while True:
        xulie1=list(range(variables.ksize,h-variables.ksize,stride))
        random.shuffle(xulie1)
        xulie2=list(range(variables.ksize,w-variables.ksize,stride))
        random.shuffle(xulie2)
        for i in xulie1:
            for j in xulie2:
                tmpl=label[i - r:i + r + 1, j - r:j + r + 1]
                mdatas.append(data[i - r:i + r + 1, j - r:j + r + 1])
                ldatas.append(tmpl)
                count += 1
                if count % batch_size == 0:
                    mdatas = np.asarray(mdatas, dtype=np.float32)
                    ldatas = np.asarray(ldatas, dtype=np.float32)
                    np.random.seed(seed)
                    index = np.random.permutation(ldatas.shape[0])
                    train_mdatas = mdatas[index]
                    random_y_train = np.random.rand(variables.BATCH_SIZE,1)
                    if len(ldatas.shape)==3:
                        train_ldatas = np.expand_dims(ldatas[index], 3)
                    if len(ldatas.shape)==4:
                        train_ldatas = ldatas[index]                     
                    if mode=='H_L':
                        yield ([train_mdatas,train_ldatas],[random_y_train,train_mdatas,train_ldatas])
                    if mode=='L_H':
                        yield (train_ldatas,train_mdatas)
                    mdatas = []
                    ldatas = []
                    train_mdatas = []
                    train_ldatas = []

def save_map(pred, show=False):
    import matplotlib.pyplot as plt
    import tifffile as tiff
    pred = np.argmax(pred, axis=2)
    pred = np.asarray(pred, dtype=np.int8)
    # print(pred.shape)
    savename = unicode(imageName.split('.')[0], "utf8", errors='ignore')
    tiff.imsave('../data/results/' + savename + '_mulscale.tif', pred)
    if show:
        plt.imshow(pred)
        plt.show()


def eval(predication, labels):
    """
    evaluate test score
    """
    num = labels.shape[0]
    count = 0
    for i in range(num):
        if(np.argmax(predication[i]) == labels[i]):
            count += 1
    return 100.0 * count / num


def generate_map(predication, idx, idy):
    maps = np.zeros([HEIGHT, WIDTH])
    for i in xrange(len(idx)):
        maps[idx[i], idy[i]] = np.argmax(predication[i]) + 1
    return maps

def generate_map(predication, idx, idy,HEIGHT,WIDTH):
    maps = np.zeros([HEIGHT, WIDTH])
    for i in xrange(len(idx)):
        maps[idx[i], idy[i]] = np.argmax(predication[i]) + 1
    return maps


def random_channel_shift(x, intensity, channel_index=0):
    x = np.rollaxis(x, channel_index, 0)
    min_x, max_x = np.min(x), np.max(x)
    channel_images = [np.clip(x_channel + np.random.uniform(-intensity, intensity), min_x, max_x)
                      for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index + 1)
    return x

def split_image_to_cols(image, iCols, stride, shapes):
    # image=cv2.imread(imagename)
    image = np.asarray(image, dtype=np.float32)
    radius = np.max(shapes) // 2
    image /= 255.0
    image -= np.mean(image)
    h, w, c = image.shape
    XS = []
    XM = []
    XL = []
    for j in xrange(radius, w - np.max(shapes), stride):
        XS.append(image[iCols - radius:iCols - radius +
                        shapes[0], j - radius:j - radius + shapes[0]])
        XM.append(image[iCols - radius:iCols - radius +
                        shapes[1], j - radius:j - radius + shapes[1]])
        XL.append(image[iCols - radius:iCols - radius +
                        shapes[2], j - radius:j - radius + shapes[2]])
    XS = np.asarray(XS, dtype=np.float32)
    XM = np.asarray(XM, dtype=np.float32)
    XL = np.asarray(XL, dtype=np.float32)
    return XS, XM, XL


def sharpen_image(image):
    """
    sharpen image with gaussian kernel
    """
    import cv2
    kernel_sharpen = np.array([[-1, -1, -1, -1, -1],
                               [-1, 2, 2, 2, -1],
                               [-1, 2, 8, 2, -1],
                               [-1, 2, 2, 2, -1],
                               [-1, -1, -1, -1, -1]]) / 8.0
    return cv2.filter2D(image, -1, kernel_sharpen)


def permu(listseq):
    """
    random permutation all file list
    """
    index = np.random.permutation(len(listseq))
    new_list = []
    for i in xrange(len(listseq)):
        new_list.append(listseq[index[i]])
    return new_list


def adjust_brightness(image, delta=None):
    if delta is None:
        delta = np.random.random() * 0.6 * np.amax(image)
        return image + delta
    else:
        return image + delta


def adjust_gamma(image, gamma=1.0):
    """ 
    build a lookup table mapping the pixel values [0, 255] to
        their adjusted gamma values
        """
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) *
                      255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table) / 255.0


def adjust_conrast(image, alpha=1.0, beta=0.0):
    image = image * 1.0 / 255.0
    return image * alpha + beta



