import keras
import os
from sklearn.metrics import roc_auc_score
import sys
import matplotlib.pyplot as plt
from keras.models import Model
import numpy as np
from config import *

class Histories(keras.callbacks.Callback):
	def __init__(self, isShow):
		self.isShow = isShow

	def on_train_begin(self, logs={}):
		self.aucs = []
		self.losses = []

	def on_train_end(self, logs={}):
		return

	def on_epoch_begin(self, epoch, logs={}):
		return

	def on_epoch_end(self, epoch, logs={}):
		
		print('\n=========')
		print(len(self.validation_data)) #be careful of the dimenstion of the self.validation_data, somehow some extra dim will be included
		print(self.validation_data[0].shape)
		print(self.validation_data[1].shape)
		print('=========')
		#(IMPORTANT) Only use one input: "inputs=self.model.input[0]"
		ip1_input = self.model.input #this can be a list or a matrix. 
		if self.isShow:
			ip1_input = self.model.input
			labels = self.validation_data[0] # original hsi data
		
		# ip1_layer_model = Model(inputs=ip1_input, outputs=self.model.get_layer('spat').output)
		ip1_layer_model = Model(inputs=[self.model.get_layer('input_2').output,self.model.get_layer('input_3').output],outputs=[self.model.get_layer('hsi').output,self.model.get_layer('lidar').output])
		ip1_output = ip1_layer_model.predict([np.array(self.validation_data[0]),np.array(self.validation_data[1])],batch_size=variables.BATCH_SIZE)
		
		visualize(ip1_output[0],ip1_output[1],epoch)
		
		return

	def on_batch_begin(self, batch, logs={}):
		return

	def on_batch_end(self, batch, logs={}):
		return


def visualize(feat, labels, epoch):
    os.makedirs('images/%s' % variables.dataset, exist_ok=True)
    r, c = 2, 2
    titles = ['HSI spatial', 'New spatial', 'HSI spectral','New spectral']
    fig, axs = plt.subplots(r, c)
    axs[0,0].imshow(labels[0][:,:,1],cmap ='gray')
    axs[0,0].set_title(titles[0])
    axs[0,0].axis('off')
   
    axs[0,1].imshow(feat[0][:,:,1],cmap ='gray')
    axs[0,1].set_title(titles[1])
    axs[0,1].axis('off')    
    
    feat_x=np.linspace(0,1,feat.shape[3])
    for n in range(len(feat[0][0])):
        for m in range(len(feat[0][0])):
            axs[1,0].plot(feat_x, labels[0,n,m,:], '-')
            # plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc = 'upper right')
    axs[1,0].set_title(titles[2])
    axs[1,0].axis('off')

    for n in range(len(feat[0][0])):
        for m in range(len(feat[0][0])):
            axs[1,1].plot(feat_x, feat[0,n,m,:], '-')
            # plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc = 'upper right')
    axs[1,1].set_title(titles[3])
    axs[1,1].axis('off')
    fig.savefig("images/%s/%d.png" % (variables.dataset, epoch))
    plt.close()
