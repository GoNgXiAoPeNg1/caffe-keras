#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import os
import sys
import cv2
import time
#import liblib.so
import keras2caffe
from PIL import Image
from keras.preprocessing.image import *
from keras.models import load_model
import keras.backend as K
from keras.applications.imagenet_utils import preprocess_input
import pdb
from models import *
import ctypes
caffe_root='/home/qxd/workspace/caffeBVLCplus'
sys.path.insert(0,os.path.join(caffe_root,'python'))
import caffe
caffe.set_mode_cpu()
# os.environ["CUDA_VISIBLE_DEVICES"]='0'
def inference():
    model_name = 'AtrousFCN_DeResnet50_asUnet'
    weight_file = 'checkpoint_weights_3.hdf5'
    model_input_size=(720,720)
    batch_shape = (1, ) + model_input_size + (3, )
    model_weight_path = os.path.join('kerasModel', weight_file)
    keras_model = globals()[model_name](batch_shape=batch_shape, input_shape=(model_input_size[0], model_input_size[1], 3))
    keras_model.load_weights(model_weight_path, by_name=True)

    input_data=np.random.random(keras_model.input.shape)

    caffe_model_dir='caffeModel'
    caffe_net_file= os.path.join(caffe_model_dir,'caffeModel_deploy_3.prototxt')
    caffe_params_file=os.path.join(caffe_model_dir,'caffeModel_3.caffemodel')
    keras2caffe.convert(keras_model, caffe_net_file, caffe_params_file)
    
    
    caffe_model = caffe.Net(caffe_net_file,caffe_params_file,caffe.TEST)
    
    caffe_model.blobs['data'].data[...] = np.transpose(input_data,(0,3,1,2))      #执行上面设置的图片预处理操作，并将图片载入到blob�?
    caffe_out=caffe_model.forward()
    print(caffe_out)
    output=caffe_out['batch_normalization_6']
    output=np.transpose(output,(0,2,3,1))
    print('caffe_out  '+str(output.shape))
    print(output[0,:,:,0])
    
    keras_out=keras_model.predict(input_data,batch_size=1)
    print('keras_out'+str(keras_out.shape)+'========================================================================================================================')
    print(keras_out[0,:,:,0])
    for layer_name, blob in caffe_model.blobs.iteritems():
        print (layer_name + '\t' + str(blob.data.shape))

if __name__ == '__main__':
    inference()
    
