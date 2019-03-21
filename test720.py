#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os

caffe_root = '/home/qxd/workspace/caffeBVLCplus'
sys.path.insert(0, os.path.join(caffe_root, 'python'))

import caffe

caffe.set_mode_gpu()

import numpy as np
import matplotlib.pyplot as plt
from pylab import *

import cv2
import time
from PIL import Image
from keras.preprocessing.image import *
# from keras.models import load_model
import keras.backend as K
from keras.applications.imagenet_utils import preprocess_input
from queue import Queue
from multiprocessing import Process, Lock
import pdb
from models import *

import math
import ctypes



def inference(weight_file, data_dir, save_dir):
     
    caffe_model_dir = 'caffeModel'
    caffe_net_file = os.path.join(caffe_model_dir, 'fasterModel720_deploy.prototxt')
    caffe_params_file = os.path.join(caffe_model_dir, weight_file)

    caffe_model = caffe.Net(caffe_net_file, caffe_params_file, caffe.TEST)
    #image_size = image_size

    image_list = os.listdir(data_dir)
    image_list.sort()
    total = 1
    for image_name in image_list:
        image = Image.open('%s/%s' % (data_dir, image_name))
        image_origin_size = image.size
        # newsize_image = resize_image(image, image_size, newsize_image_size)
        # newsize_image = Image.fromarray(np.resize(image,(720,720,3)),mode='RGB')
        newsize_image = image.resize((720, 720),Image.ANTIALIAS)
        newsize_image = img_to_array(newsize_image)  # , data_format='default')

        newsize_image = np.expand_dims(newsize_image, axis=0)
        newsize_image = preprocess_input(newsize_image)

        caffe_model.blobs['data'].data[...] = np.transpose(newsize_image, (0, 3, 1, 2))  # 
        caffe_out = caffe_model.forward()
        duration = time.time() - start_time
        print('{}s used to make predictions for one image.\n'.format(duration))
        output = caffe_out['Deconv_6']
        output = np.transpose(output, (0, 2, 3, 1))
        result = np.squeeze(output)
         
        result = np.argmax(result, axis=-1).astype(np.uint8)
        midresult_img = Image.fromarray(result, mode='P')
        result_img = midresult_img.resize(image_origin_size,Image.ANTIALIAS)
        
        
        #result_img = np.asarray(result_img)
        #result_img=cv2.GaussianBlur(result_img,(1,1),100)
        #midresult_img = Image.fromarray(result, mode='P')
        
        result_np = np.asarray(result_img)
        result_np = (result_np * 255).astype(uint8)
        #cv2.imwrite(os.path.join(save_dir, os.path.splitext(image_name)[0] + '.png'), result_np)
        cv2.imwrite(os.path.join(save_dir,image_name), result_np)
         
        print('pic: %d done!' % total)
        total = total + 1
    return result_img

if __name__ == '__main__':
    start_time = time.time()
    lock = Lock()
    lock.acquire()
    results = inference('caffeModel_3.caffemodel', 'data/image/', 'image_result')
    lock.release()
    total_time = time.time() - start_time
    print('{}s used\n'.format(total_time))
