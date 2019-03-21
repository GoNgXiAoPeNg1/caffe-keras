import os
import sys
caffe_root='/home/qxd/workspace/caffeBVLCplus'
sys.path.insert(0,os.path.join(caffe_root,'python'))
import caffe
from caffe import layers as L, params as P
import math
import numpy as np

def set_padding(config_keras, input_shape, config_caffe):
    if config_keras['padding'] == 'valid':
        return
    elif config_keras['padding'] == 'same':
        # pad = ((layer.output_shape[1] - 1)*strides[0] + pool_size[0] - layer.input_shape[1])/2
        # pad = pool_size[0]/(strides[0]*2)
        # pad = (pool_size[0]*layer.output_shape[1] - (pool_size[0]-strides[0])*(layer.output_shape[1]-1) - layer.input_shape[1])/2
        
        if 'kernel_size' in config_keras:
            kernel_size = config_keras['kernel_size']
        elif 'pool_size' in config_keras:
            kernel_size = config_keras['pool_size']
        else:
            raise Exception('Undefined kernel size')
        pad_w = int(kernel_size[1] / 2)
        pad_h = int(kernel_size[0] / 2)
        if config_keras['dilation_rate'][0]>1:
            pad_w=2
            pad_h=2
        strides = config_keras['strides']
        w = input_shape[1]
        h = input_shape[2]
        
        out_w = math.ceil(w / float(strides[1]))
        # pad_w = int((kernel_size[1]*out_w - (kernel_size[1]-strides[1])*(out_w - 1) - w)/2)
        
        out_h = math.ceil(h / float(strides[0]))
        # pad_h = int((kernel_size[0]*out_h - (kernel_size[0]-strides[0])*(out_h - 1) - h)/2)
        
        if pad_w == 0 and pad_h == 0:
            return
        
        if pad_w == pad_h:
            config_caffe['pad'] = pad_w
        else:
            config_caffe['pad_h'] = pad_h
            config_caffe['pad_w'] = pad_w
        
    else:
        raise Exception(config_keras['padding']+' padding is not supported')
def convert(keras_model, caffe_net_file, caffe_params_file):
    caffe_net = caffe.NetSpec()
    net_params = dict()
    outputs = dict()
    shape = ()
    input_str = ''

    for layer in keras_model.layers:
        name = layer.name
        print('processing'+name+"================================================================================")
        layer_type = type(layer).__name__
        config = layer.get_config()
        blobs = layer.get_weights()
        blobs_num = len(blobs)

        top2='qinxiao'
        if type(layer.output) == list:
            # raise Exception('Layers with multiply outputs are not supported')
            top=layer.output[0].name
        else: 
            top = layer.output.name
        
        if type(layer.input) != list:
            bottom = layer.input.name
        
        # data
        if layer_type == 'InputLayer' or not hasattr(caffe_net, 'data'):
            input_name = 'data'
            caffe_net[input_name] = L.Layer()
            input_shape = config['batch_input_shape']
            input_str = 'input: {}\ninput_dim: {}\ninput_dim: {}\ninput_dim: {}\ninput_dim: {}'.format('"' + input_name + '"',
                1, input_shape[3], input_shape[1], input_shape[2])
            outputs[layer.input.name] = input_name
            if layer_type == 'InputLayer':
                continue
        # conv
        if layer_type == 'Conv2D' or layer_type == 'Convolution2D':
            strides = config['strides']
            kernel_size = config['kernel_size']
            dilation=config['dilation_rate']
            kwargs = {'num_output': config['filters']}
            
            if dilation[0]==dilation[1]:
                kwargs['dilation'] = dilation[0]
            if kernel_size[0] == kernel_size[1]:
                kwargs['kernel_size'] = kernel_size[0]
            else:
                kwargs['kernel_h'] = kernel_size[0]
                kwargs['kernel_w'] = kernel_size[1]
            
            if strides[0] == strides[1]:
                kwargs['stride'] = strides[0]
            else:
                kwargs['stride_h'] = strides[0]
                kwargs['stride_w'] = strides[1]
            
            if not config['use_bias']:
                kwargs['bias_term'] = False
                # kwargs['param']=[dict(lr_mult=0)]
            else:
                # kwargs['param']=[dict(lr_mult=0), dict(lr_mult=0)]
                pass
            
            set_padding(config, layer.input_shape, kwargs)
            
            caffe_net[name] = L.Convolution(caffe_net[outputs[bottom]], **kwargs)

            blobs[0] = np.array(blobs[0]).transpose(3, 2, 0, 1)
            net_params[name] = blobs

            if config['activation'] == 'relu':
                name_s = name+'s'
                caffe_net[name_s] = L.ReLU(caffe_net[name], in_place=True)
            elif config['activation'] == 'sigmoid':
                name_s = name+'s'
                caffe_net[name_s] = L.Sigmoid(caffe_net[name], in_place=True)
            elif config['activation'] == 'tanh':
                caffe_net[name_s] = L.TanH(caffe_net[name], in_place=True)
            elif config['activation'] == 'linear':
                pass
            else:
                raise Exception('Unsupported activation '+config['activation'])
        elif layer_type == 'Conv2DTranspose':
            
            # Stride
            if layer.strides is None:
                strides = (1, 1)
            else:
                strides = layer.strides
            use_bias = config['use_bias']
            param = dict(bias_term=use_bias)

            # Padding
            if layer.padding == 'same':  # Calculate the padding for 'same'
                padding = [layer.kernel_size[0] / 2, layer.kernel_size[1] / 2]
            else:
                padding = [0, 0]  # If padding is valid(aka no padding)

            
            param['pad']=padding[0]
            if strides[0]==2:
                param['pad']=0
            param['kernel_size']=layer.kernel_size[0]
            param['stride']=strides[0]
            param['num_output']=layer.filters

            # if strides[0] == strides[1]:
            #     kwargs['stride'] = strides[0]
            # else:
            #     kwargs['stride_h'] = strides[0]
            #     kwargs['stride_w'] = strides[1]
            caffe_net[name] = L.Deconvolution(caffe_net[outputs[bottom]],
                                  convolution_param=param)
            
            # caffe_net[name] = L.Deconvolution(caffe_net[outputs[bottom]], **kwargs)

            blobs[0] = np.array(blobs[0]).transpose(3, 2, 0, 1)
            net_params[name] = blobs

            if config['activation'] == 'relu':
                name_s = name+'s'
                caffe_net[name_s] = L.ReLU(caffe_net[name], in_place=True)
            elif config['activation'] == 'sigmoid':
                name_s = name+'s'
                caffe_net[name_s] = L.Sigmoid(caffe_net[name], in_place=True)
            elif config['activation'] == 'tanh':
                caffe_net[name_s] = L.TanH(caffe_net[name], in_place=True)
            elif config['activation'] == 'linear':
                pass
            else:
                raise Exception('Unsupported activation '+config['activation'])
            
            if name=='Deconv_2':
                name_crop = name+'_crop'
                caffe_net.data1 = L.Input(shape=dict(dim=[1, 512, 90, 90]))
                caffe_net[name_crop] = L.Crop(caffe_net[name], caffe_net.data1, axis=1, offset=0)
            if name=='Deconv_3':
                name_crop = name+'_crop'
                caffe_net.data2 = L.Input(shape=dict(dim=[1, 256, 180, 180]))
                caffe_net[name_crop] = L.Crop(caffe_net[name], caffe_net.data2, axis=1, offset=0)
        elif layer_type == 'BatchNormalization':
            param = dict()
            variance = np.array(blobs[-1])
            mean = np.array(blobs[-2])
            # print('blobs'+str(blobs_num))
            # print(blobs)

            # print('config')
            # print(config)
            if config['scale']:
                gamma = np.array(blobs[0])
                sparam = [dict(lr_mult=1), dict(lr_mult=1)]
            else:
                gamma = np.ones(mean.shape, dtype=np.float32)
                # sparam = [dict(lr_mult=0, decay_mult=0), dict(lr_mult=1, decay_mult=1)]
                sparam = [dict(lr_mult=0), dict(lr_mult=1)]
                # sparam = [dict(lr_mult=0), dict(lr_mult=0)]
            
            if config['center']:
                beta = np.array(blobs[-3])
                param['bias_term'] = True
            else:
                beta = np.zeros(mean.shape, dtype=np.float32)
                param['bias_term'] = False
            
            # caffe_net[name] = L.BatchNorm(caffe_net[outputs[bottom]], moving_average_fraction=layer.momentum, eps=layer.epsilon)
            caffe_net[name] = L.BatchNorm(caffe_net[outputs[bottom]], moving_average_fraction=layer.momentum, eps=layer.epsilon)
           
            # param = [dict(lr_mult=1, decay_mult=1), dict(lr_mult=1, decay_mult=1), dict(lr_mult=0, decay_mult=0)])
            # param = [dict(lr_mult=1), dict(lr_mult=1), dict(lr_mult=0)])
                
            net_params[name] = (mean, variance, np.array(1.0)) 
            
            name_s = name+'_scale'
            
            caffe_net[name_s] = L.Scale(caffe_net[name], in_place=True, param=sparam, scale_param={'bias_term': config['center']})
           
            net_params[name_s] = (gamma, beta)
        elif layer_type == 'Activation':
            if config['activation'] == 'relu':
                # caffe_net[name] = L.ReLU(caffe_net[outputs[bottom]], in_place=True)
                if len(layer.input.consumers()) > 1:
                    caffe_net[name] = L.ReLU(caffe_net[outputs[bottom]])
                else:
                    caffe_net[name] = L.ReLU(caffe_net[outputs[bottom]], in_place=True)
            elif config['activation'] == 'tanh':
                if len(layer.input.consumers()) > 1:
                    caffe_net[name] = L.TanH(caffe_net[outputs[bottom]])
                else:
                    caffe_net[name] = L.TanH(caffe_net[outputs[bottom]], in_place=True)
            elif config['activation'] == 'relu6':
                caffe_net[name] = L.ReLU(caffe_net[outputs[bottom]])
            elif config['activation'] == 'softmax':
                caffe_net[name] = L.Softmax(caffe_net[outputs[bottom]], in_place=True)
            else:
                raise Exception('Unsupported activation '+config['activation'])
            
        elif layer_type == 'range':
            kwargs={}
            kwargs['pool'] = P.Pooling.MAX   
            # config['padding']='same'
            pool_size = (3,3)
            strides = (2,2)
            config['pool_size']=pool_size
            config['strides']=strides
            if pool_size[0] != pool_size[1]:
                raise Exception('Unsupported pool_size')
                
            if strides[0] != strides[1]:
                raise Exception('Unsupported strides')
            caffe_net[name] = L.Pooling(caffe_net[outputs[bottom]], kernel_size=pool_size[0], stride=strides[0], **kwargs)

        elif layer_type=='MaxUnpooling2D':
            kwargs={}
            kwargs['unpool'] = P.Pooling.MAX   
            # config['padding']='same'
            unpool_size = (3,3)
            strides = (2,2)
            config['unpool_size']=pool_size
            config['strides']=strides
            if unpool_size[0] != unpool_size[1]:
                raise Exception('Unsupported pool_size')
                
            if strides[0] != strides[1]:
                raise Exception('Unsupported strides')
            caffe_net[name] = L.Unpooling(caffe_net[outputs[bottom]], unpool=P.Unpooling.MAX,kernel_size=3, unpool_h=360,unpool_w=360)

        elif layer_type == 'Add':
            layers = []
            for i in layer.input:
                layers.append(caffe_net[outputs[i.name]])
            caffe_net[name] = L.Eltwise(*layers)
        else:
            raise Exception('Unsupported layer type: '+layer_type)
        outputs[top] = name
        if name=='Deconv_2':
            outputs[top] = name+'_crop'
        if name=='Deconv_3':
            outputs[top] = name+'_crop'
    net_proto = input_str + '\n' + 'layer {' + 'layer {'.join(str(caffe_net.to_proto()).split('layer {')[2:])

    f = open(caffe_net_file, 'w') 
    f.write(net_proto)
    print("prototxt is done!")
    f.close()
    
    caffe_model = caffe.Net(caffe_net_file, caffe.TEST)
    
    for layer in caffe_model.params.keys():
        if 'up_sampling2d' in layer:
            continue
        for n in range(0, len(caffe_model.params[layer])):
            print('layer:', layer)
            print("n:", n)
            print((caffe_model.params[layer][n].data[...]).shape)
            print((net_params[layer][n]).shape)
            caffe_model.params[layer][n].data[...] = net_params[layer][n]

    caffe_model.save(caffe_params_file)