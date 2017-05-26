import tensorflow as tf
import numpy as np
import scipy.io

vgg_layers = (
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
    'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
    'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
    'relu5_3', 'conv5_4', 'relu5_4'
)


def prepare_model(path):
    vgg_rawnet = scipy.io.loadmat(path)
    return vgg_rawnet['layers'][0] # another solution: global vgg_weights

def build_image_net(input_tensor, vgg_weights, feature_pooling_type):
    net = {}
    current = input_tensor

    for i, name in enumerate(vgg_layers):
        layer_kind = name[:4]
        if layer_kind == 'conv':
            weights, bias = vgg_weights[i][0][0][2][0]
            bias = bias.reshape(-1)
            current = conv_layer(current, tf.constant(weights), tf.constant(bias))
        elif layer_kind == 'relu':
            current = tf.nn.relu(current)
        elif layer_kind == 'pool':
            current = pool_layer(current, feature_pooling_type)
        net[name] = current

    return net

def conv_layer(input, W, b):
    conv =  tf.nn.conv2d(input, W, strides=[1,1,1,1], padding='SAME')
    return conv + b

def pool_layer(input, feature_pooling_type):
    if feature_pooling_type == 'avg':
        return tf.nn.avg_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    elif feature_pooling_type == 'max':
        return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def build_mask_net(input_tensor, mask_downsample_type):
    net = {}
    current = input_tensor

    # soft
    if mask_downsample_type == 'simple':
        for name in vgg_layers:
            layer_kind = name[:4]
            if layer_kind == 'pool':
                current = tf.nn.avg_pool(current, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
            net[name] = current
    # hard
    elif mask_downsample_type == 'all':
        for name in vgg_layers:
            layer_kind = name[:4]
            if layer_kind == 'conv':
                current = tf.nn.max_pool(current, ksize=[1,3,3,1], strides=[1,1,1,1], padding='SAME')
            elif layer_kind == 'pool':
                current = tf.nn.max_pool(current, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
            net[name] = current     
    # hard, keep the padding boundary unchanged
    elif mask_downsample_type == 'inside':
        current = 1 - current
        for name in vgg_layers:
            layer_kind = name[:4]
            if layer_kind == 'conv':
                current = tf.nn.max_pool(current, ksize=[1,3,3,1], strides=[1,1,1,1], padding='SAME')
            elif layer_kind == 'pool':
                current = tf.nn.max_pool(current, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
            net[name] = 1 - current 
    # soft
    elif mask_downsample_type == 'mean':
        for name in vgg_layers:
            layer_kind = name[:4]
            if layer_kind == 'conv':
                current = tf.nn.avg_pool(current, ksize=[1,3,3,1], strides=[1,1,1,1], padding='SAME')
            elif layer_kind == 'pool':
                current = tf.nn.avg_pool(current, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
            net[name] = current 

    return net






