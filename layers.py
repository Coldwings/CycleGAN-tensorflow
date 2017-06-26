"""
需要用到的层（或op）
"""

import tensorflow as tf
import numpy as np
from utils import namespace


@namespace('instance_norm')
def instancenorm(input):
    """
    单项正则化（Generator中用到）
    :param input: 输入Tensor
    :return: 输出Tensor
    """
    depth = input.get_shape()[3]
    scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
    offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
    mean, variance = tf.nn.moments(input, axes=[1, 2], keep_dims=True)
    epsilon = 1e-5
    inv = tf.rsqrt(variance + epsilon)
    normalized = (input - mean) * inv
    return tf.add(tf.multiply(scale, normalized), offset)


@namespace('batch_norm')
def batchnorm(x):
    """
    批数据正则
    此处不进行trainning控制，改而使用变量搜集后，在反向传播时限制训练变量进行控制

    :param x:
    :return:
    """
    return tf.layers.batch_normalization(x)


@namespace('reflect_pad')
def reflect_pad(x, size=1):
    """
    反射式补边
    以边框临近位置的内容进行补边
    :param x: Tensor
    :param size: 边框大小
    :return: Tensor
    """
    return tf.pad(x, ((0, 0), (size, size), (size, size), (0, 0)), mode='REFLECT')


@namespace('conv2d')
def conv2d(input, filter, kernel, strides=1, stddev=0.02, padding='VALID'):
    """
    卷积层

    :param input: Tensor
    :param filter: filter数量
    :param kernel: kernel大小（正方形kernel）
    :param strides: 步长
    :param stddev: 初始化标准差
    :param padding: 补零方式（VALID、SAME）
    :return: Tensor
    """
    w = tf.get_variable(
        'w',
        (kernel, kernel, input.get_shape()[-1], filter),
        initializer=tf.truncated_normal_initializer(stddev=stddev)
    )
    conv = tf.nn.conv2d(input, w, strides=[1, strides, strides, 1], padding=padding)
    b = tf.get_variable(
        'b',
        [filter],
        initializer=tf.constant_initializer(0.0)
    )
    conv = tf.reshape(tf.nn.bias_add(conv, b), tf.shape(conv))
    return conv


@namespace('leaky_relu')
def lrelu(x, leak=0.2):
    """
    Leaky ReLU激活函数
    :param x: Tensor
    :param leak: 负轴的泄露系数
    :return: Tensor
    """
    return tf.maximum(x, tf.multiply(x, leak))


@namespace('res_block')
def res_block(x, filters):
    """
    残差块
    
    :param x: Tensor 
    :param filters: filter数量
    :return: Tensor
    """
    y = reflect_pad(x, name='rp1')
    y = conv2d(y, filters, 3, name='conv1')
    y = lrelu(y)
    y = reflect_pad(y, name='rp2')
    y = conv2d(y, filters, 3, name='conv2')
    y = lrelu(y)
    return tf.add(x, y)


@namespace('unsampling')
def upsampling(x):
    """
    二倍上采样
    :param x: Tensor
    :return: Tensor
    """
    input_shape = x.get_shape()
    output_shape = (None, input_shape[1] * 2, input_shape[2] * 2, None)
    x = tf.image.resize_bilinear(x, tf.shape(x)[1:3] * tf.constant(np.array([2, 2], dtype='int32')), name='upsample')
    x.set_shape(output_shape)
    return x


@namespace('conv2dTranspose')
def deconv2d(x, filters):
    """
    二倍缩放的转置卷积（参考SRGAN）
    :param x: Tensor
    :param filters: filter数量
    :return: Tensor
    """
    x = upsampling(x, name='upsample')
    x = reflect_pad(x, 1, name='rp')
    x = conv2d(x, filters, 3, name='conv')
    return x


@namespace('abs_criterion')
def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))


@namespace('mae_criterion')
def mae_criterion(in_, target):
    return tf.reduce_mean((in_ - target) ** 2)


@namespace('sce_criterion')
def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
