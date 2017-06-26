"""
模块化的生成器与判别器
"""

import tensorflow as tf
from utils import namespace
from layers import *


@namespace('generator')
def generator(x, reuse, gdim=32):
    """
    生成器

    :param x: Tensor
    :param reuse: 是否重用变量
    :return: Tensor
    """
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.variable_scope('encoder'):
        # 256x256x3 --> #256x256x32
        x = reflect_pad(x, 3, name='rpinput')
        x = conv2d(x, gdim, 7, 1, name='convinput')
        x = instancenorm(x, name='insinput')
        x = lrelu(x)
        # 256x256x32 --> #64x64x256
        for i in range(1, 4):
            x = reflect_pad(x, name='rp%d' % i)
            x = conv2d(x, gdim * (2 ** i), 3, 2, name='conv%d' % i)
            x = instancenorm(x, name='insn%d' % i)
            x = lrelu(x)
        rs = x  ## for short-cut As RSGAN
    with tf.variable_scope('transform'):
        # 64x64x256 --> 64x64x256
        for i in range(9):
            x = res_block(x, gdim * (2 ** 3), name='res%d' % (i + 1))
    with tf.variable_scope('decoder'):
        # 64x64x256 --> 256x256x3
        x = x + rs
        for i in range(3, 0, -1):
            x = deconv2d(x, gdim * (2 ** i), name='convT%d' % i)
            x = instancenorm(x, name='insn%d' % i)
            x = lrelu(x)
        x = conv2d(x, 3, 3, padding='SAME', name='convout')
        x = tf.nn.sigmoid(x, name='output')
    return x


@namespace('discriminator')
def discriminator(x, reuse, ddim=64):
    """
    判别器

    :param x: Tensor
    :param reuse: 是否重用
    :return: Tensor
    """
    if reuse:
        tf.get_variable_scope().reuse_variables()
    x = conv2d(x, ddim, 7, 2, name='input')
    x = batchnorm(x, name='bn1')
    x = lrelu(x)
    x = conv2d(x, ddim * 2, 3, 2, name='conv2')
    x = batchnorm(x, name='bn2')
    x = lrelu(x)
    x = conv2d(x, ddim * 4, 3, 2, name='conv3')
    x = batchnorm(x, name='bn3')
    x = lrelu(x)
    x = conv2d(x, ddim * 8, 3, 2, name='conv4')
    x = batchnorm(x, name='bn4')
    x = lrelu(x)
    x = conv2d(x, 1, 3, name='output')
    return x
