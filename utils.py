"""
工具集合
"""

from scipy.misc import imread, imresize
import tensorflow as tf
import numpy as np
import os
import random
from glob import glob


def image_generator(root: str, batch_size=1, resize: tuple = None, crop: tuple = None, flip: float = None,
                    value_mode: str = "origin", rand=True):
    """
    图片获取生成器，随机获取图片数据

    :param root: 遍历目录
    :param batch_size: 分组数量
    :param resize: 更改大小（None为不）
    :param crop: 随机截取（None为不）
    :param flip: 概率翻转（None为不）
    :param value_mode: 结果模式（origin: 原样，sigmoid: 0~1区间，tanh: -1~1区间）
    :return: 返回一个迭代器
    """

    img_list = glob(os.path.join(root, '*.jpg'))
    if rand:
        while True:
            imgs = []
            for _ in range(batch_size):
                filename = random.choice(img_list)
                img = imread(filename, mode='RGB')
                if resize:
                    img = imresize(img, resize)
                if crop:
                    left = random.randint(0, img.shape[0] - crop[0])
                    top = random.randint(0, img.shape[1] - crop[1])
                    img = img[left:left + crop[0], top:top + crop[1]]
                if flip:
                    if random.random() < flip:
                        img = img[:, ::-1, :]
                imgs.append(img)
            imgs = np.array(imgs)
            if value_mode == 'origin':
                yield imgs
            elif value_mode == 'sigmoid':
                yield (imgs / 128.0) - 128.0
            elif value_mode == 'tanh':
                yield imgs / 255.0
    else:
        imgs = []
        for filename in img_list:
            img = imread(filename, mode='RGB')
            if resize:
                img = imresize(img, resize)
            if crop:
                left = random.randint(0, img.shape[0] - crop[0])
                top = random.randint(0, img.shape[1] - crop[1])
                img = img[left:left + crop[0], top:top + crop[1]]
            if flip:
                if random.random() < flip:
                    img = img[:, ::-1, :]
            imgs.append(img)
            if len(imgs) == batch_size:
                rt = np.array(imgs)
                imgs = []
                if value_mode == 'origin':
                    yield rt
                elif value_mode == 'sigmoid':
                    yield (rt / 128.0) - 128.0
                elif value_mode == 'tanh':
                    yield rt / 255.0


def visual_grid(X: np.array, shape: tuple((int, int))):
    """
    将X中的图片平铺放入新的numpy.array中，用于可视化

    :param X: 图片集合(numpy.array)
    :param shape: 表格形状（行，列）图片数
    :return: 合成后图片array
    """
    nh, nw = shape
    h, w = X.shape[1:3]
    img = np.zeros((h * nh, w * nw, 3))
    for n, x in enumerate(X):
        j = n // nw
        i = n % nw
        if n >= nh * nw:
            break
        img[j * h:j * h + h, i * w:i * w + w, :] = x
    return img


def namespace(default_name):
    """
    variable space装饰器

    产生带name的装饰器
    :param fn: 待装饰函数
    :return:
    """

    def deco(fn):
        def wrapper(*args, **kwargs):
            if 'name' in kwargs:
                name = kwargs['name']
                kwargs.pop('name')
            else:
                name = default_name
            with tf.variable_scope(name):
                return fn(*args, **kwargs)
        return wrapper
    return deco


class DataPool:
    """
    数据池
    用以装载固定量的数据，并提供获取全部及随机获取一个的途径
    """
    def __init__(self, size=50):
        self._pool = []
        self.size = size

    def push(self, data):
        self._pool.extend(data)
        self._pool = self._pool[-self.size:]

    def choice(self):
        return random.choice(self._pool)

    def all(self):
        return self._pool

    def size(self):
        return len(self._pool)
