# -*- coding: UTF-8 -*-
import pickle as p
import numpy as np
import os


def load_CIFAR_batch(filename):
    with open(filename, 'rb') as f:
        datadict = p.load(f, encoding='bytes')
        X = datadict[b'data']
        Y = datadict[b'labels']
        # X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        X = X.reshape(10000, 3, 32, 32).astype("float")
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(ROOT):
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)         
        ys.append(Y)
    Xtr = np.concatenate(xs) 
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte

import numpy as np
import matplotlib.pyplot as plt

cifar10_dir = 'cifar-10-batches-py'
x_train, y_train, x_test, y_test = load_CIFAR10(cifar10_dir)


if __name__ == '__main__':
    print('Training data shape: ', x_train.shape)
    print('Training labels shape: ', y_train.shape)
    print('Test data shape: ', x_test.shape)
    print('Test labels shape: ', y_test.shape)
    print(x_train[0])
    print(y_train[0:30])
