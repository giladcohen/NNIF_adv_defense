from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras.utils.data_utils import get_file
import scipy.io as sio
import gdown
import os
import numpy as np

def load_svhn():
    """ Loads SVHN dataset"""
    cache_dir = os.path.join(os.path.expanduser('~'), '.keras')
    datadir_base = os.path.expanduser(cache_dir)
    datadir = os.path.join(datadir_base, 'datasets', 'svhn')
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    X_train_path = os.path.join(datadir, 'X_train.npy')
    y_train_path = os.path.join(datadir, 'y_train.npy')
    X_test_path  = os.path.join(datadir, 'X_test.npy')
    y_test_path  = os.path.join(datadir, 'y_test.npy')

    if not os.path.exists(X_train_path):
        gdown.download('https://drive.google.com/uc?id=1G1_onGVI9OKRN9ANMS2kKX5_OkN2Pqjd', os.path.join(datadir, 'X_train.npy'), quiet=False)
    if not os.path.exists(y_train_path):
        gdown.download('https://drive.google.com/uc?id=1ijYnRSTB7S2zctjUax2ycW-TgK7Ux2cG', os.path.join(datadir, 'y_train.npy'), quiet=False)
    if not os.path.exists(X_test_path):
        gdown.download('https://drive.google.com/uc?id=1TVhS8ns7fPrtUdLZ2nYUsGtRUQT9yaKC', os.path.join(datadir, 'X_test.npy'), quiet=False)
    if not os.path.exists(y_test_path):
        gdown.download('https://drive.google.com/uc?id=1ySH19ynJmXLsAfjec0mHfdptduRGCgb2', os.path.join(datadir, 'y_test.npy'), quiet=False)

    X_train = np.load(X_train_path)
    y_train = np.load(y_train_path)
    X_test  = np.load(X_test_path)
    y_test  = np.load(y_test_path)

    return (X_train, y_train), (X_test, y_test)

def one_hot(indices, depth):
    """Converting the indices to one hot representation
    :param indices: numpy array
    :param depth: the depth of the one hot vectors
    """
    ohm = np.zeros([indices.shape[0], depth])
    ohm[np.arange(indices.shape[0]), indices] = 1
    return ohm



