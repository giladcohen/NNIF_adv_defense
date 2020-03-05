from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gdown
import os
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegressionCV
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


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


# lid of a batch of query points X
def mle_batch(data, batch, k):
    data = np.asarray(data, dtype=np.float32)
    batch = np.asarray(batch, dtype=np.float32)

    k = min(k, len(data)-1)
    f = lambda v: - k / np.sum(np.log(v/v[-1]))
    a = cdist(batch, data)
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:,1:k+1]
    a = np.apply_along_axis(f, axis=1, arr=a)
    return a

def train_lr(X, y):
    """
    :param X: the data samples
    :param y: the labels
    :return:
    """
    lr = LogisticRegressionCV(n_jobs=-1, max_iter=20000, cv=3).fit(X, y)
    return lr

def compute_roc(y_true, y_pred, plot=False):
    """
    :param y_true: ground truth
    :param y_pred: predictions
    :param plot:
    :return:
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred)
    if plot:
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color='blue',
                 label='ROC (AUC = %0.4f)' % auc_score)
        plt.legend(loc='lower right')
        plt.title("ROC Curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.show()

    return fpr, tpr, auc_score

