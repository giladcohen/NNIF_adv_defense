from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import matplotlib
import platform
# Force matplotlib to not use any Xwindows backend.
if platform.system() == 'Linux':
    matplotlib.use('Agg')

import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.decomposition import PCA
from NNIF_adv_defense.tools.utils import train_lr, compute_roc

from tensorflow.python.platform import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cifar10', 'dataset: cifar10/100 or svhn')
flags.DEFINE_string('seen_attack', '', 'Seen attack when training detector')
flags.DEFINE_string('attack', 'deepfool', 'adversarial attack: deepfool, jsma, cw, cw_nnif')
flags.DEFINE_string('characteristics', 'nnif', 'type of defence: lid/mahalanobis/dknn/nnif')
flags.DEFINE_bool('with_noise', False, 'whether or not to include noisy samples')
flags.DEFINE_bool('only_last', False, 'Using just the last layer, the embedding vector')
flags.DEFINE_integer('pca_features', -1, 'Number of PCA features to train')
flags.DEFINE_string('checkpoint_dir', '', 'Checkpoint dir, the path to the saved model architecture and weights')

# FOR LID
flags.DEFINE_integer('k_nearest', 17, 'number of nearest neighbors to use for LID/DkNN detection')

# FOR MAHANABOLIS
flags.DEFINE_float('magnitude', 0.002, 'magnitude for mahalanobis detection')

# FOR NNIF
flags.DEFINE_integer('max_indices', 200, 'maximum number of helpful indices to use in NNIF detection')
flags.DEFINE_string('ablation', '1111', 'for ablation test')

flags.DEFINE_string('mode', 'null', 'to bypass pycharm bug')
flags.DEFINE_string('port', 'null', 'to bypass pycharm bug')

assert FLAGS.with_noise is False  # TODO(support noise in the future)
rgb_scale = 1.0  # Used for the Mahalanobis detection

if FLAGS.seen_attack != '':
    SEEN_ATTACK = FLAGS.seen_attack
else:
    SEEN_ATTACK = FLAGS.attack

SEEN_ATTACK_TARGETED = SEEN_ATTACK != 'deepfool'
ATTACK_TARGETED      = FLAGS.attack != 'deepfool'

if FLAGS.checkpoint_dir != '':
    model_dir     = FLAGS.checkpoint_dir                      # set user specified dir
else:
    model_dir = os.path.join(FLAGS.dataset, 'trained_model')  # set default dir

seen_attack_dir    = os.path.join(model_dir, SEEN_ATTACK)
attack_dir         = os.path.join(model_dir, FLAGS.attack)

if SEEN_ATTACK_TARGETED:
    seen_attack_dir = seen_attack_dir + '_targeted'
if ATTACK_TARGETED:
    attack_dir      = attack_dir + '_targeted'

seen_characteristics_dir = os.path.join(seen_attack_dir, FLAGS.characteristics)
characteristics_dir      = os.path.join(attack_dir     , FLAGS.characteristics)

if FLAGS.characteristics == 'lid':
    train_characteristics_file = os.path.join(seen_characteristics_dir, 'k_{}_batch_{}_train'.format(FLAGS.k_nearest, 100))
    test_characteristics_file  = os.path.join(characteristics_dir, 'k_{}_batch_{}_test'.format(FLAGS.k_nearest, 100))
elif FLAGS.characteristics == 'mahalanobis':
    train_characteristics_file = os.path.join(seen_characteristics_dir, 'magnitude_{}_scale_{}_train'.format(FLAGS.magnitude, rgb_scale))
    test_characteristics_file  = os.path.join(characteristics_dir, 'magnitude_{}_scale_{}_test'.format(FLAGS.magnitude, rgb_scale))
elif FLAGS.characteristics == 'nnif':
    train_characteristics_file = os.path.join(seen_characteristics_dir, 'max_indices_{}_ablation_{}_train'.format(FLAGS.max_indices, FLAGS.ablation))
    test_characteristics_file  = os.path.join(characteristics_dir, 'max_indices_{}_ablation_{}_test'.format(FLAGS.max_indices, FLAGS.ablation))
elif FLAGS.characteristics == 'dknn':
    train_characteristics_file = os.path.join(seen_characteristics_dir, 'k_{}_train'.format(FLAGS.k_nearest))
    test_characteristics_file  = os.path.join(characteristics_dir, 'k_{}_test'.format(FLAGS.k_nearest))
else:
    raise AssertionError('{} is not supported'.format(FLAGS.characteristics))

if FLAGS.with_noise:
    train_characteristics_file = train_characteristics_file + '_noisy'
    test_characteristics_file  = test_characteristics_file  + '_noisy'

if FLAGS.only_last:
    assert FLAGS.characteristics != 'dknn'  # DkNN is only applied at the last layer
    train_characteristics_file = train_characteristics_file + '_only_last'
    test_characteristics_file  = test_characteristics_file  + '_only_last'

train_characteristics_file = train_characteristics_file + '.npy'
test_characteristics_file  = test_characteristics_file  + '.npy'

def load_characteristics(characteristics_file):
    X, Y = None, None
    data = np.load(characteristics_file)
    if X is None:
        X = data[:, :-1]
    if Y is None:
        Y = data[:, -1]  # labels only need to load once

    return X, Y


print("Loading train attack: {}\nTraining file: {}\nTesting file: {}".format(FLAGS.attack, train_characteristics_file, test_characteristics_file))
X_train, Y_train = load_characteristics(train_characteristics_file)
X_test, Y_test   = load_characteristics(test_characteristics_file)

scaler  = MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)

if FLAGS.pca_features > 0:
    print('Apply PCA decomposition. Reducing number of features from {} to {}'.format(X_train.shape[1], FLAGS.pca_features))
    pca = PCA(n_components=FLAGS.pca_features)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

print("Train data size: ", X_train.shape)
print("Test data size: ", X_test.shape)

## Build detector
print("LR Detector on [dataset: %s, test_attack: %s, characteristics: %s, ablation: %s]:" % (FLAGS.dataset, FLAGS.attack, FLAGS.characteristics, FLAGS.ablation))
lr = train_lr(X_train, Y_train)

## Evaluate detector
y_pred       = lr.predict_proba(X_test)[:, 1]
y_label_pred = lr.predict(X_test)

# AUC
_, _, auc_score = compute_roc(Y_test, y_pred, plot=True)
precision = precision_score(Y_test, y_label_pred)
recall    = recall_score(Y_test, y_label_pred)
acc       = accuracy_score(Y_test, y_label_pred)
print('Detector ROC-AUC score: {}, accuracy: {}, precision: {}, recall: {}'.format(auc_score, acc, precision, recall))