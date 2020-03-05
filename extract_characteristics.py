from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import matplotlib
import platform
# Force matplotlib to not use any Xwindows backend.
if platform.system() == 'Linux':
    matplotlib.use('Agg')

import logging
import numpy as np
import tensorflow as tf
import os
import pickle
from tqdm import tqdm
from tensorflow.python.platform import flags
from NNIF_adv_defense.models.darkon_resnet34_model import DarkonReplica
from NNIF_adv_defense.datasets.influence_feeder import MyFeederValTest
from NNIF_adv_defense.tools.utils import mle_batch
import sklearn.covariance
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from cleverhans.evaluation import batch_eval
from cleverhans.utils import set_log_level
import time

# TODO(support noise in the future). In my/our settings the characteristics are calculated only for real/adv images.
STDEVS = {
    'val' : {'cifar10' : {'deepfool': 0.00861, 'cw': 0.003081, 'cw_nnif': 0.003081, 'jsma': 0.001, 'fgsm': 0.001, 'pgd': 0.001, 'ead': 0.001},
             'cifar100': {'deepfool': 0.001, 'cw': 0.001, 'cw_nnif': 0.003081, 'jsma': 0.001, 'fgsm': 0.001, 'pgd': 0.001, 'ead': 0.001},
             'svhn'    : {'deepfool': 0.001, 'cw': 0.001, 'cw_nnif': 0.003081, 'jsma': 0.001, 'fgsm': 0.001, 'pgd': 0.001, 'ead': 0.001}
            },
    'test': {'cifar10' : {'deepfool': 0.00796, 'cw': 0.003057, 'cw_nnif': 0.003081, 'jsma': 0.001, 'fgsm': 0.001, 'pgd': 0.001, 'ead': 0.001},
             'cifar100': {'deepfool': 0.001, 'cw': 0.001, 'cw_nnif': 0.003081, 'jsma': 0.001, 'fgsm': 0.001, 'pgd': 0.001, 'ead': 0.001},
             'svhn'    : {'deepfool': 0.001, 'cw': 0.001, 'cw_nnif': 0.003081, 'jsma': 0.001, 'fgsm': 0.001, 'pgd': 0.001, 'ead': 0.001}
             }
}


num_of_spatial_activations = {
    'layer0': 32 * 32, 'layer1': 32 * 32, 'layer2': 32 * 32, 'layer3': 32 * 32, 'layer4': 32 * 32, 'layer5': 32 * 32,
    'layer6': 32 * 32, 'layer7': 32 * 32, 'layer8': 32 * 32, 'layer9': 32 * 32, 'layer10': 32 * 32, 'layer11': 16 * 16,
    'layer12': 16 * 16, 'layer13': 16 * 16, 'layer14': 16 * 16, 'layer15': 16 * 16, 'layer16': 16 * 16, 'layer17': 16 * 16,
    'layer18': 16 * 16, 'layer19': 16 * 16, 'layer20': 16 * 16, 'layer21': 8 * 8, 'layer22': 8 * 8, 'layer23': 8 * 8,
    'layer24': 8 * 8, 'layer25': 8 * 8, 'layer26': 8 * 8, 'layer27': 8 * 8, 'layer28': 8 * 8, 'layer29': 8 * 8, 'layer30': 8 * 8
}

FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 125, 'Size of training batches')
flags.DEFINE_string('dataset', 'cifar10', 'dataset: cifar10/100 or svhn')
flags.DEFINE_string('attack', 'deepfool', 'adversarial attack: deepfool, jsma, cw, cw_nnif')
flags.DEFINE_string('characteristics', 'nnif', 'type of defence: lid/mahalanobis/dknn/nnif')
flags.DEFINE_bool('with_noise', False, 'whether or not to include noisy samples')
flags.DEFINE_bool('only_last', False, 'Using just the last layer, the embedding vector')
flags.DEFINE_string('checkpoint_dir', '', 'Checkpoint dir, the path to the saved model architecture and weights')

# FOR DkNN and LID
flags.DEFINE_integer('k_nearest', -1, 'number of nearest neighbors to use for LID/DkNN detection')

# FOR MAHANABOLIS
flags.DEFINE_float('magnitude', -1, 'magnitude for mahalanobis detection')

# FOR NNIF
flags.DEFINE_integer('max_indices', -1, 'maximum number of helpful indices to use in NNIF detection')
flags.DEFINE_string('ablation', '1111', 'for ablation test')

#TODO: remove when done debugging
flags.DEFINE_string('mode', 'null', 'to bypass pycharm bug')
flags.DEFINE_string('port', 'null', 'to bypass pycharm bug')

assert FLAGS.with_noise is False  # TODO(support noise in the future)
rgb_scale = 1.0  # Used for the Mahalanobis detection

if FLAGS.set == 'val':
    test_val_set = True  # evaluating on the validation set
    WORKSPACE = 'influence_workspace_validation'
    USE_TRAIN_MINI = False  # use all the training set examples in evaluation
else:
    test_val_set = False  # evaluating on the
    WORKSPACE = 'influence_workspace_test_mini'
    USE_TRAIN_MINI = True

TARGETED = FLAGS.attack != 'deepfool'  # we use targeted attacks everywhere except deepfool

_classes = {
    'cifar10': (
        'airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
    ),
    'cifar100': (
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
        'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
        'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
        'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
        'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
    ),
    'svhn': (
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
    )
}

ARCH_NAME = {'cifar10': 'model1', 'cifar100': 'model_cifar_100', 'svhn': 'model_svhn'}

# Set TF random seed to improve reproducibility
superseed = 123456789
rand_gen = np.random.RandomState(superseed)
tf.set_random_seed(superseed)

# Set logging level to see debug information
set_log_level(logging.DEBUG)

# Create TF session
config_args = dict(allow_soft_placement=True)
sess = tf.Session(config=tf.ConfigProto(**config_args))

# get records from training
if FLAGS.checkpoint_dir != '':
    model_dir     = FLAGS.checkpoint_dir                      # set user specified dir
else:
    model_dir = os.path.join(FLAGS.dataset, 'trained_model')  # set default dir

workspace_dir = os.path.join(model_dir, WORKSPACE)
attack_dir    = os.path.join(model_dir, FLAGS.attack)
if TARGETED:
    attack_dir = attack_dir + '_targeted'

characteristics_dir = os.path.join(attack_dir, FLAGS.characteristics)
if not os.path.exists(characteristics_dir):
    os.makedirs(characteristics_dir)

val_indices = np.load(os.path.join(model_dir, 'val_indices.npy'))

print('loading train mini indices from {}'.format(os.path.join(model_dir, 'train_mini_indices.npy')))
mini_train_inds = np.load(os.path.join(model_dir, 'train_mini_indices.npy'))
feeder = MyFeederValTest(dataset=FLAGS.dataset, rand_gen=rand_gen, as_one_hot=True, val_inds=val_indices,
                         test_val_set=False, mini_train_inds=mini_train_inds)

# get the dataset
X_train     , y_train      = feeder.train_data     , feeder.train_label        # real train set (49k):
X_train_mini, y_train_mini = feeder.mini_train_data, feeder.mini_train_label   # mini train set (just 5k)
X_val       , y_val        = feeder.val_data       , feeder.val_label          # val set (1k)
X_test      , y_test       = feeder.test_data      , feeder.test_label         # test set
y_train_sparse             = y_train.argmax(axis=-1).astype(np.int32)
y_train_mini_sparse        = y_train_mini.argmax(axis=-1).astype(np.int32)
y_val_sparse               = y_val.argmax(axis=-1).astype(np.int32)
y_test_sparse              = y_test.argmax(axis=-1).astype(np.int32)

# if the attack is targeted, fetch the targets
if TARGETED:
    y_val_targets  = np.load(os.path.join(attack_dir, 'y_val_targets.npy'))
    y_test_targets = np.load(os.path.join(attack_dir, 'y_test_targets.npy'))

# fetch the predictions and embedding vectors
x_train_preds         = np.load(os.path.join(model_dir, 'x_train_preds.npy'))
x_train_features      = np.load(os.path.join(model_dir, 'x_train_features.npy'))

x_train_mini_preds    = np.load(os.path.join(model_dir, 'x_train_mini_preds.npy'))
x_train_mini_features = np.load(os.path.join(model_dir, 'x_train_mini_features.npy'))

x_val_preds           = np.load(os.path.join(model_dir, 'x_val_preds.npy'))
x_val_features        = np.load(os.path.join(model_dir, 'x_val_features.npy'))

x_test_preds          = np.load(os.path.join(model_dir, 'x_test_preds.npy'))
x_test_features       = np.load(os.path.join(model_dir, 'x_test_features.npy'))

X_val_adv             = np.load(os.path.join(attack_dir, 'X_val_adv.npy'))
x_val_preds_adv       = np.load(os.path.join(attack_dir, 'x_val_preds_adv.npy'))
x_val_features_adv    = np.load(os.path.join(attack_dir, 'x_val_features_adv.npy'))

X_test_adv            = np.load(os.path.join(attack_dir, 'X_test_adv.npy'))
x_test_preds_adv      = np.load(os.path.join(attack_dir, 'x_test_preds_adv.npy'))
x_test_features_adv   = np.load(os.path.join(attack_dir, 'x_test_features_adv.npy'))

# quick computations of accuracies
train_acc    = np.mean(y_train_sparse == x_train_preds)
val_acc      = np.mean(y_val_sparse   == x_val_preds)
test_acc     = np.mean(y_test_sparse  == x_test_preds)
val_adv_acc  = np.mean(y_val_sparse   == x_val_preds_adv)
test_adv_acc = np.mean(y_test_sparse  == x_test_preds_adv)
print('train set acc: {}\nvalidation set acc: {}\ntest set acc: {}'.format(train_acc, val_acc, test_acc))
print('adversarial ({}) validation set acc: {}\nadversarial ({}) test set acc: {}'.format(FLAGS.attack, val_adv_acc, FLAGS.attack, test_adv_acc))

# what are the indices of the cifar10 set which the network succeeded classifying correctly,
# but the adversarial attack changed to a different class?
info = {}
info['val'] = {}
for i, set_ind in enumerate(feeder.val_inds):
    info['val'][i] = {}
    net_succ    = x_val_preds[i] == y_val_sparse[i]
    attack_succ = x_val_preds[i] != x_val_preds_adv[i]
    info['val'][i]['global_index'] = set_ind
    info['val'][i]['net_succ']     = net_succ
    info['val'][i]['attack_succ']  = attack_succ
info['test'] = {}
for i, set_ind in enumerate(feeder.test_inds):
    info['test'][i] = {}
    net_succ    = x_test_preds[i] == y_test_sparse[i]
    attack_succ = x_test_preds[i] != x_test_preds_adv[i]
    info['test'][i]['global_index'] = set_ind
    info['test'][i]['net_succ']     = net_succ
    info['test'][i]['attack_succ']  = attack_succ

info_file = os.path.join(attack_dir, 'info.pkl')
print('loading info as pickle from {}'.format(info_file))
with open(info_file, 'rb') as handle:
    info_old = pickle.load(handle)
assert info == info_old

# Use Image Parameters
img_rows, img_cols, nchannels = X_test.shape[1:4]
nb_classes = y_test.shape[1]

# Define input TF placeholder
x     = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, nchannels), name='x')
y     = tf.placeholder(tf.float32, shape=(None, nb_classes), name='y')

model = DarkonReplica(scope=ARCH_NAME, nb_classes=feeder.num_classes, n=5, input_shape=[32, 32, 3])
preds      = model.get_predicted_class(x)
logits     = model.get_logits(x)
embeddings = model.get_embeddings(x)

# loading the checkpoint
saver = tf.train.Saver()
checkpoint_path = os.path.join(model_dir, 'best_model.ckpt')
saver.restore(sess, checkpoint_path)

# get noisy images
def get_noisy_samples(X, std):
    """ Add Gaussian noise to the samples """
    # std = STDEVS[subset][FLAGS.dataset][FLAGS.attack]
    X_noisy = np.clip(X + rand_gen.normal(loc=0.0, scale=std, size=X.shape), 0, 1)
    return X_noisy

# DEBUG: testing different scale so that L2 perturbation is the same
# diff_adv    = X_val_adv.reshape((len(X_val), -1)) - X_val.reshape((len(X_val), -1))
# l2_diff_adv = np.linalg.norm(diff_adv, axis=1).mean()
# for std in np.arange(0.0082, 0.0089, 0.00001):
#     X_val_noisy = get_noisy_samples(X_val, std)
#     diff = X_val_noisy.reshape((len(X_val), -1)) - X_val.reshape((len(X_val), -1))
#     l2_diff = np.linalg.norm(diff, axis=1).mean()
#     print('for std={}: diff of L2 perturbations is {}'.format(std, l2_diff - l2_diff_adv))
#
# diff_adv    = X_test_adv.reshape((len(X_test), -1)) - X_test.reshape((len(X_test), -1))
# l2_diff_adv = np.linalg.norm(diff_adv, axis=1).mean()
# for std in np.arange(0.003, 0.004, 0.0001):
#     X_test_noisy = get_noisy_samples(X_test, std)
#     diff = X_test_noisy.reshape((len(X_test), -1)) - X_test.reshape((len(X_test), -1))
#     l2_diff = np.linalg.norm(diff, axis=1).mean()
#     print('for std={}: diff of L2 perturbations is {}'.format(std, l2_diff - l2_diff_adv))

noisy_file = os.path.join(attack_dir, 'X_val_noisy.npy')
if os.path.isfile(noisy_file):
    print('Loading {} val noisy samples from {}'.format(FLAGS.dataset, noisy_file))
    X_val_noisy = np.load(noisy_file)
else:
    print('Crafting {} val noisy samples.'.format(FLAGS.dataset))
    X_val_noisy = get_noisy_samples(X_val, std=STDEVS['val'][FLAGS.dataset][FLAGS.attack])
    np.save(noisy_file, X_val_noisy)

noisy_file = os.path.join(attack_dir, 'X_test_noisy.npy')
if os.path.isfile(noisy_file):
    print('Loading {} noisy samples from {}'.format(FLAGS.dataset, noisy_file))
    X_test_noisy = np.load(noisy_file)
else:
    print('Crafting {} test noisy samples.'.format(FLAGS.dataset))
    X_test_noisy = get_noisy_samples(X_test, std=STDEVS['test'][FLAGS.dataset][FLAGS.attack])
    np.save(noisy_file, X_test_noisy)

# print stats for val
for s_type, subset in zip(['normal', 'noisy', 'adversarial'], [X_val, X_val_noisy, X_val_adv]):
    # acc = model_eval(sess, x, y, logits, subset, y_val, args=eval_params)
    # print("Model accuracy on the %s val set: %0.2f%%" % (s_type, 100 * acc))
    # Compute and display average perturbation sizes
    if not s_type == 'normal':
        # print for test:
        diff    = subset.reshape((len(subset), -1)) - X_val.reshape((len(subset), -1))
        l2_diff = np.linalg.norm(diff, axis=1).mean()
        print("Average L-2 perturbation size of the %s val set: %0.4f" % (s_type, l2_diff))

# print stats for test
for s_type, subset in zip(['normal', 'noisy', 'adversarial'], [X_test, X_test_noisy, X_test_adv]):
    # acc = model_eval(sess, x, y, logits, subset, y_test, args=eval_params)
    # print("Model accuracy on the %s test set: %0.2f%%" % (s_type, 100 * acc))
    # Compute and display average perturbation sizes
    if not s_type == 'normal':
        # print for test:
        diff    = subset.reshape((len(subset), -1)) - X_test.reshape((len(subset), -1))
        l2_diff = np.linalg.norm(diff, axis=1).mean()
        print("Average L-2 perturbation size of the %s test set: %0.4f" % (s_type, l2_diff))

# Refine the normal, noisy and adversarial sets to only include samples for
# which the original version was correctly classified by the model
val_inds_correct  = np.where(x_val_preds == y_val_sparse)[0]
print("Number of correctly val predict images: %s" % (len(val_inds_correct)))
X_val              = X_val[val_inds_correct]
X_val_noisy        = X_val_noisy[val_inds_correct]
X_val_adv          = X_val_adv[val_inds_correct]
x_val_preds        = x_val_preds[val_inds_correct]
x_val_features     = x_val_features[val_inds_correct]
x_val_preds_adv    = x_val_preds_adv[val_inds_correct]
x_val_features_adv = x_val_features_adv[val_inds_correct]
y_val              = y_val[val_inds_correct]
y_val_sparse       = y_val_sparse[val_inds_correct]

test_inds_correct = np.where(x_test_preds == y_test_sparse)[0]
print("Number of correctly test predict images: %s" % (len(test_inds_correct)))
X_test              = X_test[test_inds_correct]
X_test_noisy        = X_test_noisy[test_inds_correct]
X_test_adv          = X_test_adv[test_inds_correct]
x_test_preds        = x_test_preds[test_inds_correct]
x_test_features     = x_test_features[test_inds_correct]
x_test_preds_adv    = x_test_preds_adv[test_inds_correct]
x_test_features_adv = x_test_features_adv[test_inds_correct]
y_test              = y_test[test_inds_correct]
y_test_sparse       = y_test_sparse[test_inds_correct]

print("X_val: "       , X_val.shape)
print("X_val_noisy: " , X_val_noisy.shape)
print("X_val_adv: "   , X_val_adv.shape)

print("X_test: "      , X_test.shape)
print("X_test_noisy: ", X_test_noisy.shape)
print("X_test_adv: "  , X_test_adv.shape)

# if only last, make sure that only the embedding is in model.net
if FLAGS.only_last:
    print('Keeping only the embedding layer in model.net')
    model.net = {'layer31': model.net['layer31']}
    assert embeddings is model.net['layer31']

def merge_and_generate_labels(X_pos, X_neg):
    """
    merge positve and nagative artifact and generate labels
    :param X_pos: positive samples
    :param X_neg: negative samples
    :return: X: merged samples, 2D ndarray
             y: generated labels (0/1): 2D ndarray same size as X
    """
    X_pos = np.asarray(X_pos, dtype=np.float32)
    print("X_pos: ", X_pos.shape)
    X_pos = X_pos.reshape((X_pos.shape[0], -1))

    X_neg = np.asarray(X_neg, dtype=np.float32)
    print("X_neg: ", X_neg.shape)
    X_neg = X_neg.reshape((X_neg.shape[0], -1))

    X = np.concatenate((X_pos, X_neg))
    y = np.concatenate((np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0])))
    y = y.reshape((X.shape[0], 1))

    return X, y

def get_lids_random_batch(X_test, X_test_noisy, X_test_adv, k=FLAGS.k_nearest, batch_size=100):
    """
    :param X_test: normal images
    :param X_test_noisy: noisy images
    :param X_test_adv: advserial images
    :param k: the number of nearest neighbours for LID estimation
    :param batch_size: default 100
    :return: lids: LID of normal images of shape (num_examples, lid_dim)
            lids_adv: LID of advs images of shape (num_examples, lid_dim)
    """

    lid_dim = len(model.net)
    print("Number of layers to estimate: ", lid_dim)

    def estimate(i_batch):
        start = i_batch * batch_size
        end = np.minimum(len(X_test), (i_batch + 1) * batch_size)
        n_feed = end - start
        lid_batch       = np.zeros(shape=(n_feed, lid_dim))
        lid_batch_adv   = np.zeros(shape=(n_feed, lid_dim))
        lid_batch_noisy = np.zeros(shape=(n_feed, lid_dim))

        X_act       = batch_eval(sess, [x], model.net.values(), [X_test[start:end]]      , batch_size)
        X_adv_act   = batch_eval(sess, [x], model.net.values(), [X_test_adv[start:end]]  , batch_size)
        X_noisy_act = batch_eval(sess, [x], model.net.values(), [X_test_noisy[start:end]], batch_size)

        for i in range(len(model.net)):
            X_act[i]       = np.asarray(X_act[i]      , dtype=np.float32).reshape((n_feed, -1))
            X_adv_act[i]   = np.asarray(X_adv_act[i]  , dtype=np.float32).reshape((n_feed, -1))
            X_noisy_act[i] = np.asarray(X_noisy_act[i], dtype=np.float32).reshape((n_feed, -1))

            # random clean samples
            # Maximum likelihood estimation of local intrinsic dimensionality (LID)
            lid_batch[:, i]       = mle_batch(X_act[i], X_act[i]      , k=k)
            lid_batch_adv[:, i]   = mle_batch(X_act[i], X_adv_act[i]  , k=k)
            lid_batch_noisy[:, i] = mle_batch(X_act[i], X_noisy_act[i], k=k)

        return lid_batch, lid_batch_noisy, lid_batch_adv

    lids = []
    lids_adv = []
    lids_noisy = []
    n_batches = int(np.ceil(X_test.shape[0] / float(batch_size)))
    for i_batch in tqdm(range(n_batches)):
        lid_batch, lid_batch_noisy, lid_batch_adv = estimate(i_batch)
        lids.extend(lid_batch)
        lids_adv.extend(lid_batch_adv)
        lids_noisy.extend(lid_batch_noisy)

    lids       = np.asarray(lids, dtype=np.float32)
    lids_noisy = np.asarray(lids_noisy, dtype=np.float32)
    lids_adv   = np.asarray(lids_adv, dtype=np.float32)

    return lids, lids_noisy, lids_adv

def get_lid(X, X_noisy, X_adv, k, batch_size=100):
    print('Extract local intrinsic dimensionality: k = %s' % k)
    lids_normal, lids_noisy, lids_adv = get_lids_random_batch(X, X_noisy, X_adv, k, batch_size)
    print("lids_normal:", lids_normal.shape)
    print("lids_noisy:", lids_noisy.shape)
    print("lids_adv:", lids_adv.shape)

    lids_pos = lids_adv
    if FLAGS.with_noise:
        lids_neg = np.concatenate((lids_normal, lids_noisy))
    else:
        lids_neg = lids_normal
    artifacts, labels = merge_and_generate_labels(lids_pos, lids_neg)

    return artifacts, labels

def get_mahalanobis(X, X_noisy, X_adv, magnitude, sample_mean, precision, set):
    first_pass = True
    for layer in model.net.keys():
        print('Calculating Mahalanobis characteristics for set {}, {}'.format(set, layer))
        with tf.name_scope('{}_gaussian_{}'.format(set, layer)):
            gaussian_score, grads = get_mahanabolis_tensors(sample_mean, precision, feeder.num_classes, layer)

            # val
            M_in = get_Mahalanobis_score_adv(X, gaussian_score, grads, magnitude, rgb_scale)
            M_in = np.asarray(M_in, dtype=np.float32)

            M_out = get_Mahalanobis_score_adv(X_adv, gaussian_score, grads, magnitude, rgb_scale)
            M_out = np.asarray(M_out, dtype=np.float32)

            if FLAGS.with_noise:
                M_noisy = get_Mahalanobis_score_adv(X_noisy, gaussian_score, grads, magnitude, rgb_scale)
                M_noisy = np.asarray(M_noisy, dtype=np.float32)
            else:  # just a placeholder with zeros
                M_noisy = np.zeros_like(M_in)

            if first_pass:
                Mahalanobis_in    = M_in.reshape((M_in.shape[0], -1))
                Mahalanobis_out   = M_out.reshape((M_out.shape[0], -1))
                Mahalanobis_noisy = M_noisy.reshape((M_noisy.shape[0], -1))
                first_pass = False
            else:
                Mahalanobis_in    = np.concatenate((Mahalanobis_in, M_in.reshape((M_in.shape[0], -1))), axis=1)
                Mahalanobis_out   = np.concatenate((Mahalanobis_out, M_out.reshape((M_out.shape[0], -1))), axis=1)
                Mahalanobis_noisy = np.concatenate((Mahalanobis_noisy, M_noisy.reshape((M_noisy.shape[0], -1))), axis=1)

    if FLAGS.with_noise:
        Mahalanobis_neg = np.concatenate((Mahalanobis_in, Mahalanobis_noisy))
    else:
        Mahalanobis_neg = Mahalanobis_in
    Mahalanobis_pos = Mahalanobis_out
    characteristics, labels = merge_and_generate_labels(Mahalanobis_pos, Mahalanobis_neg)

    return characteristics, labels

def sample_estimator(num_classes, X, Y):
    num_output           = len(model.net)
    feature_list         = np.zeros(num_output, dtype=np.int32)   # indicates the number of features in every layer
    num_sample_per_class = np.zeros(num_classes)  # how many samples are per class
    for i, key in enumerate(model.net):
        feature_list[i] = model.net[key].shape[-1].value
    assert (feature_list > 0).all()

    list_features = []  # list_features[<layer>][<label>] is a list that holds the features in a specific layer of a specific label
                        # is it basically list_features[<num_of_layer>][<num_of_label>] = List
    for i in range(num_output):
        temp_list = []
        for j in range(num_classes):
            temp_list.append([])
        list_features.append(temp_list)

    out_features = batch_eval(sess, [x], model.net.values(), [X], FLAGS.batch_size)
    for i in range(num_output):
        if len(out_features[i].shape) == 4:
            out_features[i] = np.asarray(out_features[i], dtype=np.float32).reshape((X.shape[0], -1, out_features[i].shape[-1]))
            out_features[i] = np.mean(out_features[i], axis=1)
        elif len(out_features[i].shape) == 2:
            pass  # leave as is
        else:
            raise AssertionError('Expecting size of 2 or 4 but got {} for i={}'.format(len(out_features[i].shape), i))

    for i in range(X.shape[0]):
        label = Y[i]
        for layer in range(num_output):
            list_features_temp = out_features[layer][i].reshape(1, -1)
            list_features[layer][label].extend(list_features_temp)
        num_sample_per_class[label] += 1

    # stacking everything
    for layer in range(num_output):
        for label in range(num_classes):
            list_features[layer][label] = np.stack(list_features[layer][label])

    sample_class_mean = []
    for layer in range(num_output):
        num_feature = feature_list[layer]
        temp_list = np.zeros((num_classes, num_feature))
        for i in range(num_classes):
            temp_list[i] = np.mean(list_features[layer][i], axis=0)
        sample_class_mean.append(temp_list)

    precision = []
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    for layer in range(num_output):
        D = 0
        for i in range(num_classes):
            if i == 0:
                D = list_features[layer][i] - sample_class_mean[layer][i]
            else:
                D = np.concatenate((D, list_features[layer][i] - sample_class_mean[layer][i]), 0)

        # find inverse
        group_lasso.fit(D)
        temp_precision = group_lasso.precision_
        precision.append(temp_precision)

    return sample_class_mean, precision

def get_Mahalanobis_score_adv(test_data, gaussian_score, grads, magnitude, scale):
    grad_file = os.path.join(characteristics_dir, 'gradients_{}.npy'.format(set))
    # if os.path.exists(grad_file):
    #     print('loading gradients from {}'.format(grad_file))
    #     gradients = np.load(grad_file)
    # else:
    gradients = batch_eval(sess, [x], grads, [test_data], FLAGS.batch_size)[0]
    # print('Saving gradients to {}'.format(grad_file))
    # np.save(grad_file, gradients)

    gradients = gradients.clip(min=0)
    gradients = (gradients - 0.5) * 2

    # scale hyper params given from the official deep_Mahalanobis_detector repo:
    # https://github.com/pokaxpoka/deep_Mahalanobis_detector
    # I/We set scale=1 by default, with their params
    RED_SCALE   = 0.2023 * scale
    GREEN_SCALE = 0.1994 * scale
    BLUE_SCALE  = 0.2010 * scale

    gradients_scaled = np.zeros_like(gradients)
    gradients_scaled[:, :, :, 0] = gradients[:, :, :, 0] / RED_SCALE
    gradients_scaled[:, :, :, 1] = gradients[:, :, :, 1] / GREEN_SCALE
    gradients_scaled[:, :, :, 2] = gradients[:, :, :, 2] / BLUE_SCALE

    tempInputs = test_data - magnitude * gradients_scaled
    noise_gaussian_score = batch_eval(sess, [x], [gaussian_score], [tempInputs], FLAGS.batch_size)[0]

    Mahalanobis = np.max(noise_gaussian_score, axis=1)

    return Mahalanobis

def get_mahanabolis_tensors(sample_mean, precision, num_classes, layer):
    # here we calculate the input gradients for -pure_tau. Meaning d(-pure_tau)/dx.
    # First, how do we calculate pure_tau? This is a computation on a batch.
    layer_index = int(layer[5:])

    with tf.name_scope('Mahanabolis_grad_calc_'.format(layer)):
        if FLAGS.only_last:
            precision_mat      = tf.convert_to_tensor(precision[0]  , dtype=tf.float32)
            sample_mean_tensor = tf.convert_to_tensor(sample_mean[0], dtype=tf.float32)
        else:
            precision_mat      = tf.convert_to_tensor(precision[layer_index]    , dtype=tf.float32)
            sample_mean_tensor = tf.convert_to_tensor(sample_mean[layer_index]  , dtype=tf.float32)

        out_features       = model.net[layer]
        if len(out_features.shape) == 4:
            num_spatial = num_of_spatial_activations[layer]
            out_features = tf.reshape(out_features, [-1, num_spatial, out_features.shape[-1].value])
            out_features = tf.reduce_mean(out_features, axis=1)
        elif len(out_features.shape) == 2:
            pass  # leave as is
        else:
            raise AssertionError('Expecting size of 2 or 4 but got {} for {}'.format(len(out_features.shape), layer))

        for i in range(num_classes):
            batch_sample_mean = sample_mean_tensor[i]
            zero_f = out_features - batch_sample_mean
            zero_f_T = tf.transpose(zero_f)
            term_gau = -0.5 * tf.matmul(tf.matmul(zero_f, precision_mat), zero_f_T)
            term_gau = tf.diag_part(term_gau)
            if i == 0:
                gaussian_score = tf.reshape(term_gau, (-1, 1))
            else:
                gaussian_score_tmp = tf.reshape(term_gau, (-1, 1))
                gaussian_score = tf.concat([gaussian_score, gaussian_score_tmp], axis=1)

        # Input_processing
        sample_pred = tf.argmax(gaussian_score, axis=1)
        batch_sample_mean = tf.gather(sample_mean_tensor, axis=0, indices=sample_pred)
        zero_f = out_features - tf.identity(batch_sample_mean)
        zero_f_T = tf.transpose(zero_f)
        pure_gau = -0.5 * tf.matmul(tf.matmul(zero_f, tf.identity(precision_mat)), zero_f_T)  # 100x100
        pure_gau = tf.diag_part(pure_gau)  # 100
        gau_loss = tf.reduce_mean(-pure_gau)
        grads = tf.gradients(gau_loss, x)

    return gaussian_score, grads

def find_ranks(sub_index, sorted_influence_indices, adversarial=False):

    if adversarial:
        ni = all_adv_ranks
        nd = all_adv_dists
    else:
        ni = all_normal_ranks
        nd = all_normal_dists

    num_output = len(model.net)
    ranks = -1 * np.ones((num_output, len(sorted_influence_indices)), dtype=np.int32)
    dists = -1 * np.ones((num_output, len(sorted_influence_indices)), dtype=np.float32)

    print('Finding ranks for sub_index={} (adversarial={})'.format(sub_index, adversarial))
    for target_idx in range(len(sorted_influence_indices)):  # for only some indices (say, 0:50 only)
        idx = sorted_influence_indices[target_idx]  # selecting training sample index
        for layer_index in range(num_output):
            loc_in_knn = np.where(ni[sub_index, layer_index] == idx)[0][0]
            knn_dist   = nd[sub_index, layer_index, loc_in_knn]
            ranks[layer_index, target_idx] = loc_in_knn
            dists[layer_index, target_idx] = knn_dist

    ranks_mean = np.mean(ranks, axis=1)
    dists_mean = np.mean(dists, axis=1)

    return ranks_mean, dists_mean

def get_nnif(X, subset, max_indices):
    """Returns the knn rank of every testing sample"""
    if subset == 'val':
        inds_correct = val_inds_correct
        y_sparse     = y_val_sparse
        x_preds      = x_val_preds
        x_preds_adv  = x_val_preds_adv
    else:
        inds_correct = test_inds_correct
        y_sparse     = y_test_sparse
        x_preds      = x_test_preds
        x_preds_adv  = x_test_preds_adv
    inds_correct = feeder.get_global_index(subset, inds_correct)

    # initialize knn for layers
    num_output = len(model.net)

    ranks     = -1 * np.ones((len(X), num_output, 4))
    ranks_adv = -1 * np.ones((len(X), num_output, 4))

    for i in tqdm(range(len(inds_correct))):
        global_index = inds_correct[i]
        real_label = y_sparse[i]
        pred_label = x_preds[i]
        adv_label  = x_preds_adv[i]
        assert pred_label == real_label, 'failed for i={}, global_index={}'.format(i, global_index)
        index_dir = os.path.join(model_dir, subset, '{}_index_{}'.format(subset, global_index))

        # collect pred scores:
        scores = np.load(os.path.join(index_dir, 'real', 'scores.npy'))
        sorted_indices = np.argsort(scores)
        ranks[i, :, 0], ranks[i, :, 1] = find_ranks(i, sorted_indices[-max_indices:][::-1], adversarial=False)
        ranks[i, :, 2], ranks[i, :, 3] = find_ranks(i, sorted_indices[:max_indices], adversarial=False)

        # collect adv scores:
        scores = np.load(os.path.join(index_dir, 'adv', FLAGS.attack, 'scores.npy'))
        sorted_indices = np.argsort(scores)
        ranks_adv[i, :, 0], ranks_adv[i, :, 1] = find_ranks(i, sorted_indices[-max_indices:][::-1], adversarial=True)
        ranks_adv[i, :, 2], ranks_adv[i, :, 3] = find_ranks(i, sorted_indices[:max_indices], adversarial=True)

    print("{} ranks_normal: ".format(subset), ranks.shape)
    print("{} ranks_adv: ".format(subset), ranks_adv.shape)
    assert (ranks     != -1).all()
    assert (ranks_adv != -1).all()

    return ranks, ranks_adv

def get_calibration(x_cal_features, y_cal, k):
    knn = KNeighborsClassifier(n_neighbors=k, p=2, n_jobs=20)
    knn.fit(x_train_features, y_train_sparse)

    knn_predict_prob = knn.predict_proba(x_cal_features)
    knn_pred_cnt = np.asarray(knn_predict_prob * k, dtype=np.int32)

    # how many wrong predictions do we have for the true label?
    calibration_vec = np.zeros(x_cal_features.shape[0])
    for i in range(x_cal_features.shape[0]):
        label = y_cal[i]
        calibration_vec[i] = k - knn_pred_cnt[i, label]

    return calibration_vec

def get_dknn_nonconformity(features, calibration_vec, k):
    knn = KNeighborsClassifier(n_neighbors=k, p=2, n_jobs=20)
    knn.fit(x_train_features, y_train_sparse)

    knn_predict_prob = knn.predict_proba(features)
    knn_pred_cnt = np.asarray(knn_predict_prob * k, dtype=np.int32)

    # how many wrong predictions do we have for each label?
    nonconformity = k - knn_pred_cnt

    # get pj for every class
    empirical_p = np.zeros_like(nonconformity, dtype=np.float32)
    for i in range(len(nonconformity)):  # for every sample
        for j in range(feeder.num_classes):  # for every class
            num_of_greater_calib_values = np.sum(calibration_vec >= nonconformity[i, j])
            empirical_p[i, j] = num_of_greater_calib_values / len(calibration_vec)

    return empirical_p

def get_knn_layers(X, y):
    knn = {}

    train_features = batch_eval(sess, [x], model.net.values(), [X], FLAGS.batch_size)
    print('Fitting knn models on all layers: {}'.format(model.net.keys()))
    for layer_index, layer in enumerate(model.net.keys()):
        if len(train_features[layer_index].shape) == 4:
            train_features[layer_index] = np.asarray(train_features[layer_index], dtype=np.float32).reshape((X.shape[0], -1, train_features[layer_index].shape[-1]))
            train_features[layer_index] = np.mean(train_features[layer_index], axis=1)
        elif len(train_features[layer_index].shape) == 2:
            pass  # leave as is
        else:
            raise AssertionError('Expecting size of 2 or 4 but got {} for {}'.format(len(train_features[layer_index].shape), layer))

        knn[layer] = NearestNeighbors(n_neighbors=X.shape[0], p=2, n_jobs=20, algorithm='brute')
        knn[layer].fit(train_features[layer_index], y)

    del train_features
    return knn

def calc_all_ranks_and_dists(X, subset, knn):
    num_output = len(model.net.keys())
    n_neighbors = knn[knn.keys()[0]].n_neighbors
    all_neighbor_ranks = -1 * np.ones((len(X), num_output, n_neighbors), dtype=np.int32)
    all_neighbor_dists = -1 * np.ones((len(X), num_output, n_neighbors), dtype=np.float32)

    features = batch_eval(sess, [x], model.net.values(), [X], FLAGS.batch_size)
    for layer_index, layer in enumerate(model.net.keys()):
        print('Calculating ranks and distances for subset {} for layer {}'.format(subset, layer))
        if len(features[layer_index].shape) == 4:
            features[layer_index] = np.asarray(features[layer_index], dtype=np.float32).reshape((X.shape[0], -1, features[layer_index].shape[-1]))
            features[layer_index] = np.mean(features[layer_index], axis=1)
        elif len(features[layer_index].shape) == 2:
            pass  # leave as is
        else:
            raise AssertionError('Expecting size of 2 or 4 but got {} for {}'.format(len(features[layer_index].shape), layer))

        all_neighbor_dists[:, layer_index], all_neighbor_ranks[:, layer_index] = \
            knn[layer].kneighbors(features[layer_index], return_distance=True)

    del features
    return all_neighbor_ranks, all_neighbor_dists

def append_suffix(f):
    # if with_noisy:
    #     f = f + '_noisy_{}'.format(FLAGS.with_noise)  # TODO(remove in the future. For backward compatibility)
    if FLAGS.noisy:
        f = f + '_noisy'
    if FLAGS.only_last:
        f = f + '_only_last'
    f = f + '.npy'
    return f


start = time.time()

if FLAGS.characteristics == 'lid':

    if FLAGS.k_nearest == -1:
        k_vec = np.arange(10, 41, 2)
    else:
        k_vec = [FLAGS.k_nearest]

    for k in tqdm(k_vec):
        print('Extracting LID characteristics for k={}'.format(k))
        # for val set
        characteristics, label = get_lid(X_val, X_val_noisy, X_val_adv, k, 100)
        print("LID train: [characteristic shape: ", characteristics.shape, ", label shape: ", label.shape)

        file_name = 'k_{}_batch_{}_{}'.format(k, 100, 'train')
        file_name = append_suffix(file_name)
        file_name = os.path.join(characteristics_dir, file_name)
        data = np.concatenate((characteristics, label), axis=1)
        np.save(file_name, data)
        end_val = time.time()
        print('total feature extraction time for val: {} sec'.format(end_val - start))

        # for test set
        characteristics, labels = get_lid(X_test, X_test_noisy, X_test_adv, k, 100)
        file_name = 'k_{}_batch_{}_{}'.format(k, 100, 'test')
        file_name = append_suffix(file_name)
        file_name = os.path.join(characteristics_dir, file_name)
        data = np.concatenate((characteristics, labels), axis=1)
        np.save(file_name, data)
        end_test = time.time()
        print('total feature extraction time for test: {} sec'.format(end_test - end_val))

if FLAGS.characteristics == 'nnif':
    # assert FLAGS.only_last is True

    # for ablation:
    sel_column = []
    for i in [0, 1, 2, 3]:
        if FLAGS.ablation[i] == '1':
            sel_column.append(i)

    if FLAGS.max_indices == -1:
        max_indices_vec = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    else:
        max_indices_vec = [FLAGS.max_indices]

    for max_indices in tqdm(max_indices_vec):
        print('Extracting NNIF characteristics for max_indices={}'.format(max_indices))

        # training the knn layers
        knn_large_trainset = get_knn_layers(X_train, y_train_sparse)
        knn_small_trainset = get_knn_layers(X_train_mini, y_train_mini_sparse)

        # val
        all_normal_ranks, all_normal_dists = calc_all_ranks_and_dists(X_val, 'val', knn_large_trainset)
        all_adv_ranks   , all_adv_dists    = calc_all_ranks_and_dists(X_val_adv, 'val', knn_large_trainset)
        ranks, ranks_adv = get_nnif(X_val, 'val', max_indices)
        ranks     = ranks[:, :, sel_column]
        ranks_adv = ranks_adv[:, :, sel_column]
        characteristics, labels = merge_and_generate_labels(ranks_adv, ranks)
        print("NNIF train: [characteristic shape: ", characteristics.shape, ", label shape: ", labels.shape)
        file_name = 'max_indices_{}_ablation_{}_train'.format(max_indices, FLAGS.ablation)
        file_name = append_suffix(file_name)
        file_name = os.path.join(characteristics_dir, file_name)
        data = np.concatenate((characteristics, labels), axis=1)
        np.save(file_name, data)
        end_val = time.time()
        print('total feature extraction time for val: {} sec'.format(end_val - start))

        # test
        all_normal_ranks, all_normal_dists = calc_all_ranks_and_dists(X_test, 'test', knn_small_trainset)
        all_adv_ranks   , all_adv_dists    = calc_all_ranks_and_dists(X_test_adv, 'test', knn_small_trainset)
        ranks, ranks_adv = get_nnif(X_test, 'test', max_indices)
        ranks[:, :, 0] *= (49/5)  # The mini train set contains only 5k images, not 49k images as in the train set
        ranks[:, :, 2] *= (49/5)  # Therefore, the ranks (both helpful and harmful) are scaled.
        ranks_adv[:, :, 0] *= (49/5)
        ranks_adv[:, :, 2] *= (49/5)
        ranks     = ranks[:, :, sel_column]
        ranks_adv = ranks_adv[:, :, sel_column]
        characteristics, labels = merge_and_generate_labels(ranks_adv, ranks)
        print("NNIF test: [characteristic shape: ", characteristics.shape, ", label shape: ", labels.shape)
        file_name = 'max_indices_{}_ablation_{}_test'.format(max_indices, FLAGS.ablation)
        file_name = append_suffix(file_name)
        file_name = os.path.join(characteristics_dir, file_name)
        data = np.concatenate((characteristics, labels), axis=1)
        np.save(file_name, data)
        end_test = time.time()
        print('total feature extraction time for test: {} sec'.format(end_test - end_val))

if FLAGS.characteristics == 'mahalanobis':

    print('get sample mean and covariance of the training set...')  # included in val (non-deployment) computation time
    sample_mean, precision = sample_estimator(feeder.num_classes, X_train, y_train_sparse)
    print('Done calculating: sample_mean, precision.')

    if FLAGS.magnitude == -1:
        magnitude_vec = np.array([0.00001, 0.00002, 0.00005, 0.00008, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.008, 0.01])
    else:
        magnitude_vec = [FLAGS.magnitude]

    for magnitude in tqdm(magnitude_vec):
        print('Extracting Mahalanobis characteristics for magnitude={}'.format(magnitude))

        # for val set
        characteristics, label = get_mahalanobis(X_val, X_val_noisy, X_val_adv, magnitude, sample_mean, precision, 'train')
        print("Mahalanobis train: [characteristic shape: ", characteristics.shape, ", label shape: ", label.shape)
        file_name = 'magnitude_{}_scale_{}_{}'.format(magnitude, rgb_scale, 'train')
        file_name = append_suffix(file_name)
        file_name = os.path.join(characteristics_dir, file_name)
        data = np.concatenate((characteristics, label), axis=1)
        np.save(file_name, data)
        end_val = time.time()
        print('total feature extraction time for val: {} sec'.format(end_val - start))

        # for test set
        characteristics, labels = get_mahalanobis(X_test, X_test_noisy, X_test_adv, magnitude, sample_mean, precision, 'test')
        file_name = 'magnitude_{}_scale_{}_{}'.format(magnitude, rgb_scale, 'test')
        file_name = append_suffix(file_name)
        file_name = os.path.join(characteristics_dir, file_name)
        data = np.concatenate((characteristics, labels), axis=1)
        np.save(file_name, data)
        end_test = time.time()
        print('total feature extraction time for test: {} sec'.format(end_test - end_val))

if FLAGS.characteristics == 'dknn':
    assert FLAGS.only_last is True

    if FLAGS.k_nearest == -1:
        if FLAGS.dataset == 'cifar10':
            k_vec = np.arange(4000, 5600, 100)
        elif FLAGS.dataset == 'cifar100':
            k_vec = np.arange(10, 510, 10)
        elif FLAGS.dataset == 'svhn':
            k_vec = np.arange(1000, 5100, 200)
    else:
        k_vec = [FLAGS.k_nearest]

    for k in tqdm(k_vec):
        print('Extracting DkNN characteristics for k={}'.format(k))
        # divide the validation set for calibration and alphas
        calibration_size = int(X_val.shape[0]/3)

        X_cal          = X_val[:calibration_size]
        x_cal_features = x_val_features[:calibration_size]
        y_cal          = y_val_sparse[:calibration_size]

        print("Calculating the calibration matrix...")
        calibration_vec = get_calibration(x_cal_features, y_cal, k)  # included in val (non-deployment) computation time
        print("Done calculating the calibration matrix.")

        X_val2              = X_val[calibration_size:]
        y_val2              = y_val_sparse[calibration_size:]
        x_val2_features     = x_val_features[calibration_size:]

        X_val2_adv          = X_val_adv[calibration_size:]
        y_val2_adv          = x_val_preds_adv[calibration_size:]
        x_val2_features_adv = x_val_features_adv[calibration_size:]

        # set training set
        val_normal_characteristics = get_dknn_nonconformity(x_val2_features, calibration_vec, k)
        val_adv_characteristics    = get_dknn_nonconformity(x_val2_features_adv, calibration_vec, k)

        dknn_neg = val_normal_characteristics
        dknn_pos = val_adv_characteristics
        characteristics, labels = merge_and_generate_labels(dknn_pos, dknn_neg)

        print("DKNN train: [characteristic shape: ", characteristics.shape, ", label shape: ", labels.shape)
        file_name = 'k_{}_{}'.format(k, 'train')
        file_name = append_suffix(file_name)
        file_name = os.path.join(characteristics_dir, file_name)
        data = np.concatenate((characteristics, labels), axis=1)
        np.save(file_name, data)
        end_val = time.time()
        print('total feature extraction time for val: {} sec'.format(end_val - start))

        # set testing set
        test_normal_characteristics = get_dknn_nonconformity(x_test_features, calibration_vec, k)
        test_adv_characteristics    = get_dknn_nonconformity(x_test_features_adv, calibration_vec, k)

        dknn_neg = test_normal_characteristics
        dknn_pos = test_adv_characteristics
        characteristics, labels = merge_and_generate_labels(dknn_pos, dknn_neg)

        print("DKNN test: [characteristic shape: ", characteristics.shape, ", label shape: ", labels.shape)
        file_name = 'k_{}_{}'.format(k, 'test')
        file_name = append_suffix(file_name)
        file_name = os.path.join(characteristics_dir, file_name)
        data = np.concatenate((characteristics, labels), axis=1)
        np.save(file_name, data)
        end_test = time.time()
        print('total feature extraction time for test: {} sec'.format(end_test - end_val))
