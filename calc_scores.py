"""
This code is somewhat similar to attack.py, but allows to calculated the influence function measures (I_up_loss)
much faster is you have a multi-core CPU setup. I/We actually used this code to speed us the computations in the paper.
After creating the HVP matrices using calc_hvp.py, run this code to generate scores.npy vector, which is the I_up_loss
of a val/test sample image to each training samples image.

Run this code with CUDA_VISIBLE_DEVICES='' NNIF_adv_defense/calc_scores.py --<flags>...
I/We noticed a large improvement in computation time with this way.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import matplotlib

# Force matplotlib to not use any Xwindows backend.
# import platform
# if platform.system() == 'Linux':
matplotlib.use('Agg')

import logging
import numpy as np
import tensorflow as tf
import os
import imageio
import darkon
from cleverhans.attacks import FastGradientMethod, DeepFool, SaliencyMapMethod, CarliniWagnerL2
from tensorflow.python.platform import flags
from cleverhans.loss import CrossEntropy, WeightDecay, WeightedSum
from NNIF_adv_defense.models.darkon_resnet34_model import DarkonReplica
from cleverhans.utils import AccuracyReport, set_log_level
from NNIF_adv_defense.tools.utils import one_hot
from sklearn.neighbors import NearestNeighbors
from NNIF_adv_defense.datasets.influence_feeder import MyFeederValTest
import pickle
from cleverhans.utils import random_targets
from cleverhans.evaluation import batch_eval

import copy
from threading import Thread
from Queue import Queue


FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 125, 'Size of evaluating batches')
flags.DEFINE_string('dataset', 'cifar10', 'datasset: cifar10/100 or svhn')
flags.DEFINE_string('set', 'val', 'val or test set to evaluate')
flags.DEFINE_string('attack', 'deepfool', 'adversarial attack: deepfool, jsma, cw, cw_nnif')
flags.DEFINE_string('checkpoint_dir', '', 'Checkpoint dir, the path to the saved model architecture and weights')


# TODO: remove
flags.DEFINE_string('mode', 'null', 'to bypass pycharm bug')
flags.DEFINE_string('port', 'null', 'to bypass pycharm bug')

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

# this is the name of the scope of the author(s) Resnet34 graph. If the user wants to just load our network parameters
# and maybe later even use our scores.npy outputs (it takes a long time to compute yourself...), he/she must use
# these strings. Otherwise, any string is OK. We provide here as default the scope names we used.
ARCH_NAME = {'cifar10': 'model1', 'cifar100': 'model_cifar_100', 'svhn': 'model_svhn'}

weight_decay = 0.0004
LABEL_SMOOTHING = {'cifar10': 0.1, 'cifar100': 0.01, 'svhn': 0.1}

# Object used to keep track of (and return) key accuracies
report = AccuracyReport()

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

# make sure the attack dir is constructed
if not os.path.exists(attack_dir):
    os.makedirs(attack_dir)

val_indices = np.load(os.path.join(model_dir, 'val_indices.npy'))

mini_train_inds = None
if USE_TRAIN_MINI:
    train_mini_indices_path = os.path.join(model_dir, 'train_mini_indices.npy')
    if not os.path.exists(train_mini_indices_path):
        print('Creating train mini indices for the test set...')
        # first, we need to create a temporary feeder to fetch all the training samples indices (not validation)
        tmp_feeder = MyFeederValTest(dataset=FLAGS.dataset, rand_gen=rand_gen, as_one_hot=True, val_inds=val_indices,
                                     test_val_set=True, mini_train_inds=None)
        train_inds = tmp_feeder.train_inds
        random_inds = rand_gen.choice(train_inds, 5000, replace=False)  # collecting features for 5000 random training
                                                                        # samples (instead of 49k) due to time complexity
        random_inds.sort()
        np.save(train_mini_indices_path, random_inds)
        del tmp_feeder
    else:
        print('loading train mini indices from {}'.format(train_mini_indices_path))
        mini_train_inds = np.load(train_mini_indices_path)

feeder = MyFeederValTest(dataset=FLAGS.dataset, rand_gen=rand_gen, as_one_hot=True, val_inds=val_indices,
                         test_val_set=test_val_set, mini_train_inds=mini_train_inds)

# get the data
X_train, y_train       = feeder.train_indices(range(feeder.get_train_size()))
X_val, y_val           = feeder.val_indices(range(feeder.get_val_size()))
X_test, y_test         = feeder.test_data, feeder.test_label  # getting the real test set
y_train_sparse         = y_train.argmax(axis=-1).astype(np.int32)
y_val_sparse           = y_val.argmax(axis=-1).astype(np.int32)
y_test_sparse          = y_test.argmax(axis=-1).astype(np.int32)

if TARGETED:
    # get also the adversarial labels of the val and test sets
    if not os.path.isfile(os.path.join(attack_dir, 'y_val_targets.npy')):
        y_val_targets  = random_targets(y_val_sparse , feeder.num_classes)
        y_test_targets = random_targets(y_test_sparse, feeder.num_classes)
        assert (y_val_targets.argmax(axis=1)  != y_val_sparse).all()
        assert (y_test_targets.argmax(axis=1) != y_test_sparse).all()
        np.save(os.path.join(attack_dir, 'y_val_targets.npy') , y_val_targets)
        np.save(os.path.join(attack_dir, 'y_test_targets.npy'), y_test_targets)
    else:
        y_val_targets  = np.load(os.path.join(attack_dir, 'y_val_targets.npy'))
        y_test_targets = np.load(os.path.join(attack_dir, 'y_test_targets.npy'))

# Use Image Parameters
img_rows, img_cols, nchannels = X_test.shape[1:4]
nb_classes = y_test.shape[1]

# Define input TF placeholder
x     = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, nchannels), name='x')
y     = tf.placeholder(tf.float32, shape=(None, nb_classes), name='y')

eval_params = {'batch_size': FLAGS.batch_size}

model = DarkonReplica(scope=ARCH_NAME[FLAGS.dataset], nb_classes=feeder.num_classes, n=5, input_shape=[32, 32, 3])
preds      = model.get_predicted_class(x)
logits     = model.get_logits(x)
embeddings = model.get_embeddings(x)

loss = CrossEntropy(model, smoothing=LABEL_SMOOTHING[FLAGS.dataset])
regu_losses = WeightDecay(model)
full_loss = WeightedSum(model, [(1.0, loss), (weight_decay, regu_losses)])

# loading the checkpoint
saver = tf.train.Saver()
checkpoint_path = os.path.join(model_dir, 'best_model.ckpt')
saver.restore(sess, checkpoint_path)

# predict labels from trainset
if USE_TRAIN_MINI:
    train_preds_file    = os.path.join(model_dir, 'x_train_mini_preds.npy')
    train_features_file = os.path.join(model_dir, 'x_train_mini_features.npy')
else:
    train_preds_file    = os.path.join(model_dir, 'x_train_preds.npy')
    train_features_file = os.path.join(model_dir, 'x_train_features.npy')
if not os.path.isfile(train_preds_file):
    tf_inputs    = [x, y]
    tf_outputs   = [preds, embeddings]
    numpy_inputs = [X_train, y_train]

    x_train_preds, x_train_features = batch_eval(sess, tf_inputs, tf_outputs, numpy_inputs, FLAGS.batch_size)
    x_train_preds = x_train_preds.astype(np.int32)
    np.save(train_preds_file, x_train_preds)
    np.save(train_features_file, x_train_features)
else:
    x_train_preds    = np.load(train_preds_file)
    x_train_features = np.load(train_features_file)

# predict labels from validation set
if not os.path.isfile(os.path.join(model_dir, 'x_val_preds.npy')):
    tf_inputs    = [x, y]
    tf_outputs   = [preds, embeddings]
    numpy_inputs = [X_val, y_val]

    x_val_preds, x_val_features = batch_eval(sess, tf_inputs, tf_outputs, numpy_inputs, FLAGS.batch_size)
    x_val_preds = x_val_preds.astype(np.int32)
    np.save(os.path.join(model_dir, 'x_val_preds.npy')   , x_val_preds)
    np.save(os.path.join(model_dir, 'x_val_features.npy'), x_val_features)
else:
    x_val_preds    = np.load(os.path.join(model_dir, 'x_val_preds.npy'))
    x_val_features = np.load(os.path.join(model_dir, 'x_val_features.npy'))

# predict labels from test set
if not os.path.isfile(os.path.join(model_dir, 'x_test_preds.npy')):
    tf_inputs    = [x, y]
    tf_outputs   = [preds, embeddings]
    numpy_inputs = [X_test, y_test]

    x_test_preds, x_test_features = batch_eval(sess, tf_inputs, tf_outputs, numpy_inputs, FLAGS.batch_size)
    x_test_preds = x_test_preds.astype(np.int32)
    np.save(os.path.join(model_dir, 'x_test_preds.npy')   , x_test_preds)
    np.save(os.path.join(model_dir, 'x_test_features.npy'), x_test_features)
else:
    x_test_preds    = np.load(os.path.join(model_dir, 'x_test_preds.npy'))
    x_test_features = np.load(os.path.join(model_dir, 'x_test_features.npy'))

# initialize adversarial examples if necessary
if not os.path.exists(os.path.join(attack_dir, 'X_val_adv.npy')):
    y_adv = tf.placeholder(tf.float32, shape=(None, nb_classes), name='y_adv')

    # Initialize the advarsarial attack object and graph
    deepfool_params = {
        'clip_min': 0.0,
        'clip_max': 1.0
    }
    jsma_params = {
        'clip_min': 0.0,
        'clip_max': 1.0,
        'theta': 1.0,
        'gamma': 0.1,
    }
    cw_params = {
        'clip_min': 0.0,
        'clip_max': 1.0,
        'batch_size': 125,  # NOTE: you might need to reduce the batch size if your GPU has low memory.
        'confidence': 0.8,
        'learning_rate': 0.01,
        'initial_const': 0.1
    }
    fgsm_param = {
        'clip_min': 0.0,
        'clip_max': 1.0,
        'eps': 0.1
    }
    if TARGETED:
        jsma_params.update({'y_target': y_adv})
        cw_params.update({'y_target': y_adv})
        fgsm_param.update({'y_target': y_adv})

    if FLAGS.attack   == 'deepfool':
        attack_params = deepfool_params
        attack_class  = DeepFool
    elif FLAGS.attack == 'jsma':
        attack_params = jsma_params
        attack_class  = SaliencyMapMethod
    elif FLAGS.attack == 'cw':
        attack_params = cw_params
        attack_class  = CarliniWagnerL2
    elif FLAGS.attack == 'fgsm':
        attack_params = fgsm_param
        attack_class  = FastGradientMethod
    else:
        raise AssertionError('Attack {} is not supported'.format(FLAGS.attack))

    attack         = attack_class(model, sess=sess)
    adv_x          = attack.generate(x, **attack_params)
    preds_adv      = model.get_predicted_class(adv_x)
    logits_adv     = model.get_logits(adv_x)
    embeddings_adv = model.get_embeddings(adv_x)

    # val attack
    tf_inputs    = [x, y]
    tf_outputs   = [adv_x, preds_adv, embeddings_adv]
    numpy_inputs = [X_val, y_val]
    if TARGETED:
        tf_inputs.append(y_adv)
        numpy_inputs.append(y_val_targets)

    X_val_adv, x_val_preds_adv, x_val_features_adv = batch_eval(sess, tf_inputs, tf_outputs, numpy_inputs, FLAGS.batch_size)
    x_val_preds_adv = x_val_preds_adv.astype(np.int32)
    np.save(os.path.join(attack_dir, 'X_val_adv.npy')         , X_val_adv)
    np.save(os.path.join(attack_dir, 'x_val_preds_adv.npy')   , x_val_preds_adv)
    np.save(os.path.join(attack_dir, 'x_val_features_adv.npy'), x_val_features_adv)

    # test attack
    tf_inputs    = [x, y]
    tf_outputs   = [adv_x, preds_adv, embeddings_adv]
    numpy_inputs = [X_test, y_test]
    if TARGETED:
        tf_inputs.append(y_adv)
        numpy_inputs.append(y_test_targets)

    X_test_adv, x_test_preds_adv, x_test_features_adv = batch_eval(sess, tf_inputs, tf_outputs, numpy_inputs, FLAGS.batch_size)
    x_test_preds_adv = x_test_preds_adv.astype(np.int32)
    np.save(os.path.join(attack_dir, 'X_test_adv.npy')         , X_test_adv)
    np.save(os.path.join(attack_dir, 'x_test_preds_adv.npy')   , x_test_preds_adv)
    np.save(os.path.join(attack_dir, 'x_test_features_adv.npy'), x_test_features_adv)
else:
    X_val_adv           = np.load(os.path.join(attack_dir, 'X_val_adv.npy'))
    x_val_preds_adv     = np.load(os.path.join(attack_dir, 'x_val_preds_adv.npy'))
    x_val_features_adv  = np.load(os.path.join(attack_dir, 'x_val_features_adv.npy'))
    X_test_adv          = np.load(os.path.join(attack_dir, 'X_test_adv.npy'))
    x_test_preds_adv    = np.load(os.path.join(attack_dir, 'x_test_preds_adv.npy'))
    x_test_features_adv = np.load(os.path.join(attack_dir, 'x_test_features_adv.npy'))

# quick computations
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

# calculate number of net_succ
val_net_succ_indices              = [ind for ind in info['val'] if info['val'][ind]['net_succ']]
val_net_succ_attack_succ_indices  = [ind for ind in info['val'] if info['val'][ind]['net_succ'] and info['val'][ind]['attack_succ']]
test_net_succ_indices             = [ind for ind in info['test'] if info['test'][ind]['net_succ']]
test_net_succ_attack_succ_indices = [ind for ind in info['test'] if info['test'][ind]['net_succ'] and info['test'][ind]['attack_succ']]
val_attack_rate  = len(val_net_succ_attack_succ_indices)  / len(val_net_succ_indices)
test_attack_rate = len(test_net_succ_attack_succ_indices) / len(test_net_succ_indices)
print('adversarial ({}) validation attack rate: {}\nadversarial ({}) test attack rate: {}'.format(FLAGS.attack, val_attack_rate, FLAGS.attack, test_attack_rate))

info_file = os.path.join(attack_dir, 'info.pkl')
if not os.path.isfile(info_file):
    print('saving info as pickle to {}'.format(info_file))
    with open(info_file, 'wb') as handle:
        pickle.dump(info, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    print('loading info as pickle from {}'.format(info_file))
    with open(info_file, 'rb') as handle:
        info_old = pickle.load(handle)
    assert info == info_old

# start the knn observation
knn = NearestNeighbors(n_neighbors=feeder.get_train_size(), p=2, n_jobs=20)
knn.fit(x_train_features)
if test_val_set:
    print('predicting knn for all val set')
    features     = x_val_features
    features_adv = x_val_features_adv
else:
    print('predicting knn for all test set')
    features     = x_test_features
    features_adv = x_test_features_adv
print('predicting knn dist/indices for normal image')
all_neighbor_dists    , all_neighbor_indices     = knn.kneighbors(features, return_distance=True)
print('predicting knn dist/indices for adv image')
all_neighbor_dists_adv, all_neighbor_indices_adv = knn.kneighbors(features_adv, return_distance=True)

# setting pred feeder. This is our feeder which is used to generate the features for the natural images
pred_feeder = MyFeederValTest(dataset=FLAGS.dataset, rand_gen=rand_gen, as_one_hot=True,
                              val_inds=feeder.val_inds, test_val_set=test_val_set, mini_train_inds=mini_train_inds)
pred_feeder.val_origin_data  = X_val
pred_feeder.val_data         = X_val
pred_feeder.val_label        = one_hot(x_val_preds, feeder.num_classes).astype(np.float32)
pred_feeder.test_origin_data = X_test
pred_feeder.test_data        = X_test
pred_feeder.test_label       = one_hot(x_test_preds, feeder.num_classes).astype(np.float32)

# setting adv feeder. This is our feeder which is used to generate the features for the adv images
adv_feeder = MyFeederValTest(dataset=FLAGS.dataset, rand_gen=rand_gen, as_one_hot=True,
                             val_inds=feeder.val_inds, test_val_set=test_val_set, mini_train_inds=mini_train_inds)
adv_feeder.val_origin_data  = X_val_adv
adv_feeder.val_data         = X_val_adv
adv_feeder.val_label        = one_hot(x_val_preds_adv, feeder.num_classes).astype(np.float32)
adv_feeder.test_origin_data = X_test_adv
adv_feeder.test_data        = X_test_adv
adv_feeder.test_label       = one_hot(x_test_preds_adv, feeder.num_classes).astype(np.float32)

# now finding the influence
feeder.reset()
pred_feeder.reset()
adv_feeder.reset()

inspector_list = []
inspector_pred_list = []
inspector_adv_list = []

for ii in range(FLAGS.num_threads):
    print('Setting feeders for thread #{}...'.format(ii+1))
    inspector_pred_list.append(
        darkon.Influence(
            workspace=os.path.join(workspace_dir, 'pred'),
            feeder=copy.deepcopy(pred_feeder),
            loss_op_train=full_loss.fprop(x=x, y=y),
            loss_op_test=loss.fprop(x=x, y=y),
            x_placeholder=x,
            y_placeholder=y)
    )
    inspector_adv_list.append(
        darkon.Influence(
            workspace=os.path.join(workspace_dir, 'adv', FLAGS.attack),
            feeder=copy.deepcopy(adv_feeder),
            loss_op_train=full_loss.fprop(x=x, y=y),
            loss_op_test=loss.fprop(x=x, y=y),
            x_placeholder=x,
            y_placeholder=y)
    )

# some optimizations for the darkon influence function implementations
testset_batch_size = 100
train_batch_size = 200
train_iterations = 25 if USE_TRAIN_MINI else 245  # 5k(25x200) or 49k(245x200)
approx_params = {
    'scale': 200,
    'num_repeats': 5,
    'recursion_depth': 5 if USE_TRAIN_MINI else 49,  # 5k(5x5x200) or 49k(5x49x200)
    'recursion_batch_size': 200
}

sub_relevant_indices = [ind for ind in info[FLAGS.set]]
relevant_indices     = [info[FLAGS.set][ind]['global_index'] for ind in sub_relevant_indices]

def collect_influence(q, thread_id):
    while not q.empty():
        work = q.get()
        i = work[0]
        try:
            sub_index = sub_relevant_indices[i]
            if test_val_set:
                global_index = feeder.val_inds[sub_index]
            else:
                global_index = feeder.test_inds[sub_index]
            assert global_index == relevant_indices[i]

            _, real_label = feeder.test_indices(sub_index)
            real_label = np.argmax(real_label)

            if test_val_set:
                pred_label = x_val_preds[sub_index]
            else:
                pred_label = x_test_preds[sub_index]

            _, adv_label = adv_feeder.test_indices(sub_index)
            adv_label = np.argmax(adv_label)

            if info[FLAGS.set][sub_index]['attack_succ']:
                assert pred_label != adv_label, 'failed for i={}, sub_index={}, global_index={}'.format(i, sub_index, global_index)
            if info[FLAGS.set][sub_index]['net_succ']:
                assert pred_label == real_label, 'failed for i={}, sub_index={}, global_index={}'.format(i, sub_index, global_index)
            progress_str = 'thread_id: {}. sample {}/{}: calculating scores for {} index {} (sub={}).\n' \
                           'real label: {}, adv label: {}, pred label: {}. net_succ={}, attack_succ={}' \
                .format(thread_id, i + 1, len(sub_relevant_indices), FLAGS.set, global_index, sub_index, _classes[real_label],
                        _classes[adv_label], _classes[pred_label], info[FLAGS.set][sub_index]['net_succ'], info[FLAGS.set][sub_index]['attack_succ'])
            logging.info(progress_str)
            print(progress_str)

            cases = ['pred', 'adv']

            for case in cases:
                if case == 'pred':
                    insp = inspector_pred_list[thread_id]
                    feed = pred_feeder
                elif case == 'adv':
                    insp = inspector_adv_list[thread_id]
                    feed = adv_feeder

                # creating the relevant index folders
                dir = os.path.join(model_dir, FLAGS.set, FLAGS.set + '_index_{}'.format(global_index), case)
                if case == 'adv':
                    dir = os.path.join(dir, FLAGS.attack)
                if not os.path.exists(dir):
                    os.makedirs(dir)

                if os.path.isfile(os.path.join(dir, 'scores.npy')):
                    print('scores already exists in {}'.format(os.path.join(dir, 'scores.npy')))
                else:
                    scores = insp.upweighting_influence_batch(
                        sess=sess,
                        test_indices=[sub_index],
                        test_batch_size=testset_batch_size,
                        approx_params=approx_params,
                        train_batch_size=train_batch_size,
                        train_iterations=train_iterations)
                    np.save(os.path.join(dir, 'scores.npy'), scores)

                print('saving image to {}'.format(os.path.join(dir, 'image.npy/png')))
                image, _ = feed.test_indices(sub_index)
                imageio.imwrite(os.path.join(dir, 'image.png'), image)
                np.save(os.path.join(dir, 'image.npy'), image)
        except Exception as e:
            print('Error with influence collect function for i={}: {}'.format(i, e))
            exit(1)
            raise AssertionError('Error with influence collect function for i={}!'.format(i))

        # signal to the queue that task has been processed
        q.task_done()
    return True

print('Start setting up the queue...')
# set up a queue to hold all the jobs:
q = Queue(maxsize=0)
for i in range(len(sub_relevant_indices)):
    q.put((i,))

for thread_id in range(FLAGS.num_threads):
    print('Starting thread {}'.format(thread_id))
    worker = Thread(target=collect_influence, args=(q, thread_id))
    worker.setDaemon(True)
    worker.start()

q.join()
print('All tasks completed.')
