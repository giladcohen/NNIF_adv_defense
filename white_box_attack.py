"""
This code generate the CW-Opt adversarial examples. The CW-Opt is the CW with a new regularization term optimized
to evade my/our NNIF detection algorithm.
Run this code twice - once for the val set and second for the test set.
To make this code finish within 2-3 hours, run this code using a GPU.
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
from tqdm import tqdm
from tensorflow.python.platform import flags
from NNIF_adv_defense.models.darkon_resnet34_model import DarkonReplica
from NNIF_adv_defense.datasets.influence_feeder import MyFeederValTest
from NNIF_adv_defense.white_box.cw_opt_attack import CarliniNNIF
from cleverhans.utils import random_targets
from cleverhans.evaluation import batch_eval
from cleverhans.utils import AccuracyReport, set_log_level
from cleverhans.loss import CrossEntropy, WeightDecay, WeightedSum

FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 125, 'Size of training batches')
flags.DEFINE_string('dataset', 'cifar10', 'datasset: cifar10/100 or svhn')
flags.DEFINE_string('set', 'val', 'val or test set to evaluate')
flags.DEFINE_string('checkpoint_dir', '', 'Checkpoint dir, the path to the saved model architecture and weights')

# TODO: remove after final debug
flags.DEFINE_string('mode', 'null', 'to bypass pycharm bug')
flags.DEFINE_string('port', 'null', 'to bypass pycharm bug')

if FLAGS.set == 'val':
    test_val_set = True
    WORKSPACE = 'influence_workspace_validation'
    USE_TRAIN_MINI = False
else:
    test_val_set = False
    WORKSPACE = 'influence_workspace_test_mini'
    USE_TRAIN_MINI = True

TARGETED = True

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
NUM_INDICES = {'cifar10': 50, 'cifar100': 5, 'svhn': 50}

# Object used to keep track of (and return) key accuracies
report = AccuracyReport()

# Set TF random seed to improve reproducibility
superseed = 15101985
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
attack_dir    = os.path.join(model_dir, 'cw_nnif')
if TARGETED:
    attack_dir = attack_dir + '_targeted'

# make sure the attack dir is constructed
if not os.path.exists(attack_dir):
    os.makedirs(attack_dir)

# The val_indices file must already be in this directory after you call attack.py script
val_indices = np.load(os.path.join(model_dir, 'val_indices.npy'))

mini_train_inds = None
if USE_TRAIN_MINI:
    # The train_mini_indices file must already be in this directory after you call attack.py script with test set
    print('loading train mini indices from {}'.format(os.path.join(model_dir, 'train_mini_indices.npy')))
    mini_train_inds = np.load(os.path.join(model_dir, 'train_mini_indices.npy'))

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

model = DarkonReplica(scope=ARCH_NAME, nb_classes=feeder.num_classes, n=5, input_shape=[32, 32, 3])
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

# quick computations (without adv)
train_acc    = np.mean(y_train_sparse == x_train_preds)
val_acc      = np.mean(y_val_sparse   == x_val_preds)
test_acc     = np.mean(y_test_sparse  == x_test_preds)
print('train set acc: {}\nvalidation set acc: {}\ntest set acc: {}'.format(train_acc, val_acc, test_acc))

# what are the indices of the set which the network succeeded classifying correctly,
# but the adversarial attack changed to a different class?
info_tmp = {}
info_tmp['val'] = {}
for i, set_ind in enumerate(feeder.val_inds):
    info_tmp['val'][i] = {}
    net_succ    = x_val_preds[i] == y_val_sparse[i]
    # attack_succ = x_val_preds[i] != x_val_preds_adv[i]  # the attack success is unknown yet
    info_tmp['val'][i]['global_index'] = set_ind
    info_tmp['val'][i]['net_succ']     = net_succ
    # info_tmp['val'][i]['attack_succ']  = attack_succ
info_tmp['test'] = {}
for i, set_ind in enumerate(feeder.test_inds):
    info_tmp['test'][i] = {}
    net_succ    = x_test_preds[i] == y_test_sparse[i]
    # attack_succ = x_test_preds[i] != x_test_preds_adv[i]
    info_tmp['test'][i]['global_index'] = set_ind
    info_tmp['test'][i]['net_succ']     = net_succ
    # info_tmp['test'][i]['attack_succ']  = attack_succ  # the attack success is unknown yet

sub_relevant_indices = [ind for ind in info_tmp[FLAGS.set]]
relevant_indices     = [info_tmp[FLAGS.set][ind]['global_index'] for ind in sub_relevant_indices]

# saves time if need to re-calculated
helpful_npy_path = os.path.join(attack_dir, '{}_most_helpful.npy'.format(FLAGS.set))
harmful_npy_path = os.path.join(attack_dir, '{}_most_harmful.npy'.format(FLAGS.set))

if not os.path.exists(helpful_npy_path):
    # loading the embedding vectors of all the val's/test's most harmful/helpful training examples
    most_helpful_list = []
    most_harmful_list = []

    for i in tqdm(range(len(sub_relevant_indices))):
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

        if info_tmp[FLAGS.set][sub_index]['net_succ']:
            assert pred_label == real_label, 'failed for i={}, sub_index={}, global_index={}'.format(i, sub_index, global_index)

        progress_str = 'sample {}/{}: processing helpful/harmful for {} index {} (sub={}).\n' \
                       'real label: {}, pred label: {}. net_succ={}' \
            .format(i + 1, len(sub_relevant_indices), FLAGS.set, global_index, sub_index, _classes[real_label],
                    _classes[pred_label], info_tmp[FLAGS.set][sub_index]['net_succ'])
        logging.info(progress_str)
        print(progress_str)

        # creating the relevant index folders
        dir = os.path.join(model_dir, FLAGS.set, FLAGS.set + '_index_{}'.format(global_index), 'pred')
        scores = np.load(os.path.join(dir, 'scores.npy'))
        sorted_indices = np.argsort(scores)
        harmful_inds = sorted_indices[:NUM_INDICES[FLAGS.dataset]]
        helpful_inds = sorted_indices[-NUM_INDICES[FLAGS.dataset]:][::-1]

        # find out the embedding space of the train images in the tanh space
        # first we calculate the tanh transformation:
        X_train_transform = (np.tanh(X_train) + 1) / 2

        most_helpful_images = X_train_transform[helpful_inds]
        most_harmful_images = X_train_transform[harmful_inds]
        train_helpful_embeddings = batch_eval(sess, [x, y], [embeddings], [most_helpful_images, y_train[helpful_inds]], FLAGS.batch_size)[0]
        train_harmful_embeddings = batch_eval(sess, [x, y], [embeddings], [most_harmful_images, y_train[harmful_inds]], FLAGS.batch_size)[0]

        most_helpful_list.append(train_helpful_embeddings)
        most_harmful_list.append(train_harmful_embeddings)

    most_helpful = np.asarray(most_helpful_list)
    most_harmful = np.asarray(most_harmful_list)
    np.save(helpful_npy_path, most_helpful)
    np.save(harmful_npy_path, most_harmful)
else:
    print('{} already exist. Loading...'.format(helpful_npy_path))
    most_helpful = np.load(helpful_npy_path)
    most_harmful = np.load(harmful_npy_path)

# initialize adversarial examples if necessary
if not os.path.exists(os.path.join(attack_dir, 'X_{}_adv.npy'.format(FLAGS.set))):
    y_adv     = tf.placeholder(tf.float32, shape=(None, nb_classes), name='y_adv')
    m_help_ph = tf.placeholder(tf.float32, shape=(None,) + most_helpful.shape[1:])
    m_harm_ph = tf.placeholder(tf.float32, shape=(None,) + most_harmful.shape[1:])

    # Initialize the advarsarial attack object and graph
    attack_params = {
        'clip_min': 0.0,
        'clip_max': 1.0,
        'batch_size': 125,  # NOTE: you might need to reduce the batch size if your GPU has low memory.
        'confidence': 0.8,
        'learning_rate': 0.01,
        'initial_const': 0.1,
        'y_target': y_adv,
        'most_helpful_locs': m_help_ph,
        'most_harmful_locs': m_harm_ph
    }

    attack         = CarliniNNIF(model, sess=sess)
    adv_x          = attack.generate(x, **attack_params)
    preds_adv      = model.get_predicted_class(adv_x)
    logits_adv     = model.get_logits(adv_x)
    embeddings_adv = model.get_embeddings(adv_x)

    # attack
    tf_inputs    = [x, y, y_adv, m_help_ph, m_harm_ph]
    tf_outputs   = [adv_x, preds_adv, embeddings_adv]
    if FLAGS.set == 'val':
        numpy_inputs = [X_val, y_val, y_val_targets, most_helpful, most_harmful]
    elif FLAGS.set == 'test':
        numpy_inputs = [X_test, y_test, y_test_targets, most_helpful, most_harmful]

    X_set_adv, x_set_preds_adv, x_set_features_adv = batch_eval(sess, tf_inputs, tf_outputs, numpy_inputs, FLAGS.batch_size)
    x_set_preds_adv = x_set_preds_adv.astype(np.int32)
    np.save(os.path.join(attack_dir, 'X_{}_adv.npy'.format(FLAGS.set))         , X_set_adv)
    np.save(os.path.join(attack_dir, 'x_{}_preds_adv.npy'.format(FLAGS.set))   , x_set_preds_adv)
    np.save(os.path.join(attack_dir, 'x_{}_features_adv.npy'.format(FLAGS.set)), x_set_features_adv)
else:
    print('{} already exists'.format(os.path.join(attack_dir, 'X_{}_adv.npy'.format(FLAGS.set))))

print('Done generating adversarial images for subset: {}.\nMake sure to run this script for both val/test'.format(FLAGS.set))