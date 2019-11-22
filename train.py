"""
Training a basic Resnet 34 network for classification, splitting to train/val/test
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import numpy as np
import tensorflow as tf
import os

from cleverhans.augmentation import random_horizontal_flip, random_shift
from tensorflow.python.platform import flags
from cleverhans.loss import CrossEntropy, WeightDecay, WeightedSum
from NNIF_adv_defense.models.darkon_resnet34_model import DarkonReplica
from NNIF_adv_defense.trainer import train
from cleverhans.utils import AccuracyReport, set_log_level
from cleverhans.utils_tf import model_eval
from NNIF_adv_defense.datasets.influence_feeder import MyFeederValTest

FLAGS = flags.FLAGS

flags.DEFINE_integer('nb_epochs', 200, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 125, 'Size of training batches')
flags.DEFINE_string('dataset', 'svhn', 'dataset: cifar10/100 or svhn')
flags.DEFINE_string('checkpoint_dir', '', 'Checkpoint dir, the path to the saved model architecture and weights')

# this is the name of the scope of the Resnet34 graph. If the user wants to just load our network parameters
# and maybe later even use our scores.npy outputs (it takes a long time to compute yourself...), he/she must use
# these strings. Otherwise, any string is OK. We provide here as default the scope names we used.
ARCH_NAME = {'cifar10': 'model1', 'cifar100': 'model_cifar_100', 'svhn': 'model_svhn'}
weight_decay = 0.0004
label_smoothing = {'cifar10': 0.1, 'cifar100': 0.01, 'svhn': 0.1}
if FLAGS.checkpoint_dir != '':
    model_dir     = FLAGS.checkpoint_dir                      # set user specified dir
else:
    model_dir = os.path.join(FLAGS.dataset, 'trained_model')  # set default dir

# Object used to keep track of (and return) key accuracies
report = AccuracyReport()

# Set TF random seed to improve reproducibility
superseed = 123456789
rand_gen = np.random.RandomState(superseed)
tf.set_random_seed(superseed)

# Set logging level to see debug information
set_log_level(logging.INFO)

# Create TF session
config_args = dict(allow_soft_placement=True)
sess = tf.Session(config=tf.ConfigProto(**config_args))

feeder = MyFeederValTest(dataset=FLAGS.dataset, rand_gen=rand_gen, as_one_hot=True, test_val_set=True)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
np.save(os.path.join(model_dir, 'val_indices.npy'), feeder.val_inds)

# get the data
X_train, y_train       = feeder.train_indices(range(feeder.get_train_size()))
X_val, y_val           = feeder.val_indices(range(feeder.get_val_size()))
X_test, y_test         = feeder.test_data, feeder.test_label  # getting the real test set
y_train_sparse         = y_train.argmax(axis=-1).astype(np.int32)
y_val_sparse           = y_val.argmax(axis=-1).astype(np.int32)
y_test_sparse          = y_test.argmax(axis=-1).astype(np.int32)

dataset_size  = X_train.shape[0]
dataset_train = tf.data.Dataset.range(dataset_size)
dataset_train = dataset_train.shuffle(4096)
dataset_train = dataset_train.repeat()

def lookup(p):
    return X_train[p], y_train[p]
dataset_train = dataset_train.map(lambda i: tf.py_func(lookup, [i], [tf.float32] * 2))

dataset_train = dataset_train.map(lambda x, y: (random_shift(random_horizontal_flip(x)), y), 4)
dataset_train = dataset_train.batch(FLAGS.batch_size)
dataset_train = dataset_train.prefetch(16)

# Use Image Parameters
img_rows, img_cols, nchannels = X_val.shape[1:4]
nb_classes = y_val.shape[1]

# Define input TF placeholder
x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, nchannels))
y = tf.placeholder(tf.float32, shape=(None, nb_classes))

# Train a model
train_params = {
    'nb_epochs': FLAGS.nb_epochs,
    'batch_size': FLAGS.batch_size,
    'learning_rate': 0.1,
    'lr_factor': 0.9,
    'lr_patience': 3,
    'lr_cooldown': 2,
    'best_model_path': os.path.join(model_dir, 'best_model.ckpt')
}
eval_params = {'batch_size': FLAGS.batch_size}
fgsm_params = {
    'eps': 0.3,
    'clip_min': 0.,
    'clip_max': 1.
}

model = DarkonReplica(scope=ARCH_NAME[FLAGS.dataset], nb_classes=feeder.num_classes, n=5, input_shape=[32, 32, 3])
logits = model.get_logits(x)
loss = CrossEntropy(model, smoothing=label_smoothing[FLAGS.dataset])
regu_losses = WeightDecay(model)
full_loss = WeightedSum(model, [(1.0, loss), (weight_decay, regu_losses)])

def do_eval(preds, x_set, y_set, report_key, is_adv=None):
    acc = model_eval(sess, x, y, preds, x_set, y_set, args=eval_params)
    setattr(report, report_key, acc)
    if is_adv is None:
        report_text = None
    elif is_adv:
        report_text = 'adversarial'
    else:
        report_text = 'legitimate'
    if report_text:
        print('Test accuracy on %s examples: %0.4f' % (report_text, acc))
    return acc

def evaluate():
    return do_eval(logits, X_val, y_val, 'clean_train_clean_eval', False)


train(sess, full_loss, None, None,
      dataset_train=dataset_train, dataset_size=dataset_size,
      evaluate=evaluate, args=train_params, rng=rand_gen,
      var_list=model.get_params(),
      optimizer='mom')

# Uncomment if you want to save the latest model, instead of the best model (in terms of val accuracy)
# save_path = os.path.join(model_dir, "model_checkpoint.ckpt")
# saver = tf.train.Saver()
# saver.save(sess, save_path, global_step=tf.train.get_global_step())

# print best score
evaluate()

print('done')