"""
This CarliniNNIF class generates adversarial images using the my/our CW-Opt attack.
# This code needs the networks predictions for the untempered images (".../pred/cw/scores.npy"), so run it after
# acquiring all the CW scores using "NNIF_adv_defense/attack.py" or "NNIF_adv_defense/calc_scores.py"
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf

from cleverhans import utils
from cleverhans.attacks import CarliniWagnerL2
from cleverhans.compat import reduce_max
from cleverhans.compat import reduce_sum


_logger = utils.create_logger("cleverhans.cw_attacks")
np_dtype = np.dtype('float32')
tf_dtype = tf.as_dtype('float32')

def ZERO():
  return np.asarray(0., dtype=np_dtype)

class CWL2_NNIF(object):
    def __init__(self, sess, model, batch_size, confidence, targeted,
                 learning_rate, binary_search_steps, max_iterations,
                 abort_early, initial_const, clip_min, clip_max, num_labels,
                 shape):
        """
        Return a tensor that constructs adversarial examples for the given
        input. Generate uses tf.py_func in order to operate over tensors.

        :param sess: a TF session.
        :param model: a cleverhans.model.Model object.
        :param batch_size: Number of attacks to run simultaneously.
        :param confidence: Confidence of adversarial examples: higher produces
                           examples with larger l2 distortion, but more
                           strongly classified as adversarial.
        :param targeted: boolean controlling the behavior of the adversarial
                         examples produced. If set to False, they will be
                         misclassified in any wrong class. If set to True,
                         they will be misclassified in a chosen target class.
        :param learning_rate: The learning rate for the attack algorithm.
                              Smaller values produce better results but are
                              slower to converge.
        :param binary_search_steps: The number of times we perform binary
                                    search to find the optimal tradeoff-
                                    constant between norm of the purturbation
                                    and confidence of the classification.
        :param max_iterations: The maximum number of iterations. Setting this
                               to a larger value will produce lower distortion
                               results. Using only a few iterations requires
                               a larger learning rate, and will produce larger
                               distortion results.
        :param abort_early: If true, allows early aborts if gradient descent
                            is unable to make progress (i.e., gets stuck in
                            a local minimum).
        :param initial_const: The initial tradeoff-constant to use to tune the
                              relative importance of size of the pururbation
                              and confidence of classification.
                              If binary_search_steps is large, the initial
                              constant is not important. A smaller value of
                              this constant gives lower distortion results.
        :param clip_min: (optional float) Minimum input component value.
        :param clip_max: (optional float) Maximum input component value.
        :param num_labels: the number of classes in the model's output.
        :param shape: the shape of the model's input tensor.
        """

        self.sess = sess
        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.ABORT_EARLY = abort_early
        self.CONFIDENCE = confidence
        self.initial_const = initial_const
        self.batch_size = batch_size
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.model = model

        self.repeat = binary_search_steps >= 10

        self.shape = shape = tuple([batch_size] + list(shape))
        self.num_indices = int(-0.5 * (num_labels - 10) + 50)  # 50 for CIFAR-10/SVHN, 5 for CIFAR-100

        # the variable we're going to optimize over
        modifier = tf.Variable(np.zeros(shape, dtype=np_dtype))

        # these are variables to be more efficient in sending data to tf
        self.timg   = tf.Variable(np.zeros(shape), dtype=tf_dtype, name='timg')
        self.tlab   = tf.Variable(np.zeros((batch_size, num_labels)), dtype=tf_dtype, name='tlab')
        self.const  = tf.Variable(np.zeros(batch_size), dtype=tf_dtype, name='const')
        self.m_help = tf.Variable(np.zeros((batch_size, self.num_indices, 64)), dtype=tf_dtype, name='m_help')
        self.m_harm = tf.Variable(np.zeros((batch_size, self.num_indices, 64)), dtype=tf_dtype, name='m_harm')

        # and here's what we use to assign them
        self.assign_timg   = tf.placeholder(tf_dtype, shape, name='assign_timg')
        self.assign_tlab   = tf.placeholder(tf_dtype, (batch_size, num_labels), name='assign_tlab')
        self.assign_const  = tf.placeholder(tf_dtype, [batch_size], name='assign_const')
        self.assign_m_help = tf.placeholder(tf_dtype, (batch_size, self.num_indices, 64), name='assign_m_help')
        self.assign_m_harm = tf.placeholder(tf_dtype, (batch_size, self.num_indices, 64), name='assign_m_harm')

        # the resulting instance, tanh'd to keep bounded from clip_min
        # to clip_max
        self.newimg = (tf.tanh(modifier + self.timg) + 1) / 2
        self.newimg = self.newimg * (clip_max - clip_min) + clip_min

        # Embedding vector of the new image: (tanh(x_adv) + 1) / 2
        self.embeddings = model.get_embeddings(self.newimg)
        # prediction BEFORE-SOFTMAX of the model
        self.output = model.get_logits(self.newimg)

        # distance to the input data
        self.other = (tf.tanh(self.timg) + 1) / 2 * (clip_max - clip_min) + clip_min
        self.l2dist = reduce_sum(tf.square(self.newimg - self.other), list(range(1, len(shape))))

        # compute the probability of the label class versus the maximum other
        real = reduce_sum((self.tlab) * self.output, 1)
        other = reduce_max((1 - self.tlab) * self.output - self.tlab * 10000, 1)
        if self.TARGETED:
            # if targeted, optimize for making the other class most likely
            loss1 = tf.maximum(ZERO(), other - real + self.CONFIDENCE)
        else:
            # if untargeted, optimize for making this class least likely.
            loss1 = tf.maximum(ZERO(), real - other + self.CONFIDENCE)

        # NNID addition:
        self.embeddings = tf.expand_dims(self.embeddings, axis=1)  # expand (batch, 64) to (batch, 1, 64)
        self.embeddings = tf.tile(self.embeddings, [1, self.num_indices, 1])  # duplicate (batch, 1, 64) to (batch, self.num_indices, 64)
        self.helpful_dist = tf.norm(self.embeddings - self.m_help, ord=1, axis=(1, 2))  # , axis=list(range(1, len(self.embeddings.shape))))
        # harmful_dist = reduce_sum(tf.norm(self.embeddings - self.m_harm, ord=1, axis=list(range(1, len(self.embeddings)))))

        # sum up the losses
        self.loss2 = reduce_sum(self.l2dist)
        self.loss1 = reduce_sum(self.const * (loss1 + self.helpful_dist))  # here LID added loss1 + loss_lid
        self.loss = self.loss1 + self.loss2

        # Setup the adam optimizer and keep track of variables we're creating
        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
        self.train = optimizer.minimize(self.loss, var_list=[modifier])
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        # these are the variables to initialize when we run
        self.setup = []
        self.setup.append(self.timg.assign(self.assign_timg))
        self.setup.append(self.tlab.assign(self.assign_tlab))
        self.setup.append(self.const.assign(self.assign_const))
        self.setup.append(self.m_help.assign(self.assign_m_help))
        self.setup.append(self.m_harm.assign(self.assign_m_harm))

        self.init = tf.variables_initializer(var_list=[modifier] + new_vars)

    def attack(self, imgs, targets, m_help, m_harm):
        """
        Perform the L_2 attack on the given instance for the given targets.

        If self.targeted is true, then the targets represents the target labels
        If self.targeted is false, then targets are the original class labels
        """

        r = []
        for i in range(0, len(imgs), self.batch_size):
            _logger.debug(
                ("Running CWL2-NNIF attack on instance %s of %s", i, len(imgs)))
            r.extend(
                self.attack_batch(imgs[i:i + self.batch_size],
                                  targets[i:i + self.batch_size],
                                  m_help[i:i + self.batch_size],
                                  m_harm[i:i + self.batch_size])
            )
        return np.array(r)

    def attack_batch(self, imgs, labs, m_help, m_harm):
        """
        Run the attack on a batch of instance and labels.
        """

        def compare(x, y):
            if not isinstance(x, (float, int, np.int64)):
                x = np.copy(x)
                if self.TARGETED:
                    x[y] -= self.CONFIDENCE
                else:
                    x[y] += self.CONFIDENCE
                x = np.argmax(x)
            if self.TARGETED:
                return x == y
            else:
                return x != y

        batch_size = self.batch_size

        oimgs = np.clip(imgs, self.clip_min, self.clip_max)

        # re-scale instances to be within range [0, 1]
        imgs = (imgs - self.clip_min) / (self.clip_max - self.clip_min)
        imgs = np.clip(imgs, 0, 1)
        # now convert to [-1, 1]
        imgs = (imgs * 2) - 1
        # convert to tanh-space
        imgs = np.arctanh(imgs * .999999)

        # set the lower and upper bounds accordingly
        lower_bound = np.zeros(batch_size)
        CONST = np.ones(batch_size) * self.initial_const
        upper_bound = np.ones(batch_size) * 1e10

        # placeholders for the best l2, score, and instance attack found so far
        o_bestl2 = [1e10] * batch_size
        o_bestln = [1e10] * batch_size
        o_bestscore = [-1] * batch_size
        o_bestattack = np.copy(oimgs)

        for outer_step in range(self.BINARY_SEARCH_STEPS):
            # completely reset adam's internal state.
            self.sess.run(self.init)
            batch        = imgs[:batch_size]
            batchlab     = labs[:batch_size]
            batch_m_help = m_help[:batch_size]
            batch_m_harm = m_harm[:batch_size]

            bestl2 = [1e10] * batch_size
            bestln = [1e10] * batch_size
            bestscore = [-1] * batch_size
            _logger.debug("  Binary search step %s of %s",
                          outer_step, self.BINARY_SEARCH_STEPS)

            # The last iteration (if we run many steps) repeat the search once.
            if self.repeat and outer_step == self.BINARY_SEARCH_STEPS - 1:
                CONST = upper_bound

            # set the variables so that we don't have to send them over again
            self.sess.run(
                self.setup, {
                    self.assign_timg: batch,
                    self.assign_tlab: batchlab,
                    self.assign_const: CONST,
                    self.assign_m_help: batch_m_help,
                    self.assign_m_harm: batch_m_harm
                })

            prev = 1e6
            for iteration in range(self.MAX_ITERATIONS):
                # perform the attack
                _, l, l2s, l_nnif, scores, nimg = self.sess.run([
                    self.train, self.loss, self.l2dist, self.helpful_dist, self.output, self.newimg
                ])

                if iteration % ((self.MAX_ITERATIONS // 10) or 1) == 0:
                    _logger.debug(("    Iteration {} of {}: loss={:.3g} " +
                                   "l2={:.3g} l_nnif={:.3g} f={:.3g}").format(
                        iteration, self.MAX_ITERATIONS, l,
                        np.mean(l2s), np.mean(l_nnif), np.mean(scores)))

                # check if we should abort search if we're getting nowhere.
                if self.ABORT_EARLY and \
                        iteration % ((self.MAX_ITERATIONS // 10) or 1) == 0:
                    if l > prev * .9999:
                        msg = "    Failed to make progress; stop early"
                        _logger.debug(msg)
                        break
                    prev = l

                # adjust the best result found so far
                for e, (l2, ln, sc, ii) in enumerate(zip(l2s, l_nnif, scores, nimg)):
                    lab = np.argmax(batchlab[e])
                    if l2 < bestl2[e] and compare(sc, lab):
                        bestl2[e] = l2
                        bestln[e] = ln
                        bestscore[e] = np.argmax(sc)
                    if l2 < o_bestl2[e] and compare(sc, lab):
                        o_bestl2[e] = l2
                        o_bestln[e] = ln
                        o_bestscore[e] = np.argmax(sc)
                        o_bestattack[e] = ii

            # adjust the constant as needed
            for e in range(batch_size):
                if compare(bestscore[e], np.argmax(batchlab[e])) and \
                        bestscore[e] != -1:
                    # success, divide const by two
                    upper_bound[e] = min(upper_bound[e], CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
                else:
                    # failure, either multiply by 10 if no solution found yet
                    #          or do binary search with the known upper bound
                    lower_bound[e] = max(lower_bound[e], CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
                    else:
                        CONST[e] *= 10
            _logger.debug("  Successfully generated adversarial examples " +
                          "on {} of {} instances.".format(
                              sum(upper_bound < 1e9), batch_size))
            o_bestl2 = np.array(o_bestl2)
            o_bestln = np.array(o_bestln)
            l2_mean = np.mean(np.sqrt(o_bestl2[o_bestl2 < 1e9]))
            ln_mean = np.mean(np.sqrt(o_bestln[o_bestln < 1e9]))
            _logger.debug("   Mean successful l2 distortion: {:.4g}, l_nnif distortion: {:.4g}".format(l2_mean, ln_mean))

        # return the best solution found
        o_bestl2 = np.array(o_bestl2)
        o_bestln = np.array(o_bestln)
        return o_bestattack


class CarliniNNIF(CarliniWagnerL2):

    def __init__(self, model, sess, dtypestr='float32', **kwargs):
        super(CarliniNNIF, self).__init__(model, sess, dtypestr, **kwargs)
        self.feedable_kwargs += ('most_helpful_locs', 'most_harmful_locs')

    def generate(self, x, **kwargs):
        """
        Return a tensor that constructs adversarial examples for the given
        input. Generate uses tf.py_func in order to operate over tensors.

        :param x: A tensor with the inputs.
        :param kwargs: See `parse_params`
        """
        assert self.sess is not None, \
            'Cannot use `generate` when no `sess` was provided'
        self.parse_params(**kwargs)

        labels, nb_classes = self.get_or_guess_labels(x, kwargs)
        most_helpful_locs = kwargs['most_helpful_locs']
        most_harmful_locs = kwargs['most_harmful_locs']

        attack = CWL2_NNIF(self.sess, self.model, self.batch_size, self.confidence,
                      'y_target' in kwargs, self.learning_rate,
                      self.binary_search_steps, self.max_iterations,
                      self.abort_early, self.initial_const, self.clip_min,
                      self.clip_max, nb_classes,
                      x.get_shape().as_list()[1:])

        def cw_wrap(x_val, y_val, m_help, m_harm):
          return np.array(attack.attack(x_val, y_val, m_help, m_harm), dtype=self.np_dtype)

        wrap = tf.py_func(cw_wrap, [x, labels, most_helpful_locs, most_harmful_locs], self.tf_dtype)
        wrap.set_shape(x.get_shape())

        return wrap

    def parse_params(self,
                     y=None,
                     y_target=None,
                     most_helpful_locs=None,
                     most_harmful_locs=None,
                     batch_size=1,
                     confidence=0,
                     learning_rate=5e-3,
                     binary_search_steps=5,
                     max_iterations=1000,
                     abort_early=True,
                     initial_const=1e-2,
                     clip_min=0,
                     clip_max=1):

        # ignore the y and y_target argument
        self.batch_size = batch_size
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.binary_search_steps = binary_search_steps
        self.max_iterations = max_iterations
        self.abort_early = abort_early
        self.initial_const = initial_const
        self.clip_min = clip_min
        self.clip_max = clip_max
