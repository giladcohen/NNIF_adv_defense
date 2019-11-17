"""Extremely simple model where all parameters are from convolutions.
"""

import tensorflow as tf

from cleverhans.model import Model
from collections import OrderedDict

BN_EPSILON = 0.001

def activation_summary(x):
    '''
    :param x: A Tensor
    :return: Add histogram summary and scalar summary of the sparsity of the tensor
    '''
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def create_variables(name, shape, initializer=tf.contrib.layers.xavier_initializer(), is_fc_layer=False):
    '''
    :param name: A string. The name of the new variable
    :param shape: A list of dimensions
    :param initializer: User Xavier as default.
    :param is_fc_layer: Want to create fc layer variable? May use different weight_decay for fc
    layers.
    :return: The created variable
    '''

    ## TODO: to allow different weight decay to fully connected layer and conv layer
    if is_fc_layer is True:
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.0)
    else:
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.0)

    new_variables = tf.get_variable(name, shape=shape, initializer=initializer, regularizer=regularizer)
    return new_variables

def output_layer(input_layer, num_labels):
    '''
    :param input_layer: 2D tensor
    :param num_labels: int. How many output labels in total? (10 for cifar10 and 100 for cifar100)
    :return: output layer Y = WX + B
    '''
    input_dim = input_layer.get_shape().as_list()[-1]
    fc_w = create_variables(name='fc_weights', shape=[input_dim, num_labels], is_fc_layer=True,
                            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    fc_b = create_variables(name='fc_bias', shape=[num_labels], initializer=tf.zeros_initializer())

    fc_h = tf.matmul(input_layer, fc_w) + fc_b
    return fc_h

def batch_normalization_layer(input_layer, dimension):
    '''
    Helper function to do batch normalziation
    :param input_layer: 4D tensor
    :param dimension: input_layer.get_shape().as_list()[-1]. The depth of the 4D tensor
    :return: the 4D tensor after being normalized
    '''
    mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
    beta = tf.get_variable('beta', dimension, tf.float32,
                               initializer=tf.constant_initializer(0.0, tf.float32))
    gamma = tf.get_variable('gamma', dimension, tf.float32,
                                initializer=tf.constant_initializer(1.0, tf.float32))
    bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)

    return bn_layer

def conv_bn_relu_layer(input_layer, filter_shape, stride):
    '''
    A helper function to conv, batch normalize and relu the input tensor sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = Relu(batch_normalize(conv(X)))
    '''

    out_channel = filter_shape[-1]
    filter = create_variables(name='conv', shape=filter_shape)

    conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    bn_layer = batch_normalization_layer(conv_layer, out_channel)

    output = tf.nn.relu(bn_layer)
    return output

# def bn_relu_conv_layer(input_layer, filter_shape, stride):
#     '''
#     A helper function to batch normalize, relu and conv the input layer sequentially
#     :param input_layer: 4D tensor
#     :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
#     :param stride: stride size for conv
#     :return: 4D tensor. Y = conv(Relu(batch_normalize(X)))
#     '''
#
#     in_channel = input_layer.get_shape().as_list()[-1]
#
#     bn_layer = batch_normalization_layer(input_layer, in_channel)
#     relu_layer = tf.nn.relu(bn_layer)
#
#     filter = create_variables(name='conv', shape=filter_shape)
#     conv_layer = tf.nn.conv2d(relu_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
#     return conv_layer

def residual_block(input_layer, output_channel, first_block=False, net=None, layer_cnt=None):
    '''
    Defines a residual block in ResNet
    :param input_layer: 4D tensor
    :param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
    :param first_block: if this is the first residual block of the whole network
    :param net: dictionary of net layers
    :param layer_cnt: layer number to collect. Collecting layer_cnt and layet_cnt+1 after relus
    :return: 4D tensor.
    '''
    input_channel = input_layer.get_shape().as_list()[-1]

    # When it's time to "shrink" the image size, we use stride = 2
    if input_channel * 2 == output_channel:
        increase_dim = True
        stride = 2
    elif input_channel == output_channel:
        increase_dim = False
        stride = 1
    else:
        raise ValueError('Output and input channel does not match in residual blocks!!!')

    # The first conv layer of the first residual block does not need to be normalized and relu-ed.
    with tf.variable_scope('conv1_in_block'):
        if first_block:
            filter = create_variables(name='conv', shape=[3, 3, input_channel, output_channel])
            conv1 = tf.nn.conv2d(input_layer, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
        else:
            in_channel = input_layer.get_shape().as_list()[-1]
            bn_layer = batch_normalization_layer(input_layer, in_channel)
            relu_layer = tf.nn.relu(bn_layer)
            net['layer{}'.format(layer_cnt)] = relu_layer
            layer_cnt += 1
            filter = create_variables(name='conv', shape=[3, 3, input_channel, output_channel])
            conv1 = tf.nn.conv2d(relu_layer, filter, strides=[1, stride, stride, 1], padding='SAME')

    with tf.variable_scope('conv2_in_block'):
        in_channel = conv1.get_shape().as_list()[-1]
        bn_layer = batch_normalization_layer(conv1, in_channel)
        relu_layer = tf.nn.relu(bn_layer)
        net['layer{}'.format(layer_cnt)] = relu_layer
        filter = create_variables(name='conv', shape=[3, 3, output_channel, output_channel])
        conv2 = tf.nn.conv2d(relu_layer, filter, strides=[1, 1, 1, 1], padding='SAME')

    # When the channels of input layer and conv2 does not match, we add zero pads to increase the
    #  depth of input layers
    if increase_dim is True:
        pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2, input_channel // 2]])
    else:
        padded_input = input_layer

    output = conv2 + padded_input
    return output


class DarkonReplica(Model):

    O_EMBEDDINGS = 'embeddings'

    def __init__(self, scope, nb_classes, n, input_shape, **kwargs):
        del kwargs
        Model.__init__(self, scope, nb_classes, locals())
        self.n = n
        self.net = OrderedDict()

        # Do a dummy run of fprop to create the variables from the start
        self.fprop(tf.placeholder(tf.float32, [32] + input_shape))
        # Put a reference to the params in self so that the params get pickled
        self.params = self.get_params()

    def fprop(self, x, **kwargs):
        del kwargs
        layer_cnt = 0

        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            with tf.variable_scope('conv0', reuse=tf.AUTO_REUSE):
                x = conv_bn_relu_layer(x, [3, 3, 3, 16], 1)
                self.net['layer0'] = x
                layer_cnt += 1
                activation_summary(x)

            for i in range(self.n):
                with tf.variable_scope('conv1_%d' %i, reuse=tf.AUTO_REUSE):
                    if i == 0:
                        x = residual_block(x, 16, first_block=True, net=self.net, layer_cnt=layer_cnt)
                        layer_cnt = layer_cnt + 1
                    else:
                        x = residual_block(x, 16, net=self.net, layer_cnt=layer_cnt)
                        layer_cnt = layer_cnt + 2
                    activation_summary(x)

            for i in range(self.n):
                with tf.variable_scope('conv2_%d' %i, reuse=tf.AUTO_REUSE):
                    x = residual_block(x, 32, net=self.net, layer_cnt=layer_cnt)
                    layer_cnt = layer_cnt + 2
                    activation_summary(x)

            for i in range(self.n):
                with tf.variable_scope('conv3_%d' %i, reuse=tf.AUTO_REUSE):
                    x = residual_block(x, 64, net=self.net, layer_cnt=layer_cnt)
                    layer_cnt = layer_cnt + 2
                assert x.get_shape().as_list()[1:] == [8, 8, 64]

            with tf.variable_scope('fc', reuse=tf.AUTO_REUSE):
                in_channel = x.get_shape().as_list()[-1]
                bn_layer = batch_normalization_layer(x, in_channel)
                relu_layer = tf.nn.relu(bn_layer)
                self.net['layer{}'.format(layer_cnt)] = relu_layer  # 30
                layer_cnt += 1
                global_pool = tf.reduce_mean(relu_layer, [1, 2])

                # Embedding layer
                embedding_vector = global_pool
                self.net['layer{}'.format(layer_cnt)] = embedding_vector  # 31
                layer_cnt += 1

                assert global_pool.get_shape().as_list()[-1:] == [64]
                logits = output_layer(global_pool, self.nb_classes)
                self.net['layer{}'.format(layer_cnt)] = logits  # 32
                layer_cnt += 1

            return {self.O_EMBEDDINGS: embedding_vector,
                    self.O_LOGITS: logits,
                    self.O_PROBS: tf.nn.softmax(logits=logits)}

    def get_embeddings(self, x, **kwargs):
        outputs = self.fprop(x, **kwargs)
        if self.O_EMBEDDINGS in outputs:
            return outputs[self.O_EMBEDDINGS]
        raise NotImplementedError(str(type(self)) + "must implement `get_embeddings`"
                                  " or must define a " + self.O_EMBEDDINGS +
                                  " output in `fprop`")