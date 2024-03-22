#   Copyright 2021 Sicong Zang
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
#   P.S. We thank Ha and Eck [1] for their public source codes.
#        And the details about their work can be found below.
#
#       [1] https://github.com/tensorflow/magenta/tree/master/magenta/models/sketch_rnn
#
""" Modules for CNN & FC layers"""

import tensorflow as tf
import tensorflow_addons as tfa
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()

class DilatedConv(object):
    def __init__(self, specs, inputs, is_training, keep_prob=1.0):
        self.conv_layers = []
        self.keep_prob = keep_prob
        outputs = inputs
        for i, (fun_name, w_size, rate, out_channel) in enumerate(specs):
            self.conv_layers.append(outputs)
            with tf.compat.v1.variable_scope('conv%d' % i):
                outputs = self.build_dilated_conv_layer(outputs, w_size, out_channel, rate, fun_name, is_training=is_training)
        self.conv_layers.append(outputs)

    def xavier_init(self, size):
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        return tf.random.normal(shape=size, stddev=xavier_stddev)

    def get_filter(self, name, shape):
        return tf.compat.v1.get_variable(name, dtype=tf.float32, initializer=self.xavier_init(shape),use_resource=True)

    def select_act_func(self, actfun):
        if actfun == 'tanh':
            return tf.nn.tanh
        elif actfun == 'sigmoid':
            return tf.sigmoid
        elif actfun == 'relu':
            return tf.nn.relu
        else:
            return lambda x: x

    def build_dilated_conv_layer(self, inputs, w_size, out_channel, rate, actfun='relu', is_training=True):
        batch_size, height, width, in_channel = inputs.get_shape().as_list()
        w = self.get_filter('filter', [w_size[0], w_size[1], in_channel, out_channel])
        f = self.select_act_func(actfun)
        conv = tf.compat.v1.nn.atrous_conv2d(inputs, w, rate=rate, padding='SAME')
        InstanceNormalization()(conv)
        out = f(conv)
        return out

class ConvNet(object):
    def __init__(self, specs, inputs, is_training, deconv=False, keep_prob=1.0):
        self.conv_layers = []
        self.keep_prob = keep_prob
        outputs = inputs
        for i, (fun_name, w_size, strides, out_channel) in enumerate(specs):
            self.conv_layers.append(outputs)
            if deconv == False:
                with tf.compat.v1.variable_scope('conv%d' % i):
                    outputs = self.build_conv_layer(outputs, w_size, out_channel, strides, fun_name, is_training=is_training)
            else:
                with tf.compat.v1.variable_scope('deconv%d' % i):
                    outputs = self.build_deconv_layer(outputs, w_size, out_channel, strides, fun_name, is_training=is_training)
        self.conv_layers.append(outputs)

    def xavier_init(self, size):
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        return tf.random.normal(shape=size, stddev=xavier_stddev)

    def get_filter(self, name, shape):
        return tf.compat.v1.get_variable(name, dtype=tf.float32, initializer=self.xavier_init(shape),use_resource=True)

    def select_act_func(self, actfun):
        if actfun == 'tanh':
            return tf.nn.tanh
        elif actfun == 'sigmoid':
            return tf.sigmoid
        elif actfun == 'relu':
            return tf.nn.relu
        else:
            return lambda x: x

    def build_conv_layer(self, inputs, w_size, out_channel, strides=[1, 2, 2, 1], actfun='relu', is_training=True):
        batch_size, height, width, in_channel = inputs.get_shape().as_list()
        w = self.get_filter('filter', [w_size[0], w_size[1], in_channel, out_channel])
        f = self.select_act_func(actfun)
        conv = tf.nn.conv2d(inputs, w, strides=strides, padding='SAME')
        InstanceNormalization()(conv)
        out = f(conv)
        return out

    def build_deconv_layer(self, inputs, w_size, out_channel, strides=[1, 2, 2, 1], actfun='relu', is_training=True):
        batch_size, height, width, in_channel = inputs.get_shape().as_list()
        w = self.get_filter('filter', [w_size[0], w_size[1], out_channel, in_channel])
        if strides[1] == 1:
            deconv_shape = [batch_size, height, width, out_channel]
        else:
            deconv_shape = [batch_size, height * 2, width * 2, out_channel]
        f = self.select_act_func(actfun)
        conv = tf.nn.conv2d_transpose(inputs, w, deconv_shape, strides=strides, padding='SAME')
        InstanceNormalization()(conv)
        out = f(conv)
        return out

class FcNet(object):
    def __init__(self, specs, inputs):
        self.fc_layers = []
        outputs = inputs
        for i, (fun_name, in_size, out_size, scope) in enumerate(specs):
            self.fc_layers.append(outputs)
            with tf.compat.v1.variable_scope(scope):
                outputs = self.build_fc_layer(x=outputs, input_size=in_size, output_size=out_size, actfun=fun_name)
        self.fc_layers.append(outputs)

    def xavier_init(self, size):
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        return tf.random.normal(shape=size, stddev=xavier_stddev)

    def select_act_func(self, actfun):
        if actfun == 'tanh':
            return tf.nn.tanh
        elif actfun == 'sigmoid':
            return tf.sigmoid
        elif actfun == 'relu':
            return tf.nn.relu
        else:
            return lambda x: x

    def build_fc_layer(self, x, input_size, output_size, actfun, scope=None, use_bias=True):
        with tf.compat.v1.variable_scope(scope or 'linear'):
            w = tf.compat.v1.get_variable(name='fc_w', dtype=tf.float32, initializer=self.xavier_init([input_size, output_size]),use_resource=True)
            if use_bias:
                b = tf.compat.v1.get_variable('fc_b', [output_size], tf.float32, initializer=tf.compat.v1.constant_initializer(0.0),use_resource=True)
                temp = tf.matmul(x, w) + b
            else:
                temp = tf.matmul(x, w)
            if actfun == 'no':
                return temp
            else:
                f = self.select_act_func(actfun)
                return f(temp)