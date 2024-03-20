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
        return tf.compat.v1.get_variable(name, dtype=tf.float32, initializer=self.xavier_init(shape),use_resource=False)

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
        conv = tf.nn.conv2d(input=inputs, filters=w,strides=[1, 1, 1, 1], padding='SAME', dilations=[1, rate, rate, 1])
        # 在原始的TensorFlow 1.x版本中，tf.nn.atrous_conv2d操作中，步数默认为1，而膨胀率由rate参数直接定义，所以在直接调用时没有显示出来。而在TensorFlow 2.x的tf.nn.conv2d操作中，步数和膨胀率需要分别用strides和dilations参数来明确给出。

        instance_norm_layer = tfa.layers.InstanceNormalization()
        conv = instance_norm_layer(conv)
        # 实例归一化在tf.contrib.layers模块下，但是在TensorFlow 2.0版本中，tf.contrib模块已经被移除了。作为替代，实例归一化的功能已经迁移到了TensorFlow Addons，相应的函数为tfa.layers.InstanceNormalization
        # conv = tf.nn.atrous_conv2d(inputs, w, rate=rate, padding='SAME')
        # conv = tf.contrib.layers.instance_norm(conv)
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
        return tf.compat.v1.get_variable(name, dtype=tf.float32, initializer=self.xavier_init(shape),use_resource=False)

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
        conv = tf.nn.conv2d(input=inputs, filters=w, strides=strides, padding='SAME')
        instance_norm_layer = tfa.layers.InstanceNormalization()
        conv = instance_norm_layer(conv)
        # conv = tf.contrib.layers.instance_norm(conv)
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
        instance_norm_layer = tfa.layers.InstanceNormalization()
        conv = instance_norm_layer(conv)
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
            w = tf.compat.v1.get_variable(name='fc_w', dtype=tf.float32, initializer=self.xavier_init([input_size, output_size]),use_resource=False)
            if use_bias:
                b = tf.compat.v1.get_variable('fc_b', [output_size], tf.float32, initializer=tf.compat.v1.constant_initializer(0.0),use_resource=False)
                temp = tf.matmul(x, w) + b
            else:
                temp = tf.matmul(x, w)
            if actfun == 'no':
                return temp
            else:
                f = self.select_act_func(actfun)
                return f(temp)