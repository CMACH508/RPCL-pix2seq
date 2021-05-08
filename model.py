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
"""RPCL-pix2seq model structure file."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import rnn
import Cnn

class Model(object):
    def __init__(self, hps, reuse=tf.AUTO_REUSE):
        self.hps = hps
        with tf.variable_scope('RPCL-pix2seq', reuse=reuse):
            self.config_model()
            self.build_RPCL_pix2seq()

    def config_model(self):
        """ Model configuration """
        self.k = self.hps.num_mixture * self.hps.num_sub  # Gaussian number
        self.global_ = tf.get_variable(name='num_of_steps', shape=[], initializer=tf.ones_initializer(dtype=tf.float32), trainable=False)
        self.de_mu = tf.get_variable(name="latent_mu", shape=[self.k, self.hps.z_size],
                                     initializer=tf.random_uniform_initializer(minval=-1., maxval=1., dtype=tf.float32), trainable=False)
        self.de_sigma2 = tf.get_variable(name="latent_sigma2", shape=[self.k, self.hps.z_size],
                                         initializer=tf.ones_initializer(dtype=tf.float32), trainable=False)
        self.de_alpha = tf.get_variable(name="latent_alpha", shape=[self.k, 1],
                                        initializer=tf.constant_initializer(1. / float(self.k), dtype=tf.float32), trainable=False)

        self.input_seqs = tf.placeholder(tf.float32, [self.hps.batch_size, self.hps.max_seq_len + 1, 5], name="input_seqs")
        self.input_pngs = tf.placeholder(tf.float32, [self.hps.batch_size, self.hps.png_width, self.hps.png_width], name="input_pngs")
        self.input_x = tf.identity(self.input_seqs[:, :self.hps.max_seq_len, :], name='input_x')
        self.output_x = self.input_seqs[:, 1:self.hps.max_seq_len + 1, :]

        # Decoder cell configuration
        if self.hps.dec_model == 'lstm':
            cell_fn = rnn.LSTMCell
        elif self.hps.dec_model == 'layer_norm':
            cell_fn = rnn.LayerNormLSTMCell
        elif self.hps.dec_model == 'hyper':
            cell_fn = rnn.HyperLSTMCell
        else:
            assert False, 'please choose a respectable cell'

        # Dropout configuration
        use_input_dropout = False if self.hps.use_input_dropout == 0 else True
        use_output_dropout = False if self.hps.use_output_dropout == 0 else True
        use_recurrent_dropout = False if self.hps.use_recurrent_dropout == 0 else True

        cell = cell_fn(self.hps.dec_rnn_size,
                       use_recurrent_dropout=use_recurrent_dropout,
                       dropout_keep_prob=self.hps.recurrent_dropout_prob)

        if use_input_dropout:
            tf.logging.info('Dropout to input w/ keep_prob = %4.4f.', self.hps.input_dropout_prob)
            cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.hps.input_dropout_prob)

        if use_output_dropout:
            tf.logging.info('Dropout to output w/ keep_prob = %4.4f.', self.hps.output_dropout_prob)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.hps.output_dropout_prob)

        self.cell = cell

    def build_RPCL_pix2seq(self):
        target = tf.reshape(self.output_x, [-1, 5])
        self.x1_data, self.x2_data, self.pen_data = tf.split(target, [1, 1, 3], axis=1)

        # CNN encoder
        self.p_mu, self.p_sigma2 = self.cnn_encoder(self.input_pngs)
        self.batch_z = self.get_z(self.p_mu, self.p_sigma2)  # reparameterization

        # Compute decoder initial states and inputs
        fc_spec = [('tanh', self.hps.z_size, self.cell.state_size, 'init_state')]
        fc_net = Cnn.FcNet(fc_spec, self.batch_z)
        self.initial_state = fc_net.fc_layers[-1]
        pre_z = tf.tile(tf.reshape(self.batch_z, [self.hps.batch_size, 1, self.hps.z_size]), [1, self.hps.max_seq_len, 1])
        dec_input = tf.concat([self.input_x, pre_z], axis=2)

        # Deconv branch
        self.gen_img = self.cnn_decoder(self.batch_z)
        # Generation branch
        self.dec_out, self.final_state = self.rnn_decoder(dec_input, self.initial_state)
        self.pi, self.mux, self.muy, self.sigmax, self.sigmay, self.corr, self.pen, self.pen_logits = self.dec_out

        # Estimate the posterior
        self.p_alpha, self.gau_label = self.calculate_posterior(self.batch_z, self.de_mu, self.de_sigma2, self.de_alpha)
        # Update latent parameters (we use EM algorithm to stabilize the GMM learning)
        self.q_mu, self.q_sigma2, self.q_alpha \
            = tf.cond(tf.cast(self.global_ > 500, tf.bool),
                      lambda: tf.cond(tf.cast(self.global_ % 3, tf.bool),
                                      lambda: self.em(self.p_mu, self.p_sigma2, self.p_alpha, self.de_mu, self.de_sigma2, self.de_alpha),
                                      lambda: self.rpcl(self.p_mu, self.p_sigma2, self.p_alpha, self.de_mu, self.de_sigma2, self.de_alpha)),
                      lambda: self.em(self.p_mu, self.p_sigma2, self.p_alpha, self.de_mu, self.de_sigma2, self.de_alpha))

        # Loss function
        self.alpha_loss = self.get_alpha_loss(self.p_alpha, tf.stop_gradient(self.q_alpha))
        self.gaussian_loss = self.get_gaussian_loss(self.p_alpha, self.p_mu, self.p_sigma2, tf.stop_gradient(self.q_mu), tf.stop_gradient(self.q_sigma2))
        self.lil_loss = self.get_lil_loss(self.pi, self.mux, self.muy, self.sigmax, self.sigmay, self.corr,
                                          self.pen_logits, self.x1_data, self.x2_data, self.pen_data)
        self.de_loss = self.calculate_deconv_loss(self.input_pngs, self.gen_img, 'square')

        self.kl_weight = 1. - 0.999 * (0.9999 ** self.global_)  # Warm up
        self.loss = self.kl_weight * (self.alpha_loss + self.gaussian_loss) + self.lil_loss + self.hps.de_weight * self.de_loss

        self.lr = (self.hps.learning_rate - self.hps.min_learning_rate) * \
                  (self.hps.decay_rate ** (self.global_ / 3)) + self.hps.min_learning_rate
        optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.5)
        gvs = optimizer.compute_gradients(self.loss)
        g = self.hps.grad_clip
        for i, (grad, var) in enumerate(gvs):
            if grad is not None:
                gvs[i] = (tf.clip_by_value(grad, -g, g), var)
        self.train_op = optimizer.apply_gradients(gvs)

        # Update the GMM parameters
        self.update_gmm_mu = tf.assign(self.de_mu, self.q_mu)
        self.update_gmm_sigma2 = tf.assign(self.de_sigma2, self.q_sigma2)
        self.update_gmm_alpha = tf.assign(self.de_alpha, self.q_alpha)

    def cnn_encoder(self, inputs):
        with tf.variable_scope('encoder'):
            inputs = tf.reshape(inputs, [-1, self.hps.png_width, self.hps.png_width, 1])

            block1_out = self.create_dilated_blocks(inputs, 'layer_1', 32, 2)
            block2_out = self.create_dilated_blocks(block1_out, 'layer_2', 64, 2)

            conv_specs = [
                ('relu', (3, 3), [1, 2, 2, 1], 128),
                ('relu', (3, 3), [1, 2, 2, 1], 256),
            ]
            cn1 = Cnn.ConvNet(conv_specs, block2_out, self.hps.is_training)
            cn1_out = cn1.conv_layers[-1]
            outputs = tf.reshape(cn1_out, shape=[-1, 3 * 3 * 256])

            # Compute mean and std_square for the posterior q(z|x)
            fc_spec_mu = [('no', 3 * 3 * 256, self.hps.z_size, 'fc_mu')]
            fc_net_mu = Cnn.FcNet(fc_spec_mu, outputs)
            p_mu = fc_net_mu.fc_layers[-1]

            fc_spec_sigma2 = [('no', 3 * 3 * 256, self.hps.z_size, 'fc_sigma2')]
            fc_net_sigma2 = Cnn.FcNet(fc_spec_sigma2, outputs)
            p_sigma2 = fc_net_sigma2.fc_layers[-1]
        return p_mu, tf.nn.softplus(p_sigma2) + 1e-10

    def cnn_decoder(self, code):
        with tf.variable_scope('deconv'):
            fc_spec = [
                ('relu', self.hps.z_size, 3 * 3 * 256, 'fc1'),
            ]
            fc_net = Cnn.FcNet(fc_spec, code)
            fc1 = fc_net.fc_layers[-1]
            fc1 = tf.reshape(fc1, [-1, 3, 3, 256])

            de_conv_specs = [
                ('relu', (3, 3), [1, 2, 2, 1], 128),
                ('relu', (3, 3), [1, 2, 2, 1], 64),
                ('relu', (3, 3), [1, 2, 2, 1], 32),
                ('tanh', (3, 3), [1, 2, 2, 1], 1)
            ]
            conv_net = Cnn.ConvNet(de_conv_specs, fc1, self.hps.is_training, deconv=True)
        return conv_net.conv_layers[-1]

    def rnn_decoder(self, inputs, initial_state):
        # Number of outputs is end_of_stroke + prob + 2 * (mu + sig) + corr
        num_mixture = 20
        n_out = (3 + num_mixture * 6)

        with tf.variable_scope('decoder'):
            output, last_state = tf.nn.dynamic_rnn(
                self.cell,
                inputs,
                initial_state=initial_state,
                time_major=False,
                swap_memory=True,
                dtype=tf.float32)

            output = tf.reshape(output, [-1, self.hps.dec_rnn_size])
            fc_spec = [('no', self.hps.dec_rnn_size, n_out, 'fc')]
            fc_net = Cnn.FcNet(fc_spec, output)
            output = fc_net.fc_layers[-1]

            out = self.get_mixture_params(output)
            last_state = tf.identity(last_state, name='last_state')
            self.output = output
        return out, last_state

    def calculate_prob(self, x, q_mu, q_sigma2):
        """ Calculate the probabilistic density """
        mu = tf.tile(tf.reshape(q_mu, [1, self.k, self.hps.z_size]), [self.hps.batch_size, 1, 1])
        sigma2 = tf.tile(tf.reshape(q_sigma2, [1, self.k, self.hps.z_size]), [self.hps.batch_size, 1, 1])
        x = tf.tile(tf.reshape(x, [self.hps.batch_size, 1, self.hps.z_size]), [1, self.k, 1])

        log_exp_part = -0.5 * tf.reduce_sum(tf.divide(tf.square(x - mu), 1e-30 + sigma2), axis=2)
        log_frac_part = tf.reduce_sum(tf.log(tf.sqrt(sigma2 + 1e-30)), axis=2)
        log_prob = log_exp_part - log_frac_part - float(self.hps.z_size) / 2. * tf.log(2. * 3.1416)
        return tf.exp(tf.to_double(log_prob))

    def calculate_posterior(self, y, q_mu, q_sigma2, q_alpha):
        """ Calculate the posterior p(y|k) """
        prob = self.calculate_prob(y, q_mu, q_sigma2)
        temp = tf.multiply(tf.tile(tf.transpose(tf.to_double(q_alpha), [1, 0]), [self.hps.batch_size, 1]), prob)
        sum_temp = tf.tile(tf.reduce_sum(temp, axis=1, keep_dims=True), [1, self.k])
        gamma = tf.clip_by_value(tf.to_float(tf.divide(temp, 1e-300 + sum_temp)), 1e-5, 1.)
        gamma_st = gamma / (1e-10 + tf.tile(tf.reduce_sum(gamma, axis=1, keep_dims=True), [1, self.k]))
        return gamma_st, tf.argmax(gamma_st, axis=1)

    def rpcl(self, y, en_sigma2, gamma, q_mu_old, q_sigma2_old, q_alpha_old):
        """ EM-like algorithm enhanced by RPCL """
        en_sigma2 = tf.tile(tf.reshape(en_sigma2, [self.hps.batch_size, 1, self.hps.z_size]), [1, self.k, 1])

        with tf.variable_scope('rpcl'):
            penalize = 1e-4  # De-learning rate
            temp_y = tf.tile(tf.expand_dims(y, axis=1), [1, self.k, 1])
            winner = tf.one_hot(tf.argmax(gamma, axis=1), self.k, axis=1)  # the winner
            rival = tf.one_hot(tf.argmax(gamma - gamma * winner, axis=1), self.k, axis=1)  # the rival
            gamma_rpcl = winner - penalize * rival
            sum_gamma_rpcl = tf.tile(tf.expand_dims(tf.reduce_sum(gamma_rpcl, axis=0), axis=1), [1, self.hps.z_size])
            q_mu_new = tf.reduce_sum(temp_y * tf.tile(tf.expand_dims(gamma_rpcl, axis=2),
                                                      [1, 1, self.hps.z_size]), axis=0) / (sum_gamma_rpcl + 1e-10)
            q_sigma2_new = tf.reduce_sum((tf.square(temp_y - tf.tile(tf.expand_dims(q_mu_new, axis=0), [self.hps.batch_size, 1, 1])) + en_sigma2)
                                         * tf.tile(tf.expand_dims(gamma_rpcl, axis=2), [1, 1, self.hps.z_size]), axis=0) \
                           / (sum_gamma_rpcl + 1e-10)
            q_alpha_new = tf.expand_dims(tf.reduce_mean(gamma_rpcl, axis=0), axis=1)

            q_mu = q_mu_old * 0.95 + q_mu_new * 0.05
            q_sigma2 = tf.clip_by_value(q_sigma2_old * 0.95 + q_sigma2_new * 0.05, 1e-10, 1e10)
            q_alpha = tf.clip_by_value(q_alpha_old * 0.95 + q_alpha_new * 0.05, 0., 1.)
            q_alpha_st = q_alpha / tf.reduce_sum(q_alpha)

            return q_mu, q_sigma2, q_alpha_st

    def em(self, y, en_sigma2, gamma, q_mu_old, q_sigma2_old, q_alpha_old):
        """ EM algorithm for GMM learning """
        en_sigma2 = tf.tile(tf.reshape(en_sigma2, [self.hps.batch_size, 1, self.hps.z_size]), [1, self.k, 1])

        with tf.variable_scope('em'):
            sum_gamma = tf.tile(tf.expand_dims(tf.reduce_sum(gamma, axis=0), axis=1), [1, self.hps.z_size])
            temp_y = tf.tile(tf.expand_dims(y, axis=1), [1, self.k, 1])

            q_mu_new = tf.reduce_sum(temp_y * tf.tile(tf.expand_dims(gamma, axis=2), [1, 1, self.hps.z_size]), axis=0) / (sum_gamma + 1e-10)
            q_sigma2_new = tf.reduce_sum((tf.square(temp_y - tf.tile(tf.expand_dims(q_mu_new, axis=0), [self.hps.batch_size, 1, 1])) + en_sigma2)
                                         * tf.tile(tf.expand_dims(gamma, axis=2), [1, 1, self.hps.z_size]), axis=0) / (sum_gamma + 1e-10)
            q_alpha_new = tf.expand_dims(tf.reduce_mean(gamma, axis=0), axis=1)

            q_mu = q_mu_old * 0.95 + q_mu_new * 0.05
            q_sigma2 = tf.clip_by_value(q_sigma2_old * 0.95 + q_sigma2_new * 0.05, 1e-10, 1e10)
            q_alpha = tf.clip_by_value(q_alpha_old * 0.95 + q_alpha_new * 0.05, 0., 1.)
            q_alpha_st = q_alpha / tf.reduce_sum(q_alpha)

            return q_mu, q_sigma2, q_alpha_st

    def create_dilated_blocks(self, inputs, scope, depth, stride=1):
        with tf.variable_scope(scope):
            with tf.variable_scope('dcn1'):
                dcn1 = Cnn.DilatedConv([('no', (3, 3), 1, depth)], inputs, self.hps.is_training)
                dcn1_out = dcn1.conv_layers[-1]

            with tf.variable_scope('dcn2'):
                dcn2 = Cnn.DilatedConv([('no', (3, 3), 2, depth)], inputs, self.hps.is_training)
                dcn2_out = dcn2.conv_layers[-1]

            with tf.variable_scope('dcn3'):
                dcn3 = Cnn.DilatedConv([('no', (3, 3), 5, depth)], inputs, self.hps.is_training)
                dcn3_out = dcn3.conv_layers[-1]

            with tf.variable_scope('combine'):
                combine = Cnn.ConvNet([('no', (3, 3), [1, stride, stride, 1], depth)], tf.nn.relu(dcn1_out + dcn2_out + dcn3_out), self.hps.is_training)
                combine_out = combine.conv_layers[-1]

            return tf.nn.relu(combine_out)

    def get_z(self, mu, sigma2):
        """ Reparameterization """
        sigma = tf.sqrt(sigma2)
        eps = tf.random_normal((self.hps.batch_size, self.hps.z_size), 0.0, 1.0, dtype=tf.float32)
        z = tf.add(mu, tf.multiply(sigma, eps), name='z_code')
        return z

    def get_alpha_loss(self, p_alpha, q_alpha):
        p_alpha = tf.reshape(p_alpha, [self.hps.batch_size, self.k])
        q_alpha = tf.tile(tf.reshape(q_alpha, [1, self.k]), [self.hps.batch_size, 1])
        return tf.reduce_sum(tf.reduce_mean(p_alpha * tf.log(tf.div(p_alpha, q_alpha + 1e-10) + 1e-10), axis=0))

    def get_gaussian_loss(self, p_alpha, p_mu, p_sigma2, q_mu, q_sigma2):
        p_alpha = tf.reshape(p_alpha, [self.hps.batch_size, self.k])
        p_mu = tf.tile(tf.reshape(p_mu, [self.hps.batch_size, 1, self.hps.z_size]), [1, self.k, 1])
        p_sigma2 = tf.tile(tf.reshape(p_sigma2, [self.hps.batch_size, 1, self.hps.z_size]), [1, self.k, 1])
        q_mu = tf.tile(tf.reshape(q_mu, [1, self.k, self.hps.z_size]), [self.hps.batch_size, 1, 1])
        q_sigma2 = tf.tile(tf.reshape(q_sigma2, [1, self.k, self.hps.z_size]), [self.hps.batch_size, 1, 1])
        return tf.reduce_mean(tf.reduce_sum(0.5 * tf.multiply(p_alpha, tf.reduce_sum(
            tf.log(q_sigma2 + 1e-10) + tf.div(p_sigma2 + (p_mu - q_mu) ** 2, q_sigma2 + 1e-10) - 1.0 - tf.log(
                p_sigma2 + 1e-10), axis=2)), axis=1))

    def calculate_deconv_loss(self, img, gen_img, sign):
        img = tf.reshape(img, [self.hps.batch_size, self.hps.png_width ** 2])
        gen_img = tf.reshape(gen_img, [self.hps.batch_size, self.hps.png_width ** 2])
        if sign == 'square':
            return tf.reduce_mean(tf.reduce_sum(tf.square(img - gen_img), axis=1))
        elif sign == 'absolute':
            return tf.reduce_mean(tf.reduce_sum(tf.abs(img - gen_img), axis=1))
        else:
            assert False, 'please choose a respectable cell'

    def get_density(self, x1, x2, mu1, mu2, s1, s2, rho):
        norm1 = tf.subtract(x1, mu1)
        norm2 = tf.subtract(x2, mu2)
        s1s2 = tf.multiply(s1, s2)
        z = (tf.square(tf.div(norm1, s1)) + tf.square(tf.div(norm2, s2)) -
             2 * tf.div(tf.multiply(rho, tf.multiply(norm1, norm2)), s1s2))
        neg_rho = 1 - tf.square(rho)
        result = tf.exp(tf.div(-z, 2 * neg_rho))
        denom = 2 * np.pi * tf.multiply(s1s2, tf.sqrt(neg_rho))
        result = tf.div(result, denom)
        return result

    def get_lil_loss(self, pi, mu1, mu2, s1, s2, corr, pen_logits, x1_data, x2_data, pen_data):
        result0 = self.get_density(x1_data, x2_data, mu1, mu2, s1, s2, corr)
        epsilon = 1e-6
        result1 = tf.multiply(result0, pi)
        result1 = tf.reduce_sum(result1, axis=1, keep_dims=True)
        result1 = -tf.log(result1 + epsilon)  # Avoid log(0)

        masks = 1.0 - pen_data[:, 2]
        masks = tf.reshape(masks, [-1, 1])
        result1 = tf.multiply(result1, masks)

        result2 = tf.nn.softmax_cross_entropy_with_logits(logits=pen_logits, labels=pen_data)
        result2 = tf.reshape(result2, [-1, 1])

        if not self.hps.is_training:
            result2 = tf.multiply(result2, masks)
        return tf.reduce_mean(tf.reduce_sum(tf.reshape(result1 + result2, [self.hps.batch_size, -1]), axis=1))

    def get_mixture_params(self, output):
        pen_logits = output[:, 0:3]
        pi, mu1, mu2, sigma1, sigma2, corr = tf.split(output[:, 3:], 6, 1)

        pi = tf.nn.softmax(pi)
        pen = tf.nn.softmax(pen_logits)

        sigma1 = tf.exp(sigma1)
        sigma2 = tf.exp(sigma2)
        corr = tf.tanh(corr)

        r = [pi, mu1, mu2, sigma1, sigma2, corr, pen, pen_logits]
        return r
