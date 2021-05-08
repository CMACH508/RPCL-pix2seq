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
""" Sampling/synthesizing process file."""

import random
import os
import json
import numpy as np
import tensorflow as tf
import utils
import glob
from PIL import Image
from seq2png import draw_strokes
from model import Model
import scipy.misc
import re
from svg2png import exportsvg

def sample(sess, sample_model, z, gen_size=1, seq_len=250, temperature=0.24, greedy_mode=False):
    """ Sample a sequence of strokes """

    def adjust_pdf(pi_pdf, temp):
        """ Adjust the pdf of pi according to temperature """
        pi_pdf = np.log(pi_pdf) / temp
        pi_pdf -= pi_pdf.max()
        pi_pdf = np.exp(pi_pdf)
        pi_pdf /= pi_pdf.sum()
        return pi_pdf


    def get_pi_idx(x, pdf, temp=1.0, greedy=False):
        """ Sample from a pdf, optionally greedily """
        if greedy:
            return np.argmax(pdf)
        pdf = adjust_pdf(np.copy(pdf), temp)
        accumulate = 0
        for i in range(0, pdf.size):
            accumulate += pdf[i]
            if accumulate >= x:
                return i
        tf.logging.info('Error with sampling ensemble.')
        return -1


    def sample_gaussian_2d(mu1, mu2, s1, s2, rho, temp=1.0, greedy=False):
        """ Sample from a 2D Gaussian """
        if greedy:
            return mu1, mu2
        mean = [mu1, mu2]
        s1 *= temp * temp
        s2 *= temp * temp
        cov = [[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]]
        x = np.random.multivariate_normal(mean, cov, 1)
        return x[0][0], x[0][1]


    def get_seqs(z, seq_len, greedy, temp):
        """ Generate sequences according to latent vector """
        feed = {sample_model.batch_z: z}
        input_state = sess.run(sample_model.initial_state, feed)

        strokes = np.zeros((seq_len, len(z), 5), dtype=np.float32)
        input_x = np.zeros((len(z), 1, 5), dtype=np.float32)
        input_x[:, 0, 2] = 1  # Initially, we want to see beginning of new stroke

        for seq_i in range(seq_len):
            feed = {sample_model.initial_state: input_state,
                    sample_model.input_x: input_x,
                    sample_model.batch_z: z
                    }

            dec_out, out_state = sess.run([sample_model.dec_out, sample_model.final_state], feed)

            pi, mux, muy, sigmax, sigmay, corr, pen, pen_logits = dec_out
            input_state = out_state

            # Generate stroke position from Gaussian mixtures
            idx = get_pi_idx(random.random(), pi[0], temp, greedy)

            next_x1, next_x2 = sample_gaussian_2d(mux[0][idx], muy[0][idx],
                                                  sigmax[0][idx], sigmay[0][idx],
                                                  corr[0][idx], np.sqrt(temp), greedy)
            # Generate stroke pen status
            idx_eos = get_pi_idx(random.random(), pen[0], temp, greedy)

            eos = np.zeros(3)
            eos[idx_eos] = 1

            strokes[seq_i, :, :] = [next_x1, next_x2, eos[0], eos[1], eos[2]]

            input_x = np.array([next_x1, next_x2, eos[0], eos[1], eos[2]], dtype=np.float32)
            input_x = input_x.reshape([1, 1, 5])

        return utils.seq_5d_to_3d(np.reshape(strokes, [seq_len, 5]))


    # Generate a batch of sketches based on one latent vector
    gen_strokes = []
    for i in range(gen_size):
        sketch = get_seqs(z, seq_len, greedy_mode, temperature)
        gen_strokes.append(sketch)
    return gen_strokes


def load_model_params(model_dir):
    model_params = utils.get_default_hparams()
    with tf.gfile.Open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
        model_config = json.dumps(json.load(f))
        model_params.parse_json(model_config)
    return model_params


def modify_model_params(model_params):
    """ Adjust to the generating mode """
    model_params.use_input_dropout = 0
    model_params.use_recurrent_dropout = 0
    model_params.use_output_dropout = 0
    model_params.is_training = False
    model_params.batch_size = 1
    model_params.max_seq_len = 1

    return model_params

def sort_paths(paths):
    """ Order the loaded images """
    idxs = []
    for path in paths:
        idxs.append(int(re.findall(r'\d+', path)[-1]))

    for i in range(len(idxs)):
        for j in range(i+1, len(idxs)):
            if idxs[i] > idxs[j]:
                tmp = idxs[i]
                idxs[i] = idxs[j]
                idxs[j] = tmp

                tmp = paths[i]
                paths[i] = paths[j]
                paths[j] = tmp
    return paths

def main():
    FLAGS = tf.app.flags.FLAGS
    # Dataset directory
    tf.app.flags.DEFINE_string(
        'data_dir',
        'dataset_path',
        'The directory in which to find the dataset specified in model hparams. '
    )
    # Checkpoint directory
    tf.app.flags.DEFINE_string(
        'model_dir', 'checkpoint',
        'Directory to store the model checkpoints.'
    )
    # Output directory
    tf.app.flags.DEFINE_string(
        'output_dir', 'sample',
        'Directory to store the generated sketches.'
    )
    # Number of generated samples per category
    tf.app.flags.DEFINE_integer(
        'num_per_category', 2500,
        'Number of generated samples per category.'
    )
    # Whether the sampling needs the sketch images input as references
    tf.app.flags.DEFINE_boolean(
        'conditional', True,
        'Whether the sampling is with conditions.'
    )

    color = ['black', 'red', 'blue', 'green', 'orange', 'cyan', 'tomato', 'magenta', 'purple', 'brown']
    model_dir = FLAGS.model_dir
    data_dir = FLAGS.data_dir
    SVG_DIR = FLAGS.output_dir
    samples_per_category = FLAGS.num_per_category
    # Temperature for synthesis, details can be found in aforementioned reference [1]
    temperature = 0.24

    model_params = load_model_params(model_dir)
    max_seq_len = model_params.max_seq_len
    model_params = modify_model_params(model_params)
    draw_model = Model(model_params)
    model_params.batch_size = samples_per_category
    model = Model(model_params)

    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
    sess.run(tf.global_variables_initializer())

    utils.load_checkpoint(sess, model_dir)
    al, si = sess.run([model.de_alpha, model.de_sigma2])

    if not os.path.exists(SVG_DIR):
        os.makedirs(SVG_DIR)

    for category in range(len(model_params.categories)):
        raw_data = utils.load_data(data_dir, model_params.categories[category], model_params.num_per_category)
        print(model_params.categories[category])
        _, _, test_set, _ = utils.preprocess_data(raw_data,
                                                  model_params.batch_size,
                                                  model_params.random_scale_factor,
                                                  model_params.augment_stroke_prob,
                                                  model_params.png_scale_ratio,
                                                  model_params.png_rotate_angle,
                                                  model_params.png_translate_dist)

        index = np.arange(2500)
        np.random.shuffle(index)

        # Map the input images to the latent variables
        _, pngs, _, _ = test_set._get_batch_from_indices(index[0:samples_per_category])

        if FLAGS.conditional is True:  # Conditional sampling
            feed = {
                model.input_pngs: pngs
            }
            z = sess.run(model.p_mu, feed)
        else:  # Without references input
            z = np.random.normal(0., 1., [samples_per_category, model_params.z_size])  # Latent codes of sketches you expected to generate

        feed = {
            model.batch_z: z,
            model.input_pngs: pngs
        }
        gau_label = sess.run(model.gau_label, feed)

        # Generate strokes
        for cnt in range(samples_per_category):
            # Generated sketches
            stroke = sample(sess, draw_model, np.reshape(z[cnt, :], [1, -1]), 1, max_seq_len, temperature)
            filepath1 = os.path.join(SVG_DIR, '%d_%d.svg' % (category, cnt))
            # draw_strokes(stroke[0], filepath1, 48, margin=1.5, color='black')
            draw_strokes(stroke[0], filepath1, 48, margin=1.5, color=color[category])
            # Corresponding latent codes
            filepath2 = os.path.join(SVG_DIR, 'code_%d_%d.npy' % (category, cnt))
            np.save(filepath2, np.reshape(z[cnt, :], [1, -1]))
            # Corresponding indexes of the Gaussian components
            filepath4 = os.path.join(SVG_DIR, 'gau_idx_%d_%d.npy' % (category, cnt))
            np.save(filepath4, gau_label[cnt])

    exportsvg(SVG_DIR, SVG_DIR, 'png')

if __name__ == '__main__':
    main()
