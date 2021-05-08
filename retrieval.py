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
""" Calculate Ret."""

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
from sample import sample
import scipy.misc
import re


def load_model_params(model_dir):
    model_params = utils.get_default_hparams()
    with tf.gfile.Open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
        model_config = json.dumps(json.load(f))
        model_params.parse_json(model_config)
    return model_params


def modify_model_params(model_params):
    model_params.use_input_dropout = 0
    model_params.use_recurrent_dropout = 0
    model_params.use_output_dropout = 0
    model_params.is_training = False
    model_params.batch_size = 1
    model_params.max_seq_len = 1

    return model_params

def sort_paths(paths):
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
    # Checkpoint directory
    tf.app.flags.DEFINE_string(
        'model_dir', 'checkpoint',
        'Directory to store the model checkpoints.'
    )
    # Sample directory
    tf.app.flags.DEFINE_string(
        'output_dir', 'sample',
        'Directory to store the generated sketches.'
    )

    model_dir = FLAGS.model_dir
    SVG_DIR = FLAGS.output_dir

    model_params = load_model_params(model_dir)
    model_params = modify_model_params(model_params)
    model = Model(model_params)

    for label in range(len(model_params.categories)):
        img_paths = glob.glob(SVG_DIR + '/%d_*.png' % label)
        code_paths = glob.glob(SVG_DIR + '/code_%d_*.npy' % label)
        img_paths = sort_paths(img_paths)
        code_paths = sort_paths(code_paths)
        if label == 0:
            img = np.array(img_paths)
            code = np.array(code_paths)
        else:
            img = np.hstack((img, np.array(img_paths)))
            code = np.hstack((code, np.array(code_paths)))

    img_data = []
    for path in img:
        im = np.array(Image.open(path).convert(mode='RGB'))
        im = im[:, :, 0] / 255. * 2. - 1.
        im = np.reshape(im, [1, 48, 48])
        if img_data == []:
            img_data = im
        else:
            img_data = np.concatenate([img_data, im], axis=0)

    code_data = []
    for path in code:
        code_data.append(np.load(path))
    code_data = np.reshape(code_data, [-1, model_params.z_size])  # Real codes for original sketches

    sample_size = len(code_data)  # Number of samples for retrieval

    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
    sess.run(tf.global_variables_initializer())
    utils.load_checkpoint(sess, model_dir)

    for i in range(len(img_data[:, 0, 0])):
        feed = {
            model.input_pngs: np.reshape(img_data[i, :, :], [1, 48, 48])
        }
        z = sess.run(model.p_mu, feed)  # Codes of the samples

        if i == 0:
            batch_z = z
        else:
            batch_z = np.concatenate([batch_z, z], axis=0)  # Codes for generations

    # Begin retrieval
    top_1 = 0.
    top_10 = 0.
    top_50 = 0.

    temp_sample_size = int(sample_size / 10)
    for ii in range(10):  # reduce the risk of memory out
        real_code = np.tile(np.reshape(code_data, [sample_size, 1, model_params.z_size]), [1, temp_sample_size, 1])
        fake_code = np.tile(np.reshape(batch_z[temp_sample_size * ii:temp_sample_size * (ii + 1), :],
                                       [1, temp_sample_size, model_params.z_size]), [sample_size, 1, 1])
        distances = np.average((real_code - fake_code) ** 2,
                               axis=2)  # Distances between each two codes, sample_size * sample_size

        for n in range(50):
            temp_index = np.argmin(distances, axis=0)
            for i in range(temp_sample_size):
                distances[temp_index[i], i] = 1e10
            if n == 0:
                top_n_index = np.reshape(temp_index, [1, -1])
            else:
                top_n_index = np.concatenate([top_n_index, np.reshape(temp_index, [1, -1])], axis=0)

        for i in range(temp_sample_size):
            if top_n_index[0, i] == i + temp_sample_size * ii:
                top_1 += 1.
            for k in range(10):
                if top_n_index[k, i] == i + temp_sample_size * ii:
                    top_10 += 1.
                    break
            for k in range(50):
                if top_n_index[k, i] == i + temp_sample_size * ii:
                    top_50 += 1.
                    break

    print("Top 1 Ret: " + str(float(top_1 / sample_size)))
    print("Top 10 Ret: " + str(float(top_10 / sample_size)))
    print("Top 50 Ret: " + str(float(top_50 / sample_size)))

if __name__ == '__main__':
    main()
