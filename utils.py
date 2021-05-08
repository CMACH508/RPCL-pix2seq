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
"""RPCL-pix2seq data loading and image manipulation utilities."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import tensorflow as tf
import numpy as np
import os
import math
import cv2
import six

def get_default_hparams():
    """ Return default and initial HParams """
    hparams = tf.contrib.training.HParams(
        categories=['bee', 'bus', 'flower', 'giraffe', 'pig'],  # Sketch categories
        num_steps=1000001,  # Number of total steps (the process will stop automatically if the loss is not improved)
        save_every=1,  # Number of epochs before saving model
        dec_rnn_size=2048,  # Size of decoder
        dec_model='hyper',  # Decoder: lstm, layer_norm or hyper
        max_seq_len=-1,  # Max sequence length. Computed by DataLoader
        z_size=128,  # Size of latent variable
        batch_size=200,  # Minibatch size
        num_mixture=5,  # Recommend to set to the number of categories
        learning_rate=0.001,  # Learning rate
        decay_rate=0.9999,  # Learning rate decay per minibatch.
        min_learning_rate=0.00001,  # Minimum learning rate
        grad_clip=1.,  # Gradient clipping
        de_weight=0.5,  # Weight for deconv loss
        use_recurrent_dropout=True,  # Dropout with memory loss
        recurrent_dropout_prob=0.90,  # Probability of recurrent dropout keep
        use_input_dropout=False,  # Input dropout
        input_dropout_prob=0.90,  # Probability of input dropout keep
        use_output_dropout=False,  # Output droput
        output_dropout_prob=0.9,  # Probability of output dropout keep
        random_scale_factor=0.10,  # Random scaling data augmention proportion
        augment_stroke_prob=0.10,  # Point dropping augmentation proportion
        png_scale_ratio=0.98,  # Min scaling ratio
        png_rotate_angle=0,  # Max rotating angle (abs value)
        png_translate_dist=0,  # Max translating distance (abs value)
        is_training=True,  # Training mode or not
        png_width=48,  # Width of input images
        num_sub=2,  # Number of components for each category
        num_per_category=70000  # Training samples from each category
    )
    return hparams

def copy_hparams(hparams):
  """ Return a copy of an HParams instance """
  return tf.contrib.training.HParams(**hparams.values())


def reset_graph():
    """ Close the current default session and resets the graph """
    sess = tf.get_default_session()
    if sess:
      sess.close()
    tf.reset_default_graph()

def load_seqs(data_dir, categories):
    """ Load sequence raw data """
    if not isinstance(categories, list):
        categories = [categories]

    train_seqs = None
    valid_seqs = None
    test_seqs = None
    
    for ctg in categories:
        # load sequence data
        seq_path = os.path.join(data_dir, ctg + '.npz')
        if six.PY3:
            seq_data = np.load(seq_path, encoding='latin1', allow_pickle=True)
        else:
            seq_data = np.load(seq_path, allow_pickle=True)
        tf.logging.info('Loaded sequences {}/{}/{} from {}'.format(
            len(seq_data['train']), len(seq_data['valid']), len(seq_data['test']),
            ctg + '.npz'))

        if train_seqs is None:
            train_seqs = seq_data['train']
            valid_seqs = seq_data['valid']
            test_seqs = seq_data['test']
        else:
            train_seqs = np.concatenate((train_seqs, seq_data['train']))
            valid_seqs = np.concatenate((valid_seqs, seq_data['valid']))
            test_seqs = np.concatenate((test_seqs, seq_data['test']))

    return train_seqs, valid_seqs, test_seqs


def load_data(data_dir, categories, num_per_category):
    """ Load sequence and image raw data """
    if not isinstance(categories, list):
        categories = [categories]

    train_seqs = None
    train_pngs = None
    valid_seqs = None
    valid_pngs = None
    test_seqs = None
    test_pngs = None
    train_labels = None
    valid_labels = None
    test_labels = None

    i = 0
    for ctg in categories:
        # load sequence data
        seq_path = os.path.join(data_dir, ctg + '.npz')
        if six.PY3:
            seq_data = np.load(seq_path, encoding='latin1', allow_pickle=True)
        else:
            seq_data = np.load(seq_path, allow_pickle=True)
        tf.logging.info('Loaded sequences {}/{}/{} from {}'.format(
            len(seq_data['train']), len(seq_data['valid']), len(seq_data['test']),
            ctg + '.npz'))

        if train_seqs is None:
            train_seqs = seq_data['train'][0:num_per_category]
            valid_seqs = seq_data['valid'][0:num_per_category]
            test_seqs = seq_data['test'][0:num_per_category]
        else:
            train_seqs = np.concatenate((train_seqs, seq_data['train'][0:num_per_category]))
            valid_seqs = np.concatenate((valid_seqs, seq_data['valid'][0:num_per_category]))
            test_seqs = np.concatenate((test_seqs, seq_data['test'][0:num_per_category]))

        # load png data
        png_path = os.path.join(data_dir, ctg + '_png.npz')
        if six.PY3:
            png_data = np.load(png_path, encoding='latin1', allow_pickle=True)
        else:
            png_data = np.load(png_path, allow_pickle=True)
        tf.logging.info('Loaded pngs {}/{}/{} from {}'.format(
            len(png_data['train']), len(png_data['valid']), len(png_data['test']),
            ctg + '_png.npz'))

        if train_pngs is None:
            train_pngs = png_data['train'][0:num_per_category]
            valid_pngs = png_data['valid'][0:num_per_category]
            test_pngs = png_data['test'][0:num_per_category]
        else:
            train_pngs = np.concatenate((train_pngs, png_data['train'][0:num_per_category]))
            valid_pngs = np.concatenate((valid_pngs, png_data['valid'][0:num_per_category]))
            test_pngs = np.concatenate((test_pngs, png_data['test'][0:num_per_category]))

        # create labels
        if train_labels is None:
            train_labels = i * np.ones([num_per_category], dtype=np.int)
            valid_labels = i * np.ones([num_per_category], dtype=np.int)
            test_labels = i * np.ones([num_per_category], dtype=np.int)
        else:
            train_labels = np.concatenate([train_labels, i * np.ones([num_per_category], dtype=np.int)])
            valid_labels = np.concatenate([valid_labels, i * np.ones([num_per_category], dtype=np.int)])
            test_labels = np.concatenate([test_labels, i * np.ones([num_per_category], dtype=np.int)])
        i += 1

    return [train_seqs, valid_seqs, test_seqs,
            train_pngs, valid_pngs, test_pngs,
            train_labels, valid_labels, test_labels]


def preprocess_data(raw_data, batch_size, random_scale_factor, augment_stroke_prob, png_scale_ratio, png_rotate_angle, png_translate_dist):
    """ Convert raw data to suitable model inputs """
    train_seqs, valid_seqs, test_seqs, train_pngs, valid_pngs, test_pngs, train_labels, valid_labels, test_labels = raw_data
    all_strokes = np.concatenate((train_seqs, valid_seqs, test_seqs))
    max_seq_len = get_max_len(all_strokes)

    train_set = DataLoader(
        train_seqs,
        train_pngs,
        train_labels,
        batch_size,
        max_seq_length=max_seq_len,
        random_scale_factor=random_scale_factor,
        augment_stroke_prob=augment_stroke_prob,
        png_scale_ratio=png_scale_ratio,
        png_rotate_angle=png_rotate_angle,
        png_translate_dist=png_translate_dist)
    seq_norm = train_set.calc_seq_norm()
    train_set.normalize_seq(seq_norm)

    valid_set = DataLoader(
        valid_seqs,
        valid_pngs,
        valid_labels,
        batch_size,
        max_seq_length=max_seq_len)
    valid_set.normalize_seq(seq_norm)

    test_set = DataLoader(
        test_seqs,
        test_pngs,
        test_labels,
        batch_size,
        max_seq_length=max_seq_len)
    test_set.normalize_seq(seq_norm)

    tf.logging.info('normalizing_scale_factor %4.4f.', seq_norm)
    return train_set, valid_set, test_set, max_seq_len


def load_checkpoint(sess, checkpoint_path):
    """ Load checkpoint of saved model """
    saver = tf.train.Saver(tf.global_variables())

    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    tf.logging.info('Loading model %s.', ckpt.model_checkpoint_path)
    print('Loading model %s.', ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)


def save_model(sess, model_save_path, global_step):
    """ Save model """
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
    
    checkpoint_path = os.path.join(model_save_path, 'vector')
    tf.logging.info('saving model %s.', checkpoint_path)
    tf.logging.info('global_step %i.', global_step)
    saver.save(sess, checkpoint_path, global_step=global_step)


def summ_content(tag, val):
    """ Construct summary content """
    summ = tf.summary.Summary()
    summ.value.add(tag=tag, simple_value=float(val))
    return summ


def write_summary(summ_writer, summ_dict, step):
    """ Write summary """
    for key, val in summ_dict.iteritems():
        summ_writer.add_summary(summ_content(key, val), step)
    summ_writer.flush()


def augment_strokes(strokes, prob=0.0):
    """ Perform data augmentation by randomly dropping out strokes """
    # drop each point within a line segments with a probability of prob
    # note that the logic in the loop prevents points at the ends to be dropped.
    result = []
    prev_stroke = [0, 0, 1]
    count = 0
    stroke = [0, 0, 1]  # Added to be safe.
    for i in range(len(strokes)):
        candidate = [strokes[i][0], strokes[i][1], strokes[i][2]]
        if candidate[2] == 1 or prev_stroke[2] == 1:
            count = 0
        else:
            count += 1
        urnd = np.random.rand()  # uniform random variable
        if candidate[2] == 0 and prev_stroke[2] == 0 and count > 2 and urnd < prob:
            stroke[0] += candidate[0]
            stroke[1] += candidate[1]
        else:
            stroke = list(candidate)
            prev_stroke = list(stroke)
            result.append(stroke)
    return np.array(result)


def seq_3d_to_5d(stroke, max_len=250):
    """ Convert from 3D format (npz file) to 5D (sketch-rnn paper) """
    result = np.zeros((max_len, 5), dtype=float)
    l = len(stroke)
    assert l <= max_len
    result[0:l, 0:2] = stroke[:, 0:2]
    result[0:l, 3] = stroke[:, 2]
    result[0:l, 2] = 1 - result[0:l, 3]
    result[l:, 4] = 1
    return result


def seq_5d_to_3d(big_stroke):
    """ Convert from 5D format (sketch-rnn paper) back to 3D (npz file) """
    l = 0 # the total length of the drawing
    for i in range(len(big_stroke)):
        if big_stroke[i, 4] > 0:
            l = i
            break
        if l == 0:
            l = len(big_stroke) # restrict the max total length of drawing to be the length of big_stroke
    result = np.zeros((l, 3))
    result[:, 0:2] = big_stroke[0:l, 0:2]
    result[:, 2] = big_stroke[0:l, 3]
    return result # stroke-3


def get_max_len(strokes):
    """ Return the maximum length of an array of strokes """
    max_len = 0
    for stroke in strokes:
        ml = len(stroke)
        if ml > max_len:
            max_len = ml
    return max_len


def rescale(X, ratio=0.85):
    """ Rescale the image to a smaller size """
    h, w = X.shape

    h2 = int(h*ratio)
    w2 = int(w*ratio)

    X2 = cv2.resize(X, (w2, h2), interpolation=cv2.INTER_AREA)

    dh = int((h - h2) / 2)
    dw = int((w - w2) / 2)

    res = np.copy(X)
    res[:,:] = 1
    res[dh:(dh+h2),dw:(dw+w2)] = X2

    return res


def rotate(X, angle=15):
    """ Rotate the image """
    h, w = X.shape
    rad = np.deg2rad(angle)

    nw = ((abs(np.sin(rad)*h)) + (abs(np.cos(rad)*w)))
    nh = ((abs(np.cos(rad)*h)) + (abs(np.sin(rad)*w)))

    rot_mat = cv2.getRotationMatrix2D((nw/2,nh/2),angle,1)
    rot_move = np.dot(rot_mat,np.array([(nw-w)/2,(nh-h)/2,0]))

    rot_mat[0,2] += rot_move[0]
    rot_mat[1,2] += rot_move[1]

    res_w = int(math.ceil(nw))
    res_h = int(math.ceil(nh))

    res = cv2.warpAffine(X,rot_mat,(res_w,res_h),flags=cv2.INTER_LANCZOS4, borderValue=1)
    res = cv2.resize(res,(w,h), interpolation=cv2.INTER_AREA)

    return res


def translate(X, dx=5,dy=5):
    """ Translate the image """
    h, w = X.shape
    M = np.float32([[1,0,dx],[0,1,dy]])
    res = cv2.warpAffine(X,M,(w,h), borderValue=1)

    return res


class DataLoader(object):
    """ Class for loading data from raw data (sequence and image) """

    def __init__(self,
               strokes,
               images,
               labels,
               batch_size=100,
               max_seq_length=250,
               scale_factor=1.0,
               random_scale_factor=0.0,
               augment_stroke_prob=0.0,
               png_scale_ratio=1,
               png_rotate_angle=0,
               png_translate_dist=0,
               limit=1000):
        self.batch_size = batch_size  # minibatch size
        self.max_seq_length = max_seq_length  # N_max in sketch-rnn paper
        self.scale_factor = scale_factor  # divide data by this factor
        self.random_scale_factor = random_scale_factor  # data augmentation method
        self.limit = limit  # removes large gaps in the data
        self.augment_stroke_prob = augment_stroke_prob  # data augmentation method
        self.png_scale_ratio=png_scale_ratio  # min randomly scaled ratio
        self.png_rotate_angle=png_rotate_angle  # max randomly rotate angle (in absolute value)
        self.png_translate_dist=png_translate_dist  # max randomly translate distance (in absolute value)
        self.start_stroke_token = [0, 0, 1, 0, 0]  # S_0 in sketch-rnn paper
        self.preprocess(strokes, images)
        self.labels = labels

    def preprocess(self, strokes, images):
        # preprocess stroke data
        self.strokes = []
        count_data = 0 # the number of drawing with length less than N_max

        for i in range(len(strokes)):
            data = np.copy(strokes[i])
            if len(data) <= self.max_seq_length:    # keep data with length less than N_max
                count_data += 1
                # removes large gaps from the data
                data = np.minimum(data, self.limit)     # prevent large values
                data = np.maximum(data, -self.limit)    # prevent small values
                data = np.array(data, dtype=np.float32) # change data type
                data[:, 0:2] /= self.scale_factor       # scale the first two dims of data
                self.strokes.append(data)

        print("total sequences <= max_seq_len is %d" % count_data)
        self.num_batches = int(count_data / self.batch_size)

        # preprocess image data
        self.images = []
        for i in range(len(images)):
            data = np.copy(images[i])
            data = data / 255.0 * 2.0 - 1.0 # [0,255] => [0,1]
            self.images.append(data)

        print("total png images %d" % len(self.images))


    def random_sample(self):
        """ Return a random sample (3D stroke, png image) """
        l = len(self.strokes)
        idx = np.random.randint(0,l)
        seq = self.strokes[idx]
        png = self.images[idx]
        label = self.labels[idx]
        png = png.reshape((1,png.shape[0],png.shape[1]))
        return seq, png, label


    def idx_sample(self, idx):
        """ Return one sample by idx """
        data = self.random_scale_seq(self.strokes[idx])
        if self.augment_stroke_prob > 0:
            data = augment_strokes(data, self.augment_stroke_prob)
        strokes_3d = data
        strokes_5d = seq_3d_to_5d(strokes_3d,self.max_seq_length)

        data = np.copy(self.images[idx])
        png = np.reshape(data, [1,data.shape[0],data.shape[1]])
        png = self.random_scale_png(png)
        png = self.random_rotate_png(png)
        png = self.random_translate_png(png)
        label = self.labels[idx]
        return strokes_5d, png, label


    def random_scale_seq(self, data):
        """ Augment data by stretching x and y axis randomly [1-e, 1+e] """
        x_scale_factor = (np.random.random() - 0.5) * 2 * self.random_scale_factor + 1.0
        y_scale_factor = (np.random.random() - 0.5) * 2 * self.random_scale_factor + 1.0
        result = np.copy(data)
        result[:, 0] *= x_scale_factor
        result[:, 1] *= y_scale_factor
        return result


    def random_scale_png(self, data):
        """ Randomly scale image """
        out_pngs = np.copy(data)
        for i in range(data.shape[0]):
            in_png = data[i]
            ratio = random.uniform(self.png_scale_ratio,1)
            out_png = rescale(in_png, ratio)
            out_pngs[i] = out_png
        return out_pngs


    def random_rotate_png(self, data):
        """ Randomly rotate image """
        out_pngs = np.copy(data)
        for i in range(data.shape[0]):
            in_png = data[i]
            angle = random.uniform(-self.png_rotate_angle,self.png_rotate_angle)
            out_png = rotate(in_png, angle)
            out_pngs[i] = out_png
        return out_pngs


    def random_translate_png(self, data):
        """ Randomly translate image """
        out_pngs = np.copy(data)
        for i in range(data.shape[0]):
            in_png = data[i]
            dx = random.uniform(-self.png_translate_dist,self.png_translate_dist)
            dy = random.uniform(-self.png_translate_dist,self.png_translate_dist)
            out_png = translate(in_png, dx, dy)
            out_pngs[i] = out_png
        return out_pngs


    def calc_seq_norm(self):
        """ Calculate the normalizing factor explained in appendix of sketch-rnn """
        data = []
        for i in range(len(self.strokes)):
            if len(self.strokes[i]) > self.max_seq_length:
                continue
            for j in range(len(self.strokes[i])):
                data.append(self.strokes[i][j, 0])
                data.append(self.strokes[i][j, 1])
        data = np.array(data)
        return np.std(data) # standard dev of all the delta x and delta y in the datasets

    def normalize_seq(self, scale_factor=None):
        """ Normalize entire sequence dataset (delta_x, delta_y) by the scaling factor."""
        if scale_factor is None:
            scale_factor = self.calc_seq_norm()
        self.scale_factor = scale_factor
        for i in range(len(self.strokes)):
            self.strokes[i][:, 0:2] /= self.scale_factor


    def _get_batch_from_indices(self, indices):
        """Given a list of indices, return the potentially augmented batch."""
        seq_batch = []
        png_batch = []
        label_batch = []
        seq_len = []
        for idx in range(len(indices)):
            i = indices[idx]
            data = self.random_scale_seq(self.strokes[i])
            data_copy = np.copy(data)
            if self.augment_stroke_prob > 0:
                data_copy = augment_strokes(data_copy, self.augment_stroke_prob)
            seq_batch.append(data_copy)
            length = len(data_copy)
            seq_len.append(length)
            png_batch.append(self.images[i])
            label_batch.append(self.labels[i])

        seq_len = np.array(seq_len, dtype=int)

        png_batch = np.array(png_batch)
        png_batch = self.random_scale_png(png_batch)
        png_batch = self.random_rotate_png(png_batch)
        png_batch = self.random_translate_png(png_batch)
        seq_len = np.array(seq_len, dtype=int)
        return self.pad_seq_batch(seq_batch, self.max_seq_length), png_batch, label_batch, seq_len


    def random_batch(self):
        """Return a randomised portion of the training data."""
        idxs = np.random.permutation(list(range(0, len(self.strokes))))[0:self.batch_size]
        return self._get_batch_from_indices(idxs)


    def get_batch(self, idx):
        """Get the idx'th batch from the dataset."""
        assert idx >= 0, "idx must be non negative"
        assert idx < self.num_batches, "idx must be less than the number of batches"
        start_idx = idx * self.batch_size
        indices = list(range(start_idx, start_idx + self.batch_size))
        return self._get_batch_from_indices(indices)


    def pad_seq_batch(self, batch, max_len):
      """ Pad the batch to be 5D format, and fill the sequence to reach max_len """
      result = np.zeros((self.batch_size, max_len + 1, 5), dtype=float)
      assert len(batch) == self.batch_size
      for i in range(self.batch_size):
          l = len(batch[i])
          assert l <= max_len
          result[i, 0:l, 0:2] = batch[i][:, 0:2]
          result[i, 0:l, 3] = batch[i][:, 2]
          result[i, 0:l, 2] = 1 - result[i, 0:l, 3]
          result[i, l:, 4] = 1
          # put in the first token, as described in sketch-rnn methodology
          result[i, 1:, :] = result[i, :-1, :]
          result[i, 0, :] = 0
          result[i, 0, 2] = self.start_stroke_token[2]  # setting S_0 from paper.
          result[i, 0, 3] = self.start_stroke_token[3]
          result[i, 0, 4] = self.start_stroke_token[4]
      return result