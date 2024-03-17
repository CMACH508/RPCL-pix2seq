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
#
#       [1] https://github.com/tensorflow/magenta/tree/master/magenta/models/sketch_rnn
#
""" convert sequence data to png file """
import numpy as np
import six
import svgwrite  # conda install -c omnia svgwrite=1.1.6
import os
import tensorflow as tf
import svg2png
import glob
from PIL import Image, ImageDraw
import re
import shutil

def get_bounds(data):
  """Return bounds of data."""
  min_x = 0
  max_x = 0
  min_y = 0
  max_y = 0

  abs_x = 0
  abs_y = 0
  for i in range(len(data)):
    x = float(data[i, 0])
    y = float(data[i, 1])
    abs_x += x
    abs_y += y
    min_x = min(min_x, abs_x)
    min_y = min(min_y, abs_y)
    max_x = max(max_x, abs_x)
    max_y = max(max_y, abs_y)

  return (min_x, max_x, min_y, max_y)


def load_dataset(data_dir, dataset):
  """ fetch data from npz file """
  train_strokes = None
  valid_strokes = None
  test_strokes = None

  data_filepath = os.path.join(data_dir, dataset)

  if six.PY3:
      data = np.load(data_filepath, encoding='latin1')
  else:
      data = np.load(data_filepath)

  if train_strokes is None:
      train_strokes = data['train']
      valid_strokes = data['valid']
      test_strokes = data['test']
  else:
      train_strokes = np.concatenate((train_strokes, data['train']))
      valid_strokes = np.concatenate((valid_strokes, data['valid']))
      test_strokes = np.concatenate((test_strokes, data['test']))

  return train_strokes,valid_strokes,test_strokes


def draw_strokes(data, svg_filename='sample.svg', width=48, margin=1.5, color='black'):
    """ convert sequence data to svg format """
    min_x, max_x, min_y, max_y = get_bounds(data)
    if max_x - min_x > max_y - min_y:
        norm = max_x - min_x
        border_y = (norm - (max_y - min_y)) * 0.5
        border_x = 0
    else:
        norm = max_y - min_y
        border_x = (norm - (max_x - min_x)) * 0.5
        border_y = 0
  
    # normalize data
    norm = max(norm, 10e-6)
    scale = (width - 2*margin) / norm
    dx = 0 - min_x + border_x
    dy = 0 - min_y + border_y
  
    abs_x = (0 + dx) * scale + margin
    abs_y = (0 + dy) * scale + margin
  
    # start converting
    dwg = svgwrite.Drawing(svg_filename, size=(width,width))
    dwg.add(dwg.rect(insert=(0, 0), size=(width,width),fill='white'))
    lift_pen = 1
    p = "M%s,%s " % (abs_x, abs_y)
    command = "m"
    for i in range(len(data)):
        if (lift_pen == 1):
            command = "m"
        elif (command != "l"):
            command = "l"
        else:
            command = ""
        x = float(data[i,0]) * scale
        y = float(data[i,1]) * scale
        lift_pen = data[i, 2]
        p += command+str(x)+","+str(y)+" "
    the_color = color  # "black"
    stroke_width = 1
    dwg.add(dwg.path(p).stroke(the_color,stroke_width).fill("none"))
    dwg.save()

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

    # Input sequential-formed dataset directory
    tf.app.flags.DEFINE_string(
        'input_dir',
        'dataset_path',
        'The directory in which to find the original dataset.'
    )
    # Output pixel-formed dataset directory
    tf.app.flags.DEFINE_string(
        'output_dir',
        'output_path',
        'The directory in which to output the translated dataset.'
    )
    tf.app.flags.DEFINE_integer(
        'png_width', 48,
        'The width of the output pixel-formed sketch image.'
    )
    # Category to be translated
    tf.app.flags.DEFINE_multi_string(
        'categories', {'cat','pig'},
        'The sketch category to be translated to the pixel form from the sequential form.'
    )

    dataset = FLAGS.categories
    in_dir = FLAGS.input_dir
    out_dir = FLAGS.output_dir

    # seq2svg
    for category in range(len(dataset)):
        train_strokes, valid_strokes, test_strokes = load_dataset(in_dir, dataset[category]+'.npz')

        print('finish loading files')
        out_path = os.path.join(out_dir, dataset[category])
        if os.path.exists(out_path) is False:
            os.makedirs(out_path)
        svg_path = os.path.join(out_path, 'svg')
        if os.path.exists(svg_path) is False:
            os.makedirs(svg_path)

        train_path = os.path.join(svg_path, 'train')
        valid_path = os.path.join(svg_path, 'valid')
        test_path = os.path.join(svg_path, 'test')
        if os.path.exists(train_path) is False:
            os.makedirs(train_path)
        if os.path.exists(valid_path) is False:
            os.makedirs(valid_path)
        if os.path.exists(test_path) is False:
            os.makedirs(test_path)

        for i, stroke in enumerate(train_strokes):
            img_path = os.path.join(train_path, '%d.svg' % i)
            draw_strokes(stroke, img_path, width=FLAGS.png_width)
            if i % 100 == 0:
                print('handled train %d' % i)

        for i, stroke in enumerate(valid_strokes):
            img_path = os.path.join(valid_path, '%d.svg' % i)
            draw_strokes(stroke, img_path, width=FLAGS.png_width)
            if i % 100 == 0:
                print('handled valid %d' % i)

        for i, stroke in enumerate(test_strokes):
            img_path = os.path.join(test_path, '%d.svg' % i)
            draw_strokes(stroke, img_path, width=FLAGS.png_width)
            if i % 100 == 0:
                print('handled test %d' % i)

        # svg2png
        png_path = os.path.join(out_path, 'png')
        if os.path.exists(png_path) is False:
            os.makedirs(png_path)
        temp = ['train/', 'test/', 'valid/']
        for i in range(3):
            svg2png.main(os.path.join(svg_path, temp[i]), os.path.join(png_path, temp[i]))

            # save png data
            image_paths = glob.glob(os.path.join(png_path, temp[i], '*.png'))
            image_paths = sort_paths(image_paths)
            print(len(image_paths))
            for cnt, img_path in enumerate(image_paths):
                img = Image.open(img_path, 'r').convert('L')  # covert to grayscale
                img_data = np.array(img)
                img_data = img_data.reshape([1, img_data.shape[0], img_data.shape[1]])
                if cnt == 0:
                    pngs = img_data
                else:
                    pngs = np.concatenate((pngs, img_data), axis=0)
            if i == 0:
                train_pngs = pngs
            elif i == 1:
                test_pngs = pngs
            elif i == 2:
                valid_pngs = pngs

        np.savez(out_path + "_png", train=train_pngs, valid=valid_pngs, test=test_pngs)
        shutil.rmtree(out_path)

if __name__ == "__main__":
    main()
