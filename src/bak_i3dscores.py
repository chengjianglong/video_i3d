# Copyright 2018 Kitware Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Loads a sample video and classifies using a trained Kinetics checkpoint."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import time

import sys
import os
import cv2
from tensorboard_logger import configure, log_value
from datetime import datetime

import i3d
from mediadata import LoadItems
from mediadata import LoadInputData

_IMAGE_SIZE = 224
_NUM_CLASSES = 3
num_steps = 5000000
batch_size = 16
max_to_keep = 5

_SAMPLE_VIDEO_FRAMES = 16 

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('eval_type', 'joint', 'rgb, flow, or joint')
tf.flags.DEFINE_boolean('imagenet_pretrained', False, '')
config = tf.ConfigProto(device_count = {'GPU': 2})
config.gpu_options.allow_growth=True

vlen = 16
vlen_half = vlen/2

def generate_i3d_infile(videopath):
    vidname = videopath.split('/')[-1]
    vidname = vidname.split('\n')[0]
    in_i3dfname = vidname.split('.')[0] + '_i3d.txt'
    in_file = open(in_i3dfname, 'w')
    
    print(vidname)

    # access the frames
    cap = cv2.VideoCapture(videopath)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            num_frame += 1
        else:
            break

    print('nframes: ' + str(nframes) + ' , num_frame: ' + str(num_frame))
    if nframes > num_frame:
        nframes = num_frame

    select_list = [1]*nframes
    truth = [0]*nframes


    for i in range(len(select_list)):
        if select_list[i] == 1 and truth[i] == 1 and (i-vlen_half) > 0 and (i+vlen_half)<len(select_list):
            in_file.write(videopath + ' ' + str(int(i-vlen_half)) + ' 1\n')
        elif select_list[i] == 1 and truth[i] == 0 and (i-vlen_half) > 0 and (i+vlen_half)<len(select_list):
            in_file.write(videopath + ' ' + str(int(i-vlen_half)) + ' 0\n')

    in_file.close()

    return in_i3dfname
        


def eval_i3d_comb(test_infname):
  init_learning_rate = 0.01
  k = 500
  decay_rate = 0.96

  parttag = 'ptest'

  test_items = LoadItems(test_infname)
  ntest = len(test_items)

  tf.logging.set_verbosity(tf.logging.INFO)
  eval_type = FLAGS.eval_type
  imagenet_pretrained = FLAGS.imagenet_pretrained

  if eval_type not in ['rgb', 'flow', 'joint']:
    raise ValueError('Bad `eval_type`, must be one of rgb, flow, joint')

  kinetics_classes = [0, 1, 2] #[x.strip() for x in open(_LABEL_MAP_PATH)]
  Y = tf.placeholder('float', [None, _NUM_CLASSES])
  global_step = tf.Variable(0, trainable = False)

  if eval_type in ['rgb', 'joint']:
    # RGB input has 3 channels.
    rgb_input = tf.placeholder(
        tf.float32,
        shape=(None, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))
    with tf.variable_scope('RGB'):
      rgb_model = i3d.InceptionI3d(
          _NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
      _, rgb_logits, _ = rgb_model(rgb_input, is_training=False, dropout_keep_prob=1.0)
    rgb_variable_map = {}
    for variable in tf.global_variables():
      if variable.name.split('/')[0] == 'RGB':
        rgb_variable_map[variable.name.replace(':0', '')] = variable
    rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

  if eval_type in ['flow', 'joint']:
    # Flow input has only 2 channels.
    flow_input = tf.placeholder(
        tf.float32,
        shape=(None, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 2))
    with tf.variable_scope('Flow'):
      flow_model = i3d.InceptionI3d(
          _NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
      _, flow_logits, _ = flow_model(
          flow_input, is_training=False, dropout_keep_prob=1.0)
    flow_variable_map = {}
    for variable in tf.global_variables():
      if variable.name.split('/')[0] == 'Flow':
        flow_variable_map[variable.name.replace(':0', '')] = variable
    flow_saver = tf.train.Saver(var_list=flow_variable_map, reshape=True)

  if eval_type == 'rgb':
    model_logits = rgb_logits
  elif eval_type == 'flow':
    model_logits = flow_logits
  else:
    model_logits = rgb_logits + flow_logits
  model_predictions = tf.nn.softmax(model_logits)
  correct_pred = tf.equal(tf.argmax(model_predictions, 1), tf.argmax(Y, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
  
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model_logits, labels=Y))
  learning_rate = tf.train.inverse_time_decay(init_learning_rate, global_step, k, decay_rate)
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
  train_op = optimizer.minimize(cost, global_step=global_step)

  init = tf.global_variables_initializer()

  saver = tf.train.Saver(max_to_keep = max_to_keep)

  scores = []

  with tf.Session(config=config) as sess:
    sess.run(init)
    feed_dict = {}
    epoch = 0
    step_ival = 0
    step_eval = 0

    start_step = 1
    checkpoint_path = 'i3d-iter-312400'
    if checkpoint_path is None:
        print('The stored model is not exist.')
    else:
        print('checkpoint_path: ' + checkpoint_path)
        elems = checkpoint_path.split('-')
        start_step = int(elems[len(elems)-1]) + 1
        print('sart_step: ' + str(start_step))
        saver.restore(sess, checkpoint_path)

        batchsize = 32 #32
        k = 0

        while k*batchsize < ntest:
            time0 = time.time()

            start_id = k*batchsize
            end_id = (k+1)*batchsize
            if end_id > ntest:
                end_id = ntest

            tmp_items = test_items[start_id:end_id]
            
        
            rgbs, flows, tmp_truths = LoadInputData(tmp_items, parttag)
            truths = tmp_truths
            
#            print('rgbs.shape: {}'.format(rgbs.shape))
#            print('flows.shape: {}'.format(flows.shape))
#            print('Y.shape: {}'.format(truths.shape))
            feed_dict[rgb_input] = rgbs
            feed_dict[flow_input] = flows
            feed_dict[Y] = truths

            time1 = time.time()
            logits, predictions = sess.run([model_logits, model_predictions], feed_dict=feed_dict)
            
            for i in range(logits.shape[0]):
                scores.append([logits[i][0], logits[i][1], logits[i][2],  predictions[i][0], predictions[i][1], predictions[i][2]])
            
            time2 = time.time()

            print('test batch k=' + str(k) + ', logits: {}'.format(logits.shape) + ', time = {}'.format([time1-time0, time2-time1, time2-time0])) # + ', predictions:{}'.format(predictions) + ', truths: {}'.format(truths))

            k += 1

    #sess.run(init)
  
  print('Testing Finished.')
  os.system('rm -rf ' + test_infname)
  os.system('rm -rf tmp_' + parttag)

  return scores


def main(videopath):
    infilename = generate_i3d_infile(videopath)
    scores = eval_i3d_comb(infilename)
    print('scores: {}'.format(scores))


if __name__ == '__main__':
  main(sys.argv[1])
