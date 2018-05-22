# Copyright 2017 Google Inc.
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

import sys
import os
from tensorboard_logger import configure, log_value
from datetime import datetime

import i3d
from mediadata import LoadItems
from mediadata import LoadInputData

_IMAGE_SIZE = 224
_NUM_CLASSES = 2
num_steps = 5000000
batch_size = 16
max_to_keep = 5

#_SAMPLE_VIDEO_FRAMES = 79
_SAMPLE_VIDEO_FRAMES = 16 # by clong on 2017/11/31.

_CHECKPOINT_PATHS = {
    'rgb': 'data/checkpoints/rgb_scratch/my_model.ckpt',
    'flow': 'data/checkpoints/flow_scratch/my_model.ckpt',
    'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': 'data/checkpoints/flow_imagenet/model.ckpt',
}


FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('eval_type', 'joint', 'rgb, flow, or joint')
tf.flags.DEFINE_boolean('imagenet_pretrained', False, '')
config = tf.ConfigProto(device_count = {'GPU': 1})


def main(test_infname, res_outfname):
  init_learning_rate = 0.01
  k = 500
  decay_rate = 0.96

  test_items = LoadItems(test_infname)
  ntest = len(test_items)

  tf.logging.set_verbosity(tf.logging.INFO)
  eval_type = FLAGS.eval_type
  imagenet_pretrained = FLAGS.imagenet_pretrained

  if eval_type not in ['rgb', 'flow', 'joint']:
    raise ValueError('Bad `eval_type`, must be one of rgb, flow, joint')

  kinetics_classes = [0,1] #[x.strip() for x in open(_LABEL_MAP_PATH)]
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
  # Define loss and optimizer
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model_logits, labels=Y))
  learning_rate = tf.train.inverse_time_decay(init_learning_rate, global_step, k, decay_rate)
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
  train_op = optimizer.minimize(cost, global_step=global_step)

  init = tf.global_variables_initializer()

  checkpoint_dir = './runs/checkpoints/'
  checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
  #saver = tf.train.Saver()
  saver = tf.train.Saver(max_to_keep = max_to_keep)
  #saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)

  resfile = open(res_outfname, 'wt')

  with tf.Session(config=config) as sess:
    sess.run(init)
    feed_dict = {}
    epoch = 0
    step_ival = 0
    step_eval = 0

    start_step = 1
    checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
    if checkpoint_path is None:
        print('The stored model is not exist.')
    else:
        print('checkpoint_path: ' + checkpoint_path)
        elems = checkpoint_path.split('-')
        start_step = int(elems[len(elems)-1]) + 1
        print('sart_step: ' + str(start_step))
        saver.restore(sess, checkpoint_path)

        for k in range(ntest):
            tmp_items = test_items[k:(k+1)]
            
        
            rgbs, flows, tmp_truths = LoadInputData(tmp_items)
            truths = tmp_truths[:, 0:_NUM_CLASSES]
            
            #tf.logging.info('RGB checkpoint restored')
            #tf.logging.info('RGB data loaded, shape=%s', str(rgbs.shape))
            feed_dict[rgb_input] = rgbs
        
            #tf.logging.info('Flow checkpoint restored')
            #tf.logging.info('Flow data loaded, shape=%s', str(flows.shape))
            feed_dict[flow_input] = flows
            feed_dict[Y] = truths

            #sess.run(train_op, feed_dict=feed_dict) 
            logits, predictions = sess.run([model_logits, model_predictions], feed_dict=feed_dict)
            print('test k=' + str(k) + ', logits: {}'.format(logits) + ', predictions:{}'.format(predictions) + ', truths: {}'.format(truths))
            resfile.write(str(k) + ' ' + str(logits[0][0]) + ' ' + str(logits[0][1]) + ' ' + str(predictions[0][0]) + ' ' + str(predictions[0][1]) + ' ' + str(truths[0][1]) + '\n')



    print('Testing Finished.')
    resfile.close()

#    out_logits, out_predictions = sess.run([model_logits, model_predictions], feed_dict=feed_dict)
#
#    print('out_predictions.shape: ')
#    print(out_predictions.shape)
#    
#    out_logits = out_logits[0]
#    out_predictions = out_predictions[0]
#    sorted_indices = np.argsort(out_predictions)[::-1]
#
#    print('Norm of logits: %f' % np.linalg.norm(out_logits))
#    print('\nTop classes and probabilities')
#    for index in sorted_indices[:20]:
#      print(out_predictions[index], out_logits[index], kinetics_classes[index])




if __name__ == '__main__':
  main(sys.argv[1], sys.argv[2])
