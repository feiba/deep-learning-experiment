# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Wed May 31 11:28:02 2017

@author: winson
"""
"""
Nomnist data classification task
Use 2 conv-relu-maxpool followed by 2 fully connected layer nn
Use tensorflow

Data are already preprocessed into a pickle file
"""

"""
Minibatch loss at step 50000: 0.854693
Minibatch accuracy: 92.2%
Validation accuracy: 91.9%
Test accuracy: 96.8%
"""

import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import numpy as np

def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

pickle_file = 'notMNIST.pickle'
image_size = 28
num_labels = 10
num_channels = 1 # grayscale

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)
  

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


batch_size = 64
patch_size = 5
depth1 = 16
depth2 = 32
num_hidden1 = 512
num_hidden2 = 64
l2_reg = 0.001

graph = tf.Graph()

with graph.as_default():

  # Input data.
  tf_train_dataset = tf.placeholder(
    tf.float32, shape=(batch_size, image_size, image_size, num_channels))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  layer1_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, num_channels, depth1], stddev=0.1))
  layer1_biases = tf.Variable(tf.zeros([depth1]))
  
  layer2_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, depth1, depth2], stddev=0.1))
  layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth2]))
  
  layer3_weights = tf.Variable(tf.truncated_normal(
      [image_size // 4 * image_size // 4 * depth2, num_hidden1], stddev=0.1))
  layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden1]))
  
  layer4_weights = tf.Variable(tf.truncated_normal(
      [num_hidden1, num_hidden2], stddev=0.1))
  layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden2]))
  
  layer5_weights = tf.Variable(tf.truncated_normal(
      [num_hidden2, num_labels], stddev=0.1))
  layer5_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
  
  # Model.
  def model(data, keep_drop):
    conv1 = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
    hidden1 = tf.nn.relu(conv1 + layer1_biases)
    hidden1 = tf.nn.max_pool(hidden1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    
    conv2 = tf.nn.conv2d(hidden1, layer2_weights, [1, 1, 1, 1], padding='SAME')
    hidden2 = tf.nn.relu(conv2 + layer2_biases)
    hidden2 = tf.nn.max_pool(hidden2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    
    shape = hidden2.get_shape().as_list()
    reshape = tf.reshape(hidden2, [shape[0], shape[1] * shape[2] * shape[3]])
    hidden3 = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
    hidden3_drop = tf.nn.dropout(hidden3, keep_drop)
    
    hidden4 = tf.nn.relu(tf.matmul(hidden3_drop, layer4_weights) + layer4_biases)
    
    logit = tf.matmul(hidden4, layer5_weights) + layer5_biases
    return logit

  # Training computation.
  logits = model(tf_train_dataset, 0.9)
  loss1 = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
  loss2 = l2_reg * (tf.nn.l2_loss(layer1_weights) + 
                    tf.nn.l2_loss(layer2_weights) + 
                    tf.nn.l2_loss(layer3_weights) +
                    tf.nn.l2_loss(layer4_weights) +
                    tf.nn.l2_loss(layer5_weights))
  
  loss = loss1 + loss2
  
  global_step = tf.Variable(0)  # count the number of steps taken.
  learning_rate = tf.train.exponential_decay(0.05, global_step, 1000, 0.95, staircase=True)  
  # Training computation.
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step = global_step)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(model(tf_valid_dataset, 1))
  test_prediction = tf.nn.softmax(model(tf_test_dataset, 1))


num_steps = 50001

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print('Initialized')
  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print('Minibatch loss at step %d: %f' % (step, l))
      print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
      print('Validation accuracy: %.1f%%' % accuracy(
        valid_prediction.eval(), valid_labels))
      print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))