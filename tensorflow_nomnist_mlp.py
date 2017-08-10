# -*- coding: utf-8 -*-
"""
Nomnist data classification task
Use multi-layer nn
Use tensorflow

Data are already preprocessed into a pickle file
"""

from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle

pickle_file = 'notMNIST.pickle'
image_size = 28
num_labels = 10

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 1 to [0.0, 1.0, 0.0 ...], 2 to [0.0, 0.0, 1.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])
  
#load data
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

#reformat data
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

# this achieve Minibatch accuracy: 97.7%
#Validation accuracy: 91.1%
#Test accuracy: 96.5%
num_l1_nodes = 1024
batch_size = 128

num_steps = 150001
l2_reg = 0.0001
num_l2_nodes = 64

num_trainingsample = 200000

graph2 = tf.Graph()
with graph2.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  weights_l1 = tf.Variable(
    tf.truncated_normal([image_size * image_size, num_l1_nodes], stddev = np.sqrt(2.0/(image_size * image_size))))
  biases_l1 = tf.Variable(0.1*tf.ones([num_l1_nodes]))

  weights_l2 = tf.Variable(
    tf.truncated_normal([num_l1_nodes, num_l2_nodes], stddev = np.sqrt(2.0/num_l1_nodes)))
  biases_l2 = tf.Variable(0.1*tf.ones([num_l2_nodes]))
    
  weights_l3 = tf.Variable(
    tf.truncated_normal([num_l2_nodes, num_labels], stddev = np.sqrt(2.0/num_l2_nodes)))
  biases_l3 = tf.Variable(0.1*tf.ones([num_labels]))


    
  global_step = tf.Variable(0)  # count the number of steps taken.
  learning_rate = tf.train.exponential_decay(0.1, global_step, 1000, 0.95, staircase=True)
  
  # Training computation.
  def model(data, keep_drop):
      layer1 = tf.nn.relu(tf.matmul(data, weights_l1) + biases_l1)
      layer1_drop = tf.nn.dropout(layer1, keep_drop)
      
      layer2 = tf.nn.relu(tf.matmul(layer1_drop, weights_l2) + biases_l2)
      layer2_drop = tf.nn.dropout(layer2, keep_drop)
      logit = tf.matmul(layer2_drop, weights_l3) + biases_l3
      return logit
  
  logits2 = model(tf_train_dataset, 0.9)
  loss1 = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits2))
  loss2 = l2_reg * (tf.nn.l2_loss(weights_l1) + tf.nn.l2_loss(weights_l2) + tf.nn.l2_loss(weights_l3))
  loss = loss1 + loss2
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step = global_step)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits2)
  valid_prediction = tf.nn.softmax(model(tf_valid_dataset, 1))
  test_prediction = tf.nn.softmax(model(tf_test_dataset, 1))



with tf.Session(graph=graph2) as session:
  tf.global_variables_initializer().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (num_trainingsample - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
    [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      #print("Minibatch loss at step %d: %f, %f" % (step, loss.eval(feed_dict=feed_dict)))
      #print("Minibatch accuracy: %.1f%%, %.1f%%" % accuracy(train_prediction.eval(feed_dict=feed_dict), batch_labels))
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
      print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))