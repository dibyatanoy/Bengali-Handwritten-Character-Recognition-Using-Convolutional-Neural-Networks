# Implementation of a neural network with 2 hidden layers in addition to an input and an output layer
# Dropout and L2-regularisation are added to minimize overfitting. Stochastic gradient descent is used
# with a batch size of 128

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
import tensorflow as tf
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
from PIL import Image
from six.moves import range


# open the pickle file and retrieve the tensors

pickle_file = 'bengaliOCR.pickle'

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



image_size = 50
num_labels = 50

# flatten the input datasets and reformat the labels using one-hot encodings

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 2 to [0.0, 1.0, 0.0 ...], 3 to [0.0, 0.0, 1.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels


train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)


print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)



# function to calculate accuracy given the predictions as softmax outputs, and the actual one-hot labels

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


# Function that implements the neural network. The first hidden layer has 1024 nodes, while the second has
# 256. Dropout, L2-regularization and learning rate decay are applied. In practice, this model had an accuracy
# of 85.6% on the test data

def SGD_relu_dropout():

    batch_size = 128
    h = 1024
    h2 = 256
    num_steps = 4001
    beta = 0.0005
    keep_prob = 0.75
    decay_step = 1000
    base = 0.86

    graph = tf.Graph()
    with graph.as_default():

      # Input data. For the training data, we use a placeholder that will be fed
      # at run time with a training minibatch.
      tf_train_dataset = tf.placeholder(tf.float32,
                                        shape=(batch_size, image_size * image_size))
      tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
      tf_valid_dataset = tf.constant(valid_dataset)
      tf_test_dataset = tf.constant(test_dataset)

      # Variables.
      weights1 = tf.Variable(
        tf.truncated_normal([image_size * image_size, h], stddev = 0.02))
      biases1 = tf.Variable(tf.zeros([h]))
      weights2 = tf.Variable(
        tf.truncated_normal([h, h2], stddev = 0.03125))
      biases2 = tf.Variable(tf.zeros([h2]))
      weights3 = tf.Variable(
        tf.truncated_normal([h2, num_labels], stddev = 0.0884))
      biases3 = tf.Variable(tf.zeros([num_labels]))

      global_step = tf.Variable(0)  # count the number of steps taken.
      learning_rate = tf.train.exponential_decay(0.5, global_step, decay_step, base)

      # Training computation.

      def model(dataset, useDropout = False):

          logits1 = tf.matmul(dataset, weights1) + biases1

          relu_outputs1 = tf.nn.relu(logits1)

          if useDropout:
                dropout_layer0 = tf.nn.dropout(relu_outputs1, keep_prob)
          else:
                dropout_layer0 = relu_outputs1

          logits2 = tf.matmul(dropout_layer0, weights2) + biases2

          relu_outputs2 = tf.nn.relu(logits2)

          if useDropout:
                dropout_layer = tf.nn.dropout(relu_outputs2, keep_prob)
          else:
                dropout_layer = relu_outputs2

          logits3 = tf.matmul(dropout_layer, weights3) + biases3

          return logits3


      train_logits = model(tf_train_dataset, True)
      valid_logits = model(tf_valid_dataset)
      test_logits = model(tf_test_dataset)



      loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(train_logits, tf_train_labels)) + beta * (tf.nn.l2_loss(weights1) + tf.nn.l2_loss(weights2) + tf.nn.l2_loss(weights3))



      # Optimizer.
      optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step = global_step)
      # Predictions for the training, validation, and test data.
      train_prediction = tf.nn.softmax(model(tf_train_dataset, False))
      valid_prediction = tf.nn.softmax(valid_logits)
      test_prediction = tf.nn.softmax(test_logits)


    num_steps = 20001

    with tf.Session(graph=graph) as session:
      tf.initialize_all_variables().run()
      print("Initialized")
      for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.

        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
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
          print("Minibatch loss at step %d: %f" % (step, l))
          print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
          print("Validation accuracy: %.1f%%" % accuracy(
            valid_prediction.eval(), valid_labels))
      print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

SGD_relu_dropout()