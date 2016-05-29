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
num_channels = 1 # grayscale


# convnets in tensorflow require 4d tensor inputs [image_no, height, width, num_channels]. The following
# functin reformats the input dataset to form a 4d tensor, and reformats the labels as one-hot encodings

def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels


train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)


print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)



def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


# function that implements a simple convolutional neural network with 2 convolutional layers followed by
# 1 fully-connected layer. Both the convolutional layers use 5x5 convolutions with stride 2. This model
# achieved 86.5% accuracy after training for 4000 steps using SGD

def simple_conv_net():

    batch_size = 128
    patch_size = 5
    depth = 16
    num_hidden = 64
    beta = 0.0005

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
          [patch_size, patch_size, num_channels, depth], stddev=0.1))
      layer1_biases = tf.Variable(tf.zeros([depth]))
      layer2_weights = tf.Variable(tf.truncated_normal(
          [patch_size, patch_size, depth, depth], stddev=0.1))
      layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
      layer3_weights = tf.Variable(tf.truncated_normal(
          [((image_size + 3) // 4) * ((image_size + 3) // 4) * depth, num_hidden], stddev=0.1))
      layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
      layer4_weights = tf.Variable(tf.truncated_normal(
          [num_hidden, num_labels], stddev=0.1))
      layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

      # Model.
      def model(data):
        conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer1_biases)       # relu's are used as non-linearities
        conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer2_biases)
        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
        return tf.matmul(hidden, layer4_weights) + layer4_biases

      # Training computation.
      logits = model(tf_train_dataset)
      loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

      # Optimizer.
      optimizer = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

      # Predictions for the training, validation, and test data.
      train_prediction = tf.nn.softmax(logits)
      valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
      test_prediction = tf.nn.softmax(model(tf_test_dataset))

      num_steps = 4001

      with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print('Initialized')
        for step in range(num_steps):
          offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
          batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
          batch_labels = train_labels[offset:(offset + batch_size), :]
          feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
          _, l, predictions = session.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict)
          if (step % 50 == 0):
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
            print('Validation accuracy: %.1f%%' % accuracy(
              valid_prediction.eval(), valid_labels))
        print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))



# function that implements a convolutional network similar to the one above, but with max-pooling after
# each convolutional layer, an extra fully-connected layer, and dropout right before the first fully-connected layer.
# The convolutional layers use stride 1 instead of 2, but the max-pooling layer uses stride 2 and kernel size 2.
# The learning rate has also been annealed exponentially. Running SGD for 20,000 steps yielded an 92.2%
# test set accuracy

def improved_conv_net_2():

    batch_size = 64
    patch_size = 5
    depth = 16
    num_hidden = 64
    num_hidden2 = 32
    keep_prob = 0.75
    decay_step = 1000
    base = 0.86

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
          [patch_size, patch_size, num_channels, depth], stddev=0.1))
      layer1_biases = tf.Variable(tf.zeros([depth]))
      layer2_weights = tf.Variable(tf.truncated_normal(
          [patch_size, patch_size, depth, depth], stddev=0.1))
      layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
      layer3_weights = tf.Variable(tf.truncated_normal(
          [((image_size + 3) // 4) * ((image_size + 3) // 4) * depth, num_hidden], stddev=0.1))
      layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
      layer4_weights = tf.Variable(tf.truncated_normal(
          [num_hidden, num_hidden2], stddev=0.1))
      layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden2]))
      layer5_weights = tf.Variable(tf.truncated_normal(
          [num_hidden2, num_labels], stddev=0.1))
      layer5_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

      global_step = tf.Variable(0)  # count the number of steps taken.
      learning_rate = tf.train.exponential_decay(0.2, global_step, decay_step, base)


      # Model.
      def model(data, useDropout):
        conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
        max_pooled = tf.nn.max_pool(conv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(max_pooled + layer1_biases)
        conv = tf.nn.conv2d(hidden, layer2_weights, [1, 1, 1, 1], padding='SAME')
        max_pooled = tf.nn.max_pool(conv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(max_pooled + layer2_biases)
        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
        if useDropout == 1:
            dropout_layer2 = tf.nn.dropout(reshape, keep_prob)
        else:
            dropout_layer2 = reshape
        hidden = tf.nn.relu(tf.matmul(dropout_layer2, layer3_weights) + layer3_biases)

        hidden = tf.nn.relu(tf.matmul(hidden, layer4_weights) + layer4_biases)
        return tf.matmul(hidden, layer5_weights) + layer5_biases

      # Training computation.
      logits = model(tf_train_dataset, 1)
      loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

      # Optimizer.
      optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

      # Predictions for the training, validation, and test data.
      train_prediction = tf.nn.softmax(model(tf_train_dataset, 0))
      valid_prediction = tf.nn.softmax(model(tf_valid_dataset, 0))
      test_prediction = tf.nn.softmax(model(tf_test_dataset, 0))



    num_steps = 20001

    with tf.Session(graph=graph) as session:
      tf.initialize_all_variables().run()
      print('Initialized')
      for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run(
          [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 50 == 0):
          print('Minibatch loss at step %d: %f' % (step, l))
          print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
          print('Validation accuracy: %.1f%%' % accuracy(
            valid_prediction.eval(), valid_labels))
      print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))



# Remarks: Changing the patch size of the first convolutional layer above to 3x3 instead of 5x5 resulted
# in 93.5% accuracy on the test dataset. My reasoning was that such a convolutional layer would capture
# and preserve a little more detail with respect to the small dots that form the only distinction between
# a lot of Bengali character pairs (for instance, ড and ড়). I also used a keep-probability of 0.5 in the
# dropout layer, instead of 0.75, in the modified version, given below:



def improved_conv_net_3():

    batch_size = 64
    patch_size1 = 3         # Note: patch size for first conv layer has been changed to 3
    patch_size2 = 5
    depth = 16
    num_hidden = 64
    num_hidden2 = 32
    keep_prob = 0.5
    decay_step = 1000
    base = 0.86

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
          [patch_size1, patch_size1, num_channels, depth], stddev=0.5))
      layer1_biases = tf.Variable(tf.zeros([depth]))
      layer2_weights = tf.Variable(tf.truncated_normal(
          [patch_size2, patch_size2, depth, depth], stddev=0.1))
      layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
      layer3_weights = tf.Variable(tf.truncated_normal(
          [((image_size + 3) // 4) * ((image_size + 3) // 4) * depth, num_hidden], stddev=0.05))
      layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
      layer4_weights = tf.Variable(tf.truncated_normal(
          [num_hidden, num_hidden2], stddev=0.1))
      layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden2]))
      layer5_weights = tf.Variable(tf.truncated_normal(
          [num_hidden2, num_labels], stddev=0.1))
      layer5_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

      global_step = tf.Variable(0)  # count the number of steps taken.
      learning_rate = tf.train.exponential_decay(0.2, global_step, decay_step, base)


      # Model.
      def model(data, useDropout):
        conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
        max_pooled = tf.nn.max_pool(conv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(max_pooled + layer1_biases)
        conv = tf.nn.conv2d(hidden, layer2_weights, [1, 1, 1, 1], padding='SAME')
        max_pooled = tf.nn.max_pool(conv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(max_pooled + layer2_biases)
        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
        if useDropout == 1:
            dropout_layer2 = tf.nn.dropout(reshape, keep_prob)
        else:
            dropout_layer2 = reshape
        hidden = tf.nn.relu(tf.matmul(dropout_layer2, layer3_weights) + layer3_biases)

        hidden = tf.nn.relu(tf.matmul(hidden, layer4_weights) + layer4_biases)
        return tf.matmul(hidden, layer5_weights) + layer5_biases

      # Training computation.
      logits = model(tf_train_dataset, 1)
      loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

      # Optimizer.
      optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

      # Predictions for the training, validation, and test data.
      train_prediction = tf.nn.softmax(model(tf_train_dataset, 0))
      valid_prediction = tf.nn.softmax(model(tf_valid_dataset, 0))
      test_prediction = tf.nn.softmax(model(tf_test_dataset, 0))



    num_steps = 30001   # Train for 30000 steps

    with tf.Session(graph=graph) as session:
      tf.initialize_all_variables().run()
      print('Initialized')
      for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run(
          [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 50 == 0):
          print('Minibatch loss at step %d: %f' % (step, l))
          print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
          print('Validation accuracy: %.1f%%' % accuracy(
            valid_prediction.eval(), valid_labels))
      print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))