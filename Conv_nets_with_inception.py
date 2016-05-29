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


# Convolutional neural network using 2 convolutional layers, an inception module and then 2 fully-connected
# layers. The inception module consists of 1x1 convolutions that reduce the dimensions of the input to
# the module, followed by separate, parallel 3x3 and 5x5 convolutions. The input is also max-pooled with
# kernel size 3 and stride 1, followed by another 1x1 convolution. ReLU's are added before each of these paths
# are concatenated into one large output block. See README for details.

# Note that I used the Adam optimizer, because SGD would (mysteriously) not work, getting stuck at a loss
# of ~3.9. I also slightly reduced the batch size to 50 to allow faster training. This model yielded a
# 94.2% accuracy rate on the testset, an increase of 0.7% compared to the previous model without inception.


def deeper_inception_conv_net():

    batch_size = 50
    patch_size1 = 3
    patch_size2 = 5
    depth = 16
    depth1 = 32
    depth2 = 16
    depth3 = 8
    concat_depth = 48
    num_hidden = 64
    num_hidden2 = 32
    keep_prob = 0.5
    decay_step = 2000
    base = 0.9

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
          [patch_size1, patch_size1, num_channels, depth], stddev=0.3))
      layer1_biases = tf.Variable(tf.zeros([depth]))
      layer2_weights = tf.Variable(tf.truncated_normal(
          [patch_size2, patch_size2, depth, depth1], stddev=0.05))
      layer2_biases = tf.Variable(tf.constant(0.0, shape=[depth1]))
      layer3_weights = tf.Variable(tf.truncated_normal(
          [((image_size + 3) // 4) * ((image_size + 3) // 4) * concat_depth, num_hidden], stddev=0.05))
      layer3_biases = tf.Variable(tf.constant(0.0, shape=[num_hidden]))
      layer4_weights = tf.Variable(tf.truncated_normal(
          [num_hidden, num_hidden2], stddev=0.01))
      layer4_biases = tf.Variable(tf.constant(0.0, shape=[num_hidden2]))
      layer5_weights = tf.Variable(tf.truncated_normal(
          [num_hidden2, num_labels], stddev=0.01))
      layer5_biases = tf.Variable(tf.constant(0.0, shape=[num_labels]))

      inception1x1_weights = tf.Variable(tf.truncated_normal(
          [1, 1, depth1, depth2], stddev=0.25))
      inception1x1_biases = tf.Variable(tf.constant(0.0, shape=[depth2]))
      inception3x3_weights = tf.Variable(tf.truncated_normal(
          [patch_size1, patch_size1, depth2, depth3], stddev=0.05))
      inception3x3_biases = tf.Variable(tf.constant(0.0, shape=[depth3]))
      inception5x5_weights = tf.Variable(tf.truncated_normal(
          [patch_size2, patch_size2, depth2, depth3], stddev=0.08))
      inception5x5_biases = tf.Variable(tf.constant(0.0, shape=[depth3]))

      inception1x1_post_mxpool_wts = tf.Variable(tf.truncated_normal(
          [1, 1, depth1, depth2], stddev=0.04))
      post_maxpool_biases = tf.Variable(tf.constant(0.0, shape=[depth2]))

      global_step = tf.Variable(0, trainable = False)  # count the number of steps taken.
      learning_rate = tf.train.exponential_decay(0.005, global_step, decay_step, base)


      # Model.
      def model(data, useDropout):
        conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
        max_pooled = tf.nn.max_pool(conv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(max_pooled + layer1_biases)
        conv = tf.nn.conv2d(hidden, layer2_weights, [1, 1, 1, 1], padding='SAME')
        max_pooled = tf.nn.max_pool(conv, [1, 2, 2, 1], [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(max_pooled + layer2_biases)

        inception1x1_conv = tf.nn.conv2d(hidden, inception1x1_weights, [1, 1, 1, 1], padding='SAME')
        inception1x1_relu = tf.nn.relu(inception1x1_conv + inception1x1_biases)

        inception3x3_conv = tf.nn.conv2d(inception1x1_relu, inception3x3_weights, [1, 1, 1, 1], padding='SAME')
        inception3x3_relu = tf.nn.relu(inception3x3_conv + inception3x3_biases)

        inception5x5_conv = tf.nn.conv2d(inception1x1_relu, inception5x5_weights, [1, 1, 1, 1], padding='SAME')
        inception5x5_relu = tf.nn.relu(inception5x5_conv + inception5x5_biases)

        inception3x3_maxpool = tf.nn.max_pool(hidden, [1, 3, 3, 1], [1, 1, 1, 1], padding='SAME')
        inception1x1_post_maxpool = tf.nn.conv2d(inception3x3_maxpool, inception1x1_post_mxpool_wts, [1, 1, 1, 1], padding='SAME')
        inception1x1_post_maxpool = tf.nn.relu(inception1x1_post_maxpool + post_maxpool_biases)

        concat_filter = tf.concat(3, [inception1x1_relu, inception3x3_relu, inception5x5_relu, inception1x1_post_maxpool])
        concat_maxpooled = tf.nn.max_pool(concat_filter, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        shape = concat_maxpooled.get_shape().as_list()

        reshape = tf.reshape(concat_maxpooled, [shape[0], shape[1] * shape[2] * shape[3]])

        if useDropout == 1:
            dropout_layer2 = tf.nn.dropout(tf.nn.relu(reshape), keep_prob)
        else:
            dropout_layer2 = tf.nn.relu(reshape)
        hidden = tf.nn.relu(tf.matmul(dropout_layer2, layer3_weights) + layer3_biases)

        hidden = tf.nn.relu(tf.matmul(hidden, layer4_weights) + layer4_biases)
        return tf.matmul(hidden, layer5_weights) + layer5_biases

      # Training computation.
      logits = model(tf_train_dataset, 1)
      loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

      # Optimizer.
      optimizer = tf.train.AdamOptimizer(0.001).minimize(loss, global_step=global_step)

      # Predictions for the training, validation, and test data.
      train_prediction = tf.nn.softmax(model(tf_train_dataset, 0))
      valid_prediction = tf.nn.softmax(model(tf_valid_dataset, 0))
      test_prediction = tf.nn.softmax(model(tf_test_dataset, 0))



    num_steps = 30001

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
          #print(tf.Print(layer1_weights, [layer1_weights]).eval())
          print('Validation accuracy: %.1f%%' % accuracy(
            valid_prediction.eval(), valid_labels))
      print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))


deeper_inception_conv_net()