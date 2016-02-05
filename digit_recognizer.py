from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import numpy as np
import tensorflow as tf

class DigitRecognizer:

  def __init__(self):
    # self.sess_softmax = tf.Session()
    self.sess_deep = tf.Session()
    self.x = tf.placeholder(tf.float32, [None, 784])

    # the softmax model
    # self.W_softmax = tf.Variable(tf.zeros([784, 10]), name="weights")
    # self.b_softmax = tf.Variable(tf.zeros([10]), name="bias")
    # self.y_softmax = tf.nn.softmax(tf.matmul(self.x, self.W_softmax) + self.b_softmax)


    # the deep learning model
    x_image = tf.reshape(self.x, [-1, 28, 28, 1])
    self.W_conv1_deep = self.weight_variable([5, 5, 1, 32], "w_conv1")
    self.b_conv1_deep = self.bias_variable([32], "b_conv1")
    h_conv1 = tf.nn.relu(self.conv2d(x_image, self.W_conv1_deep) + self.b_conv1_deep)
    h_pool1 = self.max_pool_2x2(h_conv1)

    self.W_conv2_deep = self.weight_variable([5, 5, 32, 64], "w_conv2")
    self.b_conv2_deep = self.bias_variable([64], "b_conv2")
    h_conv2 = tf.nn.relu(self.conv2d(h_pool1, self.W_conv2_deep) + self.b_conv2_deep)
    h_pool2 = self.max_pool_2x2(h_conv2)

    self.W_fc1_deep = self.weight_variable([7 * 7 * 64, 1024], "w_fc1")
    self.b_fc1_deep = self.bias_variable([1024], "b_fc1")
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, self.W_fc1_deep) + self.b_fc1_deep)
    self.keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

    self.W_fc2_deep = self.weight_variable([1024, 10], "w_c2")
    self.b_fc2_deep = self.bias_variable([10], "b_fc2")

    self.y_deep = tf.nn.softmax(tf.matmul(h_fc1_drop, self.W_fc2_deep) + self.b_fc2_deep)


    # init all variables
    init_op = tf.initialize_all_variables()
    # self.sess_softmax.run(init_op)
    self.sess_deep.run(init_op)

    # load models
    self.saver = tf.train.Saver()
    # self.saver.restore(self.sess_softmax, os.environ['HOME'] + "/Data/mnist_softmax_model.ckpt")
    self.saver.restore(self.sess_deep, os.environ['HOME'] + "/Data/mnist_deep_model.ckpt")

  def weight_variable(self, shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

  def bias_variable(self, shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

  def conv2d(self, x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

  def max_pool_2x2(self, x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')



  def do_softmax(self, image):
    ret = self.sess_softmax.run(tf.argmax(self.y_softmax, 1), feed_dict={self.x: [image]})
    return(ret[0])

  def do_deep(self, image):
    ret = self.sess_deep.run(tf.argmax(self.y_deep, 1), feed_dict={self.x: [image], self.keep_prob: 1.0})
    return(ret[0])

def img2binary(image):
  ret = [ ]
  for index, ele in enumerate(image):
    if(ele > 0):
      ret.append(index)
  return ret

def index2ary(len, index):
  img = np.zeros(len)
  for ele in index:
    img[ele] = 1
  return img

def main():
  from tensorflow.examples.tutorials.mnist import input_data
  mnist = input_data.read_data_sets(os.environ['HOME'] + "/Data", one_hot=True)
  image = mnist.test.images[1]

  image = img2binary(image)
  image = index2ary(784, image)

  rec = DigitRecognizer()
  print(rec.do(image))

if __name__ == "__main__":
  main()
