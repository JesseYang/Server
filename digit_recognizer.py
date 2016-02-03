from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import numpy as np
import tensorflow as tf

class DigitRecognizer:

  def __init__(self):
    self.sess = tf.Session()

    self.x = tf.placeholder(tf.float32, [None, 784])
    self.W = tf.Variable(tf.zeros([784, 10]), name="weights")
    self.b = tf.Variable(tf.zeros([10]), name="bias")
    self.y = tf.nn.softmax(tf.matmul(self.x, self.W) + self.b)

    init_op = tf.initialize_all_variables()
    self.sess.run(init_op)

    self.saver = tf.train.Saver()
    self.saver.restore(self.sess, os.environ['HOME'] + "/Data/mnist_model.ckpt")

  def do(self, image):
    ret = self.sess.run(tf.argmax(self.y, 1), feed_dict={self.x:[image]})
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
