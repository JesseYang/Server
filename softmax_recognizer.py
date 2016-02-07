from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import numpy as np
import tensorflow as tf
import util
import recognizer as rec

class SoftmaxRecognizer(rec.Recognizer):

  def __init__(self, model_name):
    rec.Recognizer.__init__(self)
    self.x = tf.placeholder(tf.float32, [None, util.img_size * util.img_size])

    # the softmax model
    self.W = tf.Variable(tf.zeros([util.img_size * util.img_size, 10]), name="weights")
    self.b = tf.Variable(tf.zeros([10]), name="bias")
    self.y = tf.nn.softmax(tf.matmul(self.x, self.W) + self.b)

    # init all variables
    init_op = tf.initialize_all_variables()
    self.sess.run(init_op)

    # load models
    self.saver = tf.train.Saver()
    self.saver.restore(self.sess, os.environ['HOME'] + util.model_dir + model_name + ".ckpt")

def main():
  from tensorflow.examples.tutorials.mnist import input_data
  mnist = input_data.read_data_sets(os.environ['HOME'] + util.data_dir, one_hot=True)
  image = mnist.test.images[1]

  image = img2binary(image)
  image = index2ary(util.img_size * util.img_size, image)

  rec = SoftmaxRecognizer()
  print(rec.do(image))

if __name__ == "__main__":
  main()
