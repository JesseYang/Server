from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import numpy as np
import tensorflow as tf
import util

class Recognizer:

  def __init__(self):
    self.sess = tf.Session()

  def do(self, image):
    ret = self.sess.run(tf.argmax(self.y, 1), feed_dict={self.x: [image]})
    return(ret[0])

if __name__ == "__main__":
  ""
