from flask import Flask, request
# import deep_recognizer as recognizer
import softmax_recognizer as recognizer
import os
import numpy as np
import Image
import uuid
import util

app = Flask(__name__)
# rec = recognizer.DeepRecognizer("mnist_deep_model")
rec = recognizer.SoftmaxRecognizer("mnist_softmax")

@app.route("/")
def hello():
  from tensorflow.examples.tutorials.mnist import input_data
  mnist = input_data.read_data_sets(os.environ['HOME'] + util.data_dir, one_hot=True)
  image = mnist.test.images[1]
  image = util.img2binary(image)
  image = util.index2ary(util.img_size * util.img_size, image)
  ret = rec.do(image)
  return str(ret)

@app.route("/digit/", methods=['POST'])
def digit():
  raw_data = request.form['image']
  data = map(lambda x: int(x), raw_data.split(','))
  ret = rec.do(util.index2ary(util.img_size * util.img_size, data))
  return str(ret)

@app.route("/create/", methods=['POST'])
def create():
  raw_data = request.form['image']
  label = request.form['label']
  data = map(lambda x: int(x), raw_data.split(','))
  data = util.index2ary(util.img_size * util.img_size, data)
  data_str = "".join(map(lambda x: x == 0 and '\xff' or '\x00', data))
  buf = bytes(data_str)
  img = Image.frombuffer('L', (util.img_size, util.img_size), buf, 'raw', 'L', 0, 1)
  img.save(os.environ['HOME'] + util.img_dir + str(label) + "/" + str(uuid.uuid1()) + ".bmp")
  return 1

if __name__ == "__main__":
  app.debug = True
  app.run(host='0.0.0.0')
