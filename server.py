from flask import Flask, request
import digit_recognizer as dr
import os
import numpy as np
import Image
import uuid

app = Flask(__name__)
rec = dr.DigitRecognizer()

@app.route("/")
def hello():
  from tensorflow.examples.tutorials.mnist import input_data
  mnist = input_data.read_data_sets(os.environ['HOME'] + "/Data", one_hot=True)
  image = mnist.test.images[1]
  image = dr.img2binary(image)
  image = dr.index2ary(784, image)
  # ret = rec.do_softmax(image)
  ret = rec.do_deep(image)
  return str(ret)

@app.route("/digit/", methods=['POST'])
def digit():
  raw_data = request.form['image']
  data = map(lambda x: int(x), raw_data.split(','))
  ret = rec.do_deep(dr.index2ary(784, data))
  return str(ret)

@app.route("/create/", methods=['POST'])
def create():
  raw_data = request.form['image']
  label = request.form['label']
  data = map(lambda x: int(x), raw_data.split(','))
  data = dr.index2ary(784, data)
  data_str = "".join(map(lambda x: x == 0 and '\xff' or '\x00', data))
  buf = bytes(data_str)
  print(len(buf))
  img = Image.frombuffer('L', (28, 28), buf, 'raw', 'L', 0, 1)
  img.save(os.environ['HOME'] + "/Data/" + str(label) + "/" + str(uuid.uuid1()) + ".bmp")
  return 1

if __name__ == "__main__":
  app.debug = True
  app.run(host='0.0.0.0')
  # app.run()
