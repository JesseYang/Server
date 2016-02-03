from flask import Flask
import digit_recognizer as dr
import os

app = Flask(__name__)

rec = dr.DigitRecognizer()

@app.route("/")
def hello():
  from tensorflow.examples.tutorials.mnist import input_data
  mnist = input_data.read_data_sets(os.environ['HOME'] + "/Data", one_hot=True)
  image = mnist.test.images[1]
  image = dr.img2binary(image)
  image = dr.index2ary(784, image)
  ret = rec.do(image)
  return str(ret)

@app.route("/digit/", methods=['POST'])
def digit():
  ret = rec.do(request.form['image'])
  return str(ret)

if __name__ == "__main__":
  # app.debug = True
  app.run(host='0.0.0.0')
  # app.run()
