import numpy as np

img_size = 28;
model_dir = "/Data/models/"
data_dir = "/Data/data/"
img_dir = "/Data/img/"


def img2binary(image):
  ret = [ ]
  for index, ele in enumerate(image):
    if(ele > 0):
      ret.append(index)
  return ret

def index2ary(len, index):
  img = np.zeros(len)
  for ele in index:
    img[ele] = 255
  return img

