
import sys, os
sys.path.append(os.pardir)
import numpy as np
from keras.datasets.mnist import load_data
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()
(x_train, y_train), (x_test, y_test) = load_data()
#print(x_train.shape)
#print(y_train.shape)
#print(x_test.shape)
#print(y_test.shape)
img = x_train[0]
label = y_train[0]
print(label)
img_show(img)
