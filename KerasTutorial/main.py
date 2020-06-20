
import sys, os
#import dataset
import keras.dataset.mnist
sys.path.append(os.pardir)

(x_train, y_train), (x_test, y_test) = mnist.load_mnist(flatten=True, normalize=False)
#(x_train, y_train), (x_test, y_test) = dataset.mnist.load_mnist(flatten=True, normalize=False)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
