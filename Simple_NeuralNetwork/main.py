import sys, os
sys.path.append(os.pardir)
import numpy as np
from keras.datasets.mnist import load_data
#from gradient_simplenet
from two_layer_net import TwoLayerNet

if __name__=='__main__':
    (x_train, t_train), (x_test, t_test) = load_data() # mnist-data
    train_loss_list = []

    # hyper parameters
    iters_num = 10000
    print(x_train.shape)
    #train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1

    network = TwoLayerNet(input_size=784, hidden_size=50, ouput_size=10)
