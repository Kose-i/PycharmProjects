
import sys, os
sys.path.append(os.pardir)
import numpy as np
import pickle
from load_img import load_mnist

def sigmoid(x):
    return 1/(1+np.exp(-x))
def relu(x):
    return np.maximum(0, x)
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c) # overflow
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    return network
def init_network_pkl(pkl_name):
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    return y
def predict(network, x):
    return forward(network, x)
def print_accuracy():
    train_data, test_data = load_mnist()
    network = init_network_pkl("sample_weight.pkl")
    accuracy_cnt = 0

    x = train_data[0]
    t = train_data[1]
    print(x.shape)
    for i in range(len(x)):
        y = predict(network, x[i].reshape(784, ))
        p = np.argmax(y)
        if p == t[i]:
            accuracy_cnt += 1
    print("Accuracy:" + str(float(accuracy_cnt)/len(x)))

if __name__=='__main__':
    print_accuracy()
