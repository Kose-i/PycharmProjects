from load_img import load_mnist
from make_network import *

def run():
    (x_train, y_train), (x_test, y_test) = load_mnist()
    network = init_network_pkl("sample_weight.pkl")
    accuracy_cnt = 0

    x = x_train
    t = y_train
    print(x.shape)
    for i in range(len(x)):
        y = predict(network, x[i].reshape(784, ))
        p = np.argmax(y)
        if p == t[i]:
            accuracy_cnt += 1
    print("Accuracy:" + str(float(accuracy_cnt)/len(x)))

if __name__=='__main__':
    run()
