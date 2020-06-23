import numpy as np

def Mean_Squared_Error(y, t):
    return 0.5 * np.sum((y-t)**2)
def Cross_Entropy_Error(y, t):
    delta = 1e-7
    return -np.sum(t*np.log(y + delta))
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    for idx, tmp_val in enumerate(x):
        x1 = tmp_val + h
        fxh1 = f(x1)
        x2 = tmp_val - h
        fxh2 = f(x2)
        grad[idx] = (fxh1 - fxh2) / (2*h)
    return grad
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr*grad
    return x
if __name__=='__main__':
    t = [  0,    0,   1,   0,    0,   0,   0,   0,   0,   0]
    y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
    print(Mean_Squared_Error( np.array(y), np.array(t) ))
    print(Cross_Entropy_Error( np.array(y), np.array(t) ))
    def function_1(x):
        return x**2 + 2*x + 1
    print(numerical_gradient(function_1, np.array([3.0, 4.0])))
    print(gradient_descent(function_1, init_x=np.array([3.0, 4.0])))
