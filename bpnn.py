import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
     sig = sigmoid(x)
     return sig * (1 - sig)

def init_weights(inout_layer, hidden_layer, output_layer):
    w1 = np.random.randn(inout_layer, hidden_layer)
    w2 = np.random.randn(hidden_layer, output_layer)
    return w1, w2

def forward_propagation(x, w1, w2):
    a1 = x
    a2 = np.dot(a1, w1)
    z2 = sigmoid(a2)
    a3 = np.dot(a2, w2)
    z3 = sigmoid(a3)
    return z3, a2
def back_propagation(x, z3, y, alpha, a2, w2, w1,):
    gj = z3 * (1-z3) * (y-z3)
    dw2 = np.dot((sigmoid(a2).T),gj)
    g2 = sigmoid_derivative(a2)
    dw1 = np.dot(x.T, np.dot(gj, w2.T) * g2)
    w2 = w2 - (alpha * dw2)
    w1 =w1 - (alpha * dw1 )
    return w1, w2


def bpnn(x, y, hidden_layer, iters, alpha):
    inout_layer = x.shape[1]
    output_layer = y.shape[1]
    w1, w2 = init_weights(inout_layer, hidden_layer, output_layer)
    for i in range(iters):
        z3, a2 = forward_propagation(x, w1, w2)

        cost = np.sum((z3 - y)**2) / (2*x.shape[0])


        if cost < 0.0001:
            break
        else:
            w1, w2 = back_propagation(x, z3, y, alpha, a2, w2, w1)
    y_pred, _ = forward_propagation(x, w1, w2)
    y_pred = np.round(y_pred)
    return y_pred, w1, w2

#test
x = np.random.randn(100, 4)
y = np.random.randint(0, 2, (100, 1))

y_pred, w1, w2 = bpnn(x, y, 2, 10000, 0.01)
print("y_pred:", y_pred)
print("w1:", w1)
print("w2：", w2)