import numpy as np
import pandas as pd

def cost_func(X, y, w, b):
    J = np.sum((w*X + b - y)**2) / (2*X.shape[0])
    return J

def gradient_descent(X, y, w, b, alpha, iters, const = 0.00001):
    m = X.shape[0]
    for i in range(iters):
        cost = cost_func(X, y, w, b)

        w = w - alpha / m * (np.sum((w * X + b - y) * X))
        b = b - alpha / m * (np.sum(w * X + b - y))

        if cost < const:
            break

    return w, b

#test
X = 2 * np.random.rand(100, 2)
y = 3 + 5 * X + np.random.randn(100, 1)

w ,b  = gradient_descent(X, y, 1, 1, 0.001, 100000)

print("w:", w)
print("b:", b)