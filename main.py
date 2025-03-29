import numpy as np
import random
import matplotlib.pyplot as plt
from time import sleep

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def dih_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)

x_i = 0
R = 0.01

v = [8*random.random()-4 for i in range(13)]
W1 = [v[i] for i in range(8)]
B1 = [v[i] for i in range(8,13)]

def f(x):
    return np.cos(x)

x_interval = 3
x_points = 50
n_i = np.linspace(-x_interval, 1.5, x_points)
n_i = [float(n_i[i]) for i in range(len(n_i))]
N = [(i,f(i)) for i in n_i]
L = len(N)
Nx = [N[i][0] for i in range(L)]
Ny = [N[i][1] for i in range(L)]

def zh1(x):
    return  W1[0]*x+B1[0]
def zh2(x):
    return  W1[1]*x+B1[1]
def h1(x):
    return sigmoid(zh1(x))
def h2(x):
    return sigmoid(zh2(x))
def zh3(x):
    return W1[2]*h1(x) +W1[3]*h2(x)+B1[2]
def zh4(x):
    return W1[4]*h1(x)+W1[5]*h2(x)+B1[4]
def h3(x):
    return sigmoid(zh3(x))
def h4(x):
    return sigmoid(zh4(x))
def z1(x):
    return W1[6]*h3(x)+W1[7]*h4(x)+B1[-1]
def yh(x):
    return z1(x)
def zh1h3(x):
    return W1[2]*h1(x)+B1[2]
def zh1h4(x):
    return W1[4]*h1(x)+B1[3]
def zh2h3(x):
    return W1[3] * h2(x) + W1[2] * h1(x) + B1[2]
def zh2h4(x):
    return W1[5] * h2(x) + W1[4] * h1(x) + B1[3]

n1 = [2*(yh(Nx[i])-Ny[i]) for i in range(L)]
def loss():
    return sum([(n1[i] / 2) ** 2 for i in range(len(n1))]) / len(n1)

iteration = 0
losses = []
while loss() > 0.008:
    if iteration > 1000 and loss() > 0.4:
        W1 = [random.random() for i in range(8)]
        B1 = [random.random() for i in range(5)]
        iteration = 0

    iteration += 1

    n1 = [2 * (yh(Nx[i]) - Ny[i]) for i in range(L)]

    dh4 = [W1[7] * n1[i] for i in range(L)]
    dh3 = [W1[6] * n1[i] for i in range(L)]
    dh2 = [W1[3] * dih_sigmoid(zh2h3(Nx[i])) * dh3[i] + W1[5] * dih_sigmoid(zh2h4(Nx[i])) * dh4[i] for i in range(L)]
    dh1 = [W1[2] * dih_sigmoid(zh1h3(Nx[i])) * dh3[i] + W1[4] * dih_sigmoid(zh1h4(Nx[i])) * dh4[i] for i in range(L)]

    db1 = sum([dih_sigmoid(zh1(Nx[i])) * dh1[i] for i in range(L)])
    db2 = sum([dih_sigmoid(zh2(Nx[i])) * dh2[i] for i in range(L)])
    db3 = sum([dih_sigmoid(zh3(Nx[i])) * dh3[i] for i in range(L)])
    db4 = sum([dih_sigmoid(zh4(Nx[i])) * dh4[i] for i in range(L)])
    db5 = sum(n1)

    dw1 = sum([Nx[i] * dih_sigmoid(zh1(Nx[i])) * dh1[i] for i in range(L)])
    dw2 = sum([Nx[i] * dih_sigmoid(zh2(Nx[i])) * dh2[i] for i in range(L)])
    dw3 = sum([h1(Nx[i]) * dih_sigmoid(zh3(Nx[i])) * dh3[i] for i in range(L)])
    dw4 = sum([h2(Nx[i]) * dih_sigmoid(zh3(Nx[i])) * dh3[i] for i in range(L)])
    dw5 = sum([h1(Nx[i]) * dih_sigmoid(zh4(Nx[i])) * dh4[i] for i in range(L)])
    dw6 = sum([h2(Nx[i]) * dih_sigmoid(zh4(Nx[i])) * dh4[i] for i in range(L)])
    dw7 = sum([h3(Nx[i]) * n1[i] for i in range(L)])
    dw8 = sum([h4(Nx[i]) * n1[i] for i in range(L)])

    W1[0] -= dw1 * R
    W1[1] -= dw2 * R
    W1[2] -= dw3 * R
    W1[3] -= dw4 * R
    W1[4] -= dw5 * R
    W1[5] -= dw6 * R
    W1[6] -= dw7 * R
    W1[7] -= dw8 * R
    B1[0] -= db1 * R
    B1[1] -= db2 * R
    B1[2] -= db3 * R
    B1[3] -= db4 * R
    B1[4] -= db5 * R

    if iteration % 100 == 0:
        print(f"loss: {loss()}")
        losses.append(loss())

print(loss())
plt.plot(Nx, Ny, label="true")
plt.plot(Nx, [yh(x) for x in Nx], label="predicted")
plt.legend()
plt.show()
