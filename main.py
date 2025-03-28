import numpy as np
import random
import matplotlib.pyplot


def sigmoid(x):
    return 1/(1+np.exp(-x))

x_i = 0
R = 0.0014

v = [10*random.random()-5 for i in range(13)]

def f(x):
    return x**2

x_interval = 1.5
x_points = 24
n_i = np.linspace(-x_interval, 1.5, x_points)
N = [(float(i),float(f(i))) for i in n_i]
L = len(N)
