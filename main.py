import numpy as np
import random

def sigmoid(x):
    return 1/(1+np.exp(-x))

x_i = 0
R = 0.0014

v = [10*random.random()-5 for i in range(13)]
