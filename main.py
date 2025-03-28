import numpy as np
import random
import matplotlib.pyplot

def sigmoid(x):
    return float(1/(1+np.exp(-x)))
def dih_sigmoid(x):
    return float(np.exp(x)/np.exp(x)+1)

x_i = 0
R = 0.0014

v = [10*random.random()-5 for i in range(13)]
W1 = [v[i] for i in range(8)]
B1 = [v[i] for i in range(8,13)]

def f(x):
    return x**2

x_interval = 1.5
x_points = 24
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
    return W1[3]*h2(x)+B1[2]
def zh2h4(x):
    return W1[5]*h2(x)+B1[3]
n1_org = [yh(Nx[i])-Ny[i] for i in range(L)]
n1 = [2*(yh(Nx[i])-Ny[i]) for i in range(L)]
loss = (sum(n1_org)/len(n1_org))**2

dh4 = [W1[7]*n1[i] for i in range(len(n1))]
dh3 = [W1[6]*n1[i] for i in range(len(n1))]
dh2 = [W1[3]*dih_sigmoid(zh2h3(Nx[i]))*dh3[i]+W1[5]*dih_sigmoid(zh2h4(Nx[i]))*dh4[i] for i in range(len(n1))]
dh1 = [W1[2]*dih_sigmoid(zh1h3(Nx[i]))*dh3[i]+W1[4]*dih_sigmoid(zh1h4(Nx[i]))*dh4[i] for i in range(len(n1))]

db1 = sum([dih_sigmoid(zh1(Nx[i]))*dh1[i] for i in range(L)])
db2 = sum([dih_sigmoid(zh2(Nx[i]))*dh2[i] for i in range(L)])
print(db1)
