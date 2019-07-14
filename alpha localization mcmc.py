# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 17:20:40 2018

@author: Chulin
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import digamma
from scipy import optimize
import datetime

np.random.seed(1874617301)

def deltaG(params, B):
    B_d, B_dphi, A = params
    A = A + 1e-20
    B_d = np.abs(B_d) + 1e-20
    B_dphi = np.abs(B_dphi) + 1e-20
    return(A * (digamma(1/2 + B_dphi/(2*B)) - digamma(1/2 + B_d/(2*B)) + np.log(B_d/B_dphi)))
    
def Cost(params, X_data, Y_data):
    return np.sum((Y_data - deltaG(params, X_data))**2)

def readPlot(filename):
    B_1 = []
    G_1 = []
    B_2 = []
    G_2 = []
    with open(filename, "r") as file:
        for line in file:
            b1, g1, b2, g2 = list(map(float, line.split(",")))
            B_1.append(b1)
            G_1.append(g1)
            B_2.append(b2)
            G_2.append(g2)
    return (np.array(B_1), np.array(G_1), np.array(B_2), np.array(G_2))


B_1, G_1, B_2, G_2 = readPlot("delta G 7K B G1 B G2 unit S.txt")

ee = 1.60217662e-19
hbar = 1.0545718e-34
prefct = 4*np.pi**2 * hbar / ee**2

X = B_1
Y = G_1

leng = X.shape[0]

paramstore = []
resid = []
data = []

t = datetime.datetime.now()
print(t)
Ntry = 100
for i in range(Ntry):
    N = leng // 20
    idx = np.random.randint(0, leng, size=N)
    Y_data = Y[idx]
    X_data = np.abs(X[idx])
    initialParam = [np.random.exponential(scale=16), np.random.exponential(scale=16),\
                    np.random.exponential(scale=10)]
    Niter = int(5e4)
    dist = 1
    T = 1e8
    param = initialParam
    steps = np.random.normal(size=(Niter,3))
    acc = np.random.uniform(low=0, high=1, size=Niter)
    param_all = []
    cost_all = []
    cost = Cost(param, X_data, Y_data)
    decay = np.power(1/T, 1/Niter)
    T = 1
    for a, step in zip(acc, steps):
        p = np.abs(param * (1 + step*dist))
        if(Cost(p, X_data, Y_data) / Cost(param, X_data, Y_data)) < np.exp(a * T):
            param = p
            cost = Cost(p, X_data, Y_data)
        param_all.append(param)
        cost_all.append(cost)
        T = T * decay
    tnow = datetime.datetime.now()
    eta = (Ntry-i-1) / (i+1) * (tnow - t) + tnow
    print("{}\t{}\t{}".format(eta, i, cost))
    data.append((X_data, Y_data))
    paramstore.append(param)
    resid.append(cost)
paramstore = np.array(paramstore)
resid = np.array(resid)

plt.figure()
plt.hist(np.log(resid), bins=20)

plt.figure()
xx = np.arange(0.01,16,0.01)
order = np.argsort(np.abs(B_1))
plt.plot(np.abs(X)[order], Y[order], "r")
resultLen = resid.shape[0]
resultOrd = np.argsort(resid)
nn = 4 #  top 1/4

sortedCost = resid[resultOrd[:resultLen//nn]]
sortedParam = []
for param in paramstore[resultOrd[:resultLen//nn]]:
    param = np.abs(param)
    sortedParam.append([max(param[:2]), min(param[:2]), param[2] * prefct])

for param, cost in zip(sortedParam, sortedCost):
    plt.plot(xx, deltaG(param, xx))
    print("B_d:  {:.8f}\tB_dphi:  {:.8f}\tN:  {:.8f}\tcost:  {:.8e}"\
          .format(param[0], param[1], param[2], cost))
plt.legend(["Experiment","Localization fit"])

finalParam = np.mean(sortedParam, axis=0)
print("Final:\nB_d:  {:.8f}\tB_dphi:  {:.8f}\tN:  {:.8f}"\
      .format(finalParam[0], finalParam[1], finalParam[2]))
