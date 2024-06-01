import numpy as np
from numpy import dot

def nonlin(x,deriv=False):
    if (deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

X = np.array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])

Y = np.array([[0,1,1,0]]).T

np.random.seed(1)

syn0 = 2*np.random.random((3,1))-1

l0 = X
l1 = nonlin(dot(X,syn0))

l1_error = Y - l1

l1_delta = l1_error * nonlin(l1,True)

syn0 += dot(l0.T,l1_delta)

print(l1)




