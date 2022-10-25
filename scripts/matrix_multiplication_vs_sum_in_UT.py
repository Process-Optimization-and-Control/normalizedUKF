# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 16:02:29 2022

@author: halvorak

Compare the two methods for calculating the covariance matrix in the UT.

Pseudo-codes:
mat1 = sum(Wc[i]*sigmas[:,i] @ sigmas[:,i].T) #the standard way of computing (slow). Various methods for doing this
mat2 = (Wc*sigmas) @ sigmas.T #implement as matrix-multiplications (fast)
"""

import numpy as np

def matrix_multiplication(sigmas, w):
    mat = np.outer(w[0]*sigmas[:,0], sigmas[:,0])
    for i in range(1, w.shape[0]):
        mat += np.outer(w[i]*sigmas[:,i], sigmas[:,i])
    return mat
#define dimensions and generate random data
dim_x = 30   
dim_sig = int(2*dim_x)+1
sigmas = np.random.rand(dim_x, dim_sig)
w = np.random.rand(dim_sig)

#calculate the two methods and see that they give the same result (with round-off)
mat1 = sum([np.outer(wi*si, si) for wi, si in zip(w, sigmas.T)])
mat1_1 = sum([(wi*si).reshape(-1,1) @ si.reshape(1,-1) for wi, si in zip(w, sigmas.T)])
mat1_2 = matrix_multiplication(sigmas, w)
mat2 = (w*sigmas) @ sigmas.T

eps = np.linalg.norm(mat1-mat2)
eps1 = np.linalg.norm(mat1-mat1_1)
eps2 = np.linalg.norm(mat1-mat1_2)
eps_vec = np.array([eps, eps1, eps2])
print(f"eps: {eps_vec}")
assert (eps_vec < 1e-13).all()

#Check how fast the two methods are
%timeit sum([np.outer(wi*si, si) for wi, si in zip(w, sigmas.T)]) #std
%timeit sum([(wi*si).reshape(-1,1) @ si.reshape(1,-1) for wi, si in zip(w, sigmas.T)]) #std
%timeit matrix_multiplication(sigmas, w) #std, worst performance
%timeit (w*sigmas) @ sigmas.T #matri-matrix mulitplication, clearly best

