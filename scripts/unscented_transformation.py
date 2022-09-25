# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 12:27:50 2021

@author: halvorak
"""
import numpy as np

def unscented_transformation(sigmas, w, fx = None):
    """
    Calculates mean and covariance of a nonlinear function by the unscented transform. Every sigma point is propagated through the function, and combined with their weights the mean and covariance is calculated.

    Parameters
    ----------
    sigmas : TYPE np.ndarray(n, dim_sigma)
        DESCRIPTION. Array of sigma points. Each column contains a sigma point
    w : TYPE np.array(dim_sigma,)
        DESCRIPTION. Weights of each sigma point.
    fx : TYPE, optional function
        DESCRIPTION. The default is None. The non-linear function which the RV is propagated through. If None is supplied, the identity function f(x) = x is used. If another function is supplied, it must return a np.array()

    Returns
    -------
    mean : TYPE np.array(dim_y,)
        DESCRIPTION. Mean value of Y=f(X), X is a random variable (RV) 
    Py : TYPE np.array(dim_y,dim_y)
        DESCRIPTION. Covariance matrix, cov(Y) where Y=f(X)

    """
    if fx is None:
        fx = lambda x: x
    
    (n, dim_sigma) = sigmas.shape
    
    y0 = fx(sigmas[:, 0])
    dim_y = y0.shape[0]
    yi = np.zeros((dim_y, dim_sigma))
    yi[:,0] = y0
    for i in range(1, dim_sigma):
        yi[:, i] = fx(sigmas[:, i])
    # print(f"yi: {yi.shape}",
    #       f"w: {w.shape}")
    mean = np.dot(yi, w)
    
    Py = np.zeros((dim_y,dim_y))
    for i in range(dim_sigma):
        Py += w[i]*np.dot((yi[:, i] - mean).reshape(-1,1), 
                          (yi[:, i] - mean).reshape(-1,1).T)
    # print(f"mean: {mean.shape}\n",
    #       f"Py: {Py.shape}")
    return mean, Py

