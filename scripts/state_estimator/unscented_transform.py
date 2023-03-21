# -*- coding: utf-8 -*-
# pylint: disable=invalid-name, too-many-arguments

"""Copyright 2015 Roger R Labbe Jr.

FilterPy library.
http://github.com/rlabbe/filterpy

Documentation at:
https://filterpy.readthedocs.org

Supporting book at:
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

This is licensed under an MIT license. See the readme.MD file
for more information.
"""

import numpy as np



def unscented_transformation_std(sigmas, wm, wc, symmetrization = True):
    """
    Calculates mean and covariance of sigma points by the unscented transform.

    Parameters
    ----------
    sigmas : TYPE np.ndarray(n, dim_sigma)
        DESCRIPTION. Array of sigma points. Each column contains a sigma point
    wm : TYPE np.array(dim_sigma,)
        DESCRIPTION. Weights for the mean calculation of each sigma point.
    wc : TYPE np.array(dim_sigma,)
        DESCRIPTION. Weights for the covariance calculation of each sigma point.
    symmetrization : TYPE bool, optional
        DESCRIPTION. Default is true. Symmetrize covariance/correlation matrix afterwards with Py = .5*(Py+Py.T)
    

    Returns
    -------
    mean : TYPE np.array(dim_y,)
        DESCRIPTION. Mean value of Y=f(X), X is a random variable (RV) 
    Py : TYPE np.array(dim_y,dim_y)
        DESCRIPTION. Covariance matrix, cov(Y) where Y=f(X)

    """
    try:
        (n, dim_sigma) = sigmas.shape
    except ValueError: #sigmas is 1D
        sigmas = np.atleast_2d(sigmas)
        (n, dim_sigma) = sigmas.shape 
        assert dim_sigma == wm.shape[0], "Dimensions are wrong"
    
    mean = sigmas @ wm
    
    #will only work with residuals between sigmas and mean, so "recalculate" sigmas (it is a new variable - but save memory cost by calling it the same as sigmas)
    sigmas = sigmas - mean.reshape(-1, 1)
    
    # Py = sum([wc_i*(np.outer(sig_i, sig_i)) for wc_i, sig_i in zip(wc, sigmas.T)])
    Py = (wc*sigmas) @ sigmas.T

    if symmetrization:
        Py = .5*(Py + Py.T)
    
    return mean, Py


def normalized_unscented_transformation_additive_noise(sigmas, wm, wc, noise_mat = None, symmetrization = True):
    """
    The Normalized Unscented Transformation (NUT): Calculates mean, standard deviation and correlation of sigma points by the unscented transform. The standard deviation is found first by explicitly calculating the diagonal of the resulting covariance matrix. Then, the sigma-points are scaled such that the resulting matrix from the UT is actually a correlation matrix

    Parameters
    ----------
    sigmas : TYPE np.ndarray(n, dim_sigma)
        DESCRIPTION. Array of sigma points. Each column contains a sigma point
    wm : TYPE np.array(dim_sigma,)
        DESCRIPTION. Weights for the mean calculation of each sigma point.
    wc : TYPE np.array(dim_sigma,)
        DESCRIPTION. Weights for the covariance calculation of each sigma point.
    noise_mat : TYPE np.array(n,n)
        DESCRIPTION. Noise matrix. If None is supplied, it is set to zeros.
    symmetrization : TYPE bool, optional
        DESCRIPTION. Default is true. Symmetrize covariance/correlation matrix afterwards with corr_y = .5*(corr_y+corr_y.T)

    Returns
    -------
    mean : TYPE np.array(dim_y,)
        DESCRIPTION. Mean value of Y=f(X), X is a random variable (RV) 
    corr_y : TYPE np.array(dim_y,dim_y)
        DESCRIPTION. Estimated correlation matrix, corr(Y) where Y=f(X)

    """
    try:
        (n, dim_sigma) = sigmas.shape
    except ValueError: #sigmas is 1D
        sigmas = np.atleast_2d(sigmas)
        (n, dim_sigma) = sigmas.shape 
        assert dim_sigma == wm.shape[0], "Dimensions are wrong"
    
    if noise_mat is None:
        noise_mat = np.zeros((n, n))
    
    mean = sigmas @ wm
    
    #will only work with residuals between sigmas and mean, so "recalculate" sigmas (it is a new variable - but save memory cost by calling it the same as sigmas)
    sigmas = sigmas - mean.reshape(-1, 1)
    
    sigmas_w = np.multiply(wc, sigmas)
    
    #Calculate diagonal elements of covariance matrix + noise mat and then take the square-root
    std_dev = np.sqrt(
        [sigmas_wi@sigmas_i + noise_ii 
         for sigmas_wi, sigmas_i, noise_ii 
         in zip(sigmas_w, sigmas, np.diag(noise_mat))
         ])
    
    
    #normalize sigma-points and noise-matrix ==> we calculate correlations and not covariance
    sigmas_norm = np.divide(sigmas, std_dev.reshape(-1,1))
    sigmas_w_norm = np.divide(sigmas_w, std_dev.reshape(-1,1))
    
    std_dev_mat = np.outer(std_dev, std_dev) #matrix required to get
    noise_mat_norm = np.divide(noise_mat, std_dev_mat)
    
    corr_y = sigmas_w_norm @ sigmas_norm.T + noise_mat_norm
    
    #check solution
    # print(np.diag(corr_y) - np.ones(mean.shape[0]))
    # if not np.linalg.norm(np.diag(corr_y) - np.ones(mean.shape[0])) < 1e-12:
    #     print(np.diag(corr_y) - np.ones(mean.shape[0]))
    #     print(np.diag(corr_y))
    if symmetrization:
        corr_y = .5*(corr_y + corr_y.T)
    return mean, corr_y, np.diag(std_dev)
