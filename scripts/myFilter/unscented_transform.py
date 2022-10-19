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


# def unscented_transform(sigmas, Wm, Wc, noise_cov=None,
#                         mean_fn=None, residual_fn=None):
#     r"""
#     Computes unscented transform of a set of sigma points and weights.
#     returns the mean and covariance in a tuple.

#     This works in conjunction with the UnscentedKalmanFilter class.


#     Parameters
#     ----------

#     sigmas: ndarray, of size (n, 2n+1)
#         2D array of sigma points.

#     Wm : ndarray [# sigmas per dimension]
#         Weights for the mean.


#     Wc : ndarray [# sigmas per dimension]
#         Weights for the covariance.

#     noise_cov : ndarray, optional
#         noise matrix added to the final computed covariance matrix.

#     mean_fn : callable (sigma_points, weights), optional
#         Function that computes the mean of the provided sigma points
#         and weights. Use this if your state variable contains nonlinear
#         values such as angles which cannot be summed.

#         .. code-block:: Python

#             def state_mean(sigmas, Wm):
#                 x = np.zeros(3)
#                 sum_sin, sum_cos = 0., 0.

#                 for i in range(len(sigmas)):
#                     s = sigmas[i]
#                     x[0] += s[0] * Wm[i]
#                     x[1] += s[1] * Wm[i]
#                     sum_sin += sin(s[2])*Wm[i]
#                     sum_cos += cos(s[2])*Wm[i]
#                 x[2] = atan2(sum_sin, sum_cos)
#                 return x

#     residual_fn : callable (x, y), optional

#         Function that computes the residual (difference) between x and y.
#         You will have to supply this if your state variable cannot support
#         subtraction, such as angles (359-1 degreees is 2, not 358). x and y
#         are state vectors, not scalars.

#         .. code-block:: Python

#             def residual(a, b):
#                 y = a[0] - b[0]
#                 y = y % (2 * np.pi)
#                 if y > np.pi:
#                     y -= 2*np.pi
#                 return y

#     Returns
#     -------

#     x : ndarray [dimension]
#         Mean of the sigma points after passing through the transform.

#     P : ndarray
#         covariance of the sigma points after passing throgh the transform.

#     Examples
#     --------

#     See my book Kalman and Bayesian Filters in Python
#     https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
#     """

#     kmax, n = sigmas.shape

#     try:
#         if mean_fn is None:
#             # new mean is just the sum of the sigmas * weight
#             x = np.dot(Wm, sigmas)    # dot = \Sigma^n_1 (W[k]*Xi[k])
#         else:
#             x = mean_fn(sigmas, Wm)
#     except:
#         print(sigmas)
#         raise


#     # new covariance is the sum of the outer product of the residuals
#     # times the weights

#     # this is the fast way to do this - see 'else' for the slow way
#     if residual_fn is np.subtract or residual_fn is None:
#         y = sigmas - x[np.newaxis, :]
#         P = np.dot(y.T, np.dot(np.diag(Wc), y))
#     else:
#         P = np.zeros((n, n))
#         for k in range(kmax):
#             y = residual_fn(sigmas[k], x)
#             P += Wc[k] * np.outer(y, y)
            
#     # print(f"P_predict before Q: \n{P}")
#     # print(f"Noise cov in unscented_transform.py: \n{noise_cov}")
    
#     if noise_cov is not None:
#         P += noise_cov
#     # print(f"P_predict after Q: \n{P}")

#     return (x, P)

def unscented_transformation_gut(sigmas, wm, wc, symmetrization = True):
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
    
    Py = sum([wc_i*(np.outer(sig_i, sig_i)) for wc_i, sig_i in zip(wc, sigmas.T)])
    
    if symmetrization:
        Py = .5*(Py + Py.T)
    
    return mean, Py

def unscented_transformation_corr_std_dev(sigmas, wm, wc, std_dev, symmetrization = True):
    """
    Calculates mean and "correlation" of sigma points by the unscented transform. This is not the true correlation, as we use a "guess" for the standard deviation (we use e.g. the prior standard deviation of x when we want to calculate the posterior correlation. The true posterior correlation must use the posterior standard deviation). The update of standard deviation/correlation is handled outside this function 

    Parameters
    ----------
    sigmas : TYPE np.ndarray(n, dim_sigma)
        DESCRIPTION. Array of sigma points. Each column contains a sigma point
    wm : TYPE np.array(dim_sigma,)
        DESCRIPTION. Weights for the mean calculation of each sigma point.
    wc : TYPE np.array(dim_sigma,)
        DESCRIPTION. Weights for the covariance calculation of each sigma point.
    std_dev : TYPE np.array(n,)
        DESCRIPTION. A priori/estimated standard deviation of the covariance 
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
    
    mean = sigmas @ wm
    
    #will only work with residuals between sigmas and mean, so "recalculate" sigmas (it is a new variable - but save memory cost by calling it the same as sigmas)
    sigmas = sigmas - mean.reshape(-1, 1)
    
    #normalize ==> we calculate correlations and not covariance
    sigmas_norm = np.divide(sigmas, std_dev.reshape(-1,1))
    corr_y = sum([wc_i*(np.outer(sig_i, sig_i)) for wc_i, sig_i in zip(wc, sigmas_norm.T)])
    
    if symmetrization:
        corr_y = .5*(corr_y + corr_y.T)
    return mean, corr_y

def unscented_transformation_corr_std_dev_v2(sigmas, wm, wc, noise_mat = None, symmetrization = True):
    """
    Calculates mean and correlation of sigma points by the unscented transform. The standard deviation is found first by explicitly calculating the diagonal of the resulting covariance matrix. Then, the sigma-points are scaled such that the resulting matrix from the UT is actually a correlation matrix

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
    
    # print(f"sigmas: {sigmas.shape}\n",
    #       f"wm: {wm.shape}\n",
    #       f"wc: {wc.shape}\n",
    #       f"Q: {noise_mat.shape}\n",
    #       )
    
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
