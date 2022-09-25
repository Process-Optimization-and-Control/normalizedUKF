# -*- coding: utf-8 -*-
# pylint: disable=invalid-name, too-many-instance-attributes

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

from __future__ import division
import numpy as np
from scipy.linalg import cholesky
#from helpers import pretty_str
# import os
# import sys
# wdir = os.getcwd()
# module_path = os.path.join(wdir, "utils_filter")
# sys.path.append(module_path)
# from utils_filter import helpers 
from . import helpers 

import scipy.stats

class MerweScaledSigmaPoints(object):

    """
    Generates sigma points and weights according to Van der Merwe's
    2004 dissertation[1] for the UnscentedKalmanFilter class.. It
    parametizes the sigma points using alpha, beta, kappa terms, and
    is the version seen in most publications.

    Unless you know better, this should be your default choice.

    Parameters
    ----------

    n : int
        Dimensionality of the state. 2n+1 weights will be generated.

    alpha : float
        Determins the spread of the sigma points around the mean.
        Usually a small positive value (1e-3) according to [3].

    beta : float
        Incorporates prior knowledge of the distribution of the mean. For
        Gaussian x beta=2 is optimal, according to [3].

    kappa : float, default=0.0
        Secondary scaling parameter usually set to 0 according to [4],
        or to 3-n according to [5].

    sqrt_method : function(ndarray), default=scipy.linalg.cholesky
        Defines how we compute the square root of a matrix, which has
        no unique answer. Cholesky is the default choice due to its
        speed. Typically your alternative choice will be
        scipy.linalg.sqrtm. Different choices affect how the sigma points
        are arranged relative to the eigenvectors of the covariance matrix.
        Usually this will not matter to you; if so the default cholesky()
        yields maximal performance. As of van der Merwe's dissertation of
        2004 [6] this was not a well reseached area so I have no advice
        to give you.

        If your method returns a triangular matrix it must be upper
        triangular. Do not use numpy.linalg.cholesky - for historical
        reasons it returns a lower triangular matrix. The SciPy version
        does the right thing.

    subtract : callable (x, y), optional
        Function that computes the difference between x and y.
        You will have to supply this if your state variable cannot support
        subtraction, such as angles (359-1 degreees is 2, not 358). x and y
        are state vectors, not scalars.

    Attributes
    ----------

    Wm : np.array
        weight for each sigma point for the mean

    Wc : np.array
        weight for each sigma point for the covariance

    Examples
    --------

    See my book Kalman and Bayesian Filters in Python
    https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python


    References
    ----------

    .. [1] R. Van der Merwe "Sigma-Point Kalman Filters for Probabilitic
           Inference in Dynamic State-Space Models" (Doctoral dissertation)

    """


    def __init__(self, n, alpha, beta, kappa, sqrt_method=None, subtract=None):
        #pylint: disable=too-many-arguments

        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        if sqrt_method is None:
            self.sqrt = cholesky
        else:
            self.sqrt = sqrt_method

        if subtract is None:
            self.subtract = np.subtract
        else:
            self.subtract = subtract

        self._compute_weights()


    def num_sigmas(self):
        """ Number of sigma points for each variable in the state x"""
        return 2*self.n + 1


    def sigma_points(self, x, P):
        """ Computes the sigma points for an unscented Kalman filter
        given the mean (x) and covariance(P) of the filter.
        Returns tuple of the sigma points and weights.

        Works with both scalar and array inputs:
        sigma_points (5, 9, 2) # mean 5, covariance 9
        sigma_points ([5, 2], 9*eye(2), 2) # means 5 and 2, covariance 9I

        Parameters
        ----------

        x : An array-like object of the means of length n
            Can be a scalar if 1D.
            examples: 1, [1,2], np.array([1,2])

        P : scalar, or np.array
           Covariance of the filter. If scalar, is treated as eye(n)*P.

        Returns
        -------

        sigmas : np.array, of size (n, 2n+1)
            Two dimensional array of sigma points. Each column contains all of
            the sigmas for one dimension in the problem space.

            Ordered by Xi_0, Xi_{1..n}, Xi_{n+1..2n}
        """

        if self.n != np.size(x):
            raise ValueError("expected size(x) {}, but size is {}".format(
                self.n, np.size(x)))

        n = self.n

        if np.isscalar(x):
            x = np.asarray([x])

        if  np.isscalar(P):
            P = np.eye(n)*P
        else:
            P = np.atleast_2d(P)

        lambda_ = self.alpha**2 * (n + self.kappa) - n
        U = self.sqrt((lambda_ + n)*P)

        sigmas = np.zeros((2*n+1, n))
        sigmas[0] = x
        for k in range(n):
            # pylint: disable=bad-whitespace
            sigmas[k+1]   = self.subtract(x, -U[k])
            sigmas[n+k+1] = self.subtract(x, U[k])

        return sigmas


    def _compute_weights(self):
        """ Computes the weights for the scaled unscented Kalman filter.

        """

        n = self.n
        lambda_ = self.alpha**2 * (n +self.kappa) - n

        c = .5 / (n + lambda_)
        self.Wc = np.full(2*n + 1, c)
        self.Wm = np.full(2*n + 1, c)
        self.Wc[0] = lambda_ / (n + lambda_) + (1 - self.alpha**2 + self.beta)
        self.Wm[0] = lambda_ / (n + lambda_)



    def __repr__(self):

        return '\n'.join([
            'MerweScaledSigmaPoints object',
            helpers.pretty_str('n', self.n),
            helpers.pretty_str('alpha', self.alpha),
            helpers.pretty_str('beta', self.beta),
            helpers.pretty_str('kappa', self.kappa),
            helpers.pretty_str('Wm', self.Wm),
            helpers.pretty_str('Wc', self.Wc),
            helpers.pretty_str('subtract', self.subtract),
            helpers.pretty_str('sqrt', self.sqrt)
            ])


class JulierSigmaPoints(object):
    """
    Generates sigma points and weights according to Simon J. Julier
    and Jeffery K. Uhlmann's original paper[1]. It parametizes the sigma
    points using kappa.

    Parameters
    ----------

    n : int
        Dimensionality of the state. 2n+1 weights will be generated.

    kappa : float, default=0.
        Scaling factor that can reduce high order errors. kappa=0 gives
        the standard unscented filter. According to [Julier], if you set
        kappa to 3-dim_x for a Gaussian x you will minimize the fourth
        order errors in x and P.

    sqrt_method : function(ndarray), default=scipy.linalg.cholesky
        Defines how we compute the square root of a matrix, which has
        no unique answer. Cholesky is the default choice due to its
        speed. Typically your alternative choice will be
        scipy.linalg.sqrtm. Different choices affect how the sigma points
        are arranged relative to the eigenvectors of the covariance matrix.
        Usually this will not matter to you; if so the default cholesky()
        yields maximal performance. As of van der Merwe's dissertation of
        2004 [6] this was not a well reseached area so I have no advice
        to give you.

        If your method returns a triangular matrix it must be upper
        triangular. Do not use numpy.linalg.cholesky - for historical
        reasons it returns a lower triangular matrix. The SciPy version
        does the right thing.

    subtract : callable (x, y), optional
        Function that computes the difference between x and y.
        You will have to supply this if your state variable cannot support
        subtraction, such as angles (359-1 degreees is 2, not 358). x and y

    Attributes
    ----------

    Wm : np.array
        weight for each sigma point for the mean

    Wc : np.array
        weight for each sigma point for the covariance

    References
    ----------

    .. [1] Julier, Simon J.; Uhlmann, Jeffrey "A New Extension of the Kalman
        Filter to Nonlinear Systems". Proc. SPIE 3068, Signal Processing,
        Sensor Fusion, and Target Recognition VI, 182 (July 28, 1997)
   """

    def __init__(self, n, kappa=0., sqrt_method=None, subtract=None):

        self.n = n
        self.kappa = kappa
        if sqrt_method is None:
            self.sqrt = cholesky
        else:
            self.sqrt = sqrt_method

        if subtract is None:
            self.subtract = np.subtract
        else:
            self.subtract = subtract

        self._compute_weights()


    def num_sigmas(self):
        """ Number of sigma points for each variable in the state x"""
        return 2*self.n + 1


    def sigma_points(self, x, P):
        r""" Computes the sigma points for an unscented Kalman filter
        given the mean (x) and covariance(P) of the filter.
        kappa is an arbitrary constant. Returns sigma points.

        Works with both scalar and array inputs:
        sigma_points (5, 9, 2) # mean 5, covariance 9
        sigma_points ([5, 2], 9*eye(2), 2) # means 5 and 2, covariance 9I

        Parameters
        ----------

        x : array-like object of the means of length n
            Can be a scalar if 1D.
            examples: 1, [1,2], np.array([1,2])

        P : scalar, or np.array
           Covariance of the filter. If scalar, is treated as eye(n)*P.

        kappa : float
            Scaling factor.

        Returns
        -------

        sigmas : np.array, of size (n, 2n+1)
            2D array of sigma points :math:`\chi`. Each column contains all of
            the sigmas for one dimension in the problem space. They
            are ordered as:

            .. math::
                :nowrap:

                \begin{eqnarray}
                  \chi[0]    = &x \\
                  \chi[1..n] = &x + [\sqrt{(n+\kappa)P}]_k \\
                  \chi[n+1..2n] = &x - [\sqrt{(n+\kappa)P}]_k
                \end{eqnarray}

        """

        if self.n != np.size(x):
            raise ValueError("expected size(x) {}, but size is {}".format(
                self.n, np.size(x)))

        n = self.n

        if np.isscalar(x):
            x = np.asarray([x])

        n = np.size(x)  # dimension of problem

        if np.isscalar(P):
            P = np.eye(n) * P
        else:
            P = np.atleast_2d(P)

        sigmas = np.zeros((2*n+1, n))

        # implements U'*U = (n+kappa)*P. Returns lower triangular matrix.
        # Take transpose so we can access with U[i]
        try:
            U = self.sqrt((n + self.kappa) * P)
            # print("cholesky worked")
            
        except:
            print("scipy.linalg.cholesky((n+kappa)*P failed. Have \n",
                  f"n: {n}\n",
                  f"kappa: {self.kappa}\n",
                  f"P: {P}\n",
                  f"eig(P): {np.linalg.eigvalsh(P)}")
            # print(P)
            # print(np.linalg.eigvalsh(P))
            raise ValueError
            
        sigmas[0] = x
        for k in range(n):
            # pylint: disable=bad-whitespace
            sigmas[k+1]   = self.subtract(x, -U[k])
            sigmas[n+k+1] = self.subtract(x, U[k])
        return sigmas


    def _compute_weights(self):
        """ Computes the weights for the unscented Kalman filter. In this
        formulation the weights for the mean and covariance are the same.
        """

        n = self.n
        k = self.kappa

        self.Wm = np.full(2*n+1, .5 / (n + k))
        self.Wm[0] = k / (n+k)
        self.Wc = self.Wm


    def __repr__(self):

        return '\n'.join([
            'JulierSigmaPoints object',
            helpers.pretty_str('n', self.n),
            helpers.pretty_str('kappa', self.kappa),
            helpers.pretty_str('Wm', self.Wm),
            helpers.pretty_str('Wc', self.Wc),
            helpers.pretty_str('subtract', self.subtract),
            helpers.pretty_str('sqrt', self.sqrt)
            ])


class SimplexSigmaPoints(object):

    """
    Generates sigma points and weights according to the simplex
    method presented in [1].

    Parameters
    ----------

    n : int
        Dimensionality of the state. n+1 weights will be generated.

    sqrt_method : function(ndarray), default=scipy.linalg.cholesky
        Defines how we compute the square root of a matrix, which has
        no unique answer. Cholesky is the default choice due to its
        speed. Typically your alternative choice will be
        scipy.linalg.sqrtm

        If your method returns a triangular matrix it must be upper
        triangular. Do not use numpy.linalg.cholesky - for historical
        reasons it returns a lower triangular matrix. The SciPy version
        does the right thing.

    subtract : callable (x, y), optional
        Function that computes the difference between x and y.
        You will have to supply this if your state variable cannot support
        subtraction, such as angles (359-1 degreees is 2, not 358). x and y
        are state vectors, not scalars.

    Attributes
    ----------

    Wm : np.array
        weight for each sigma point for the mean

    Wc : np.array
        weight for each sigma point for the covariance

    References
    ----------

    .. [1] Phillippe Moireau and Dominique Chapelle "Reduced-Order
           Unscented Kalman Filtering with Application to Parameter
           Identification in Large-Dimensional Systems"
           DOI: 10.1051/cocv/2010006
    """

    def __init__(self, n, alpha=1, sqrt_method=None, subtract=None):
        self.n = n
        self.alpha = alpha
        if sqrt_method is None:
            self.sqrt = cholesky
        else:
            self.sqrt = sqrt_method

        if subtract is None:
            self.subtract = np.subtract
        else:
            self.subtract = subtract

        self._compute_weights()


    def num_sigmas(self):
        """ Number of sigma points for each variable in the state x"""
        return self.n + 1


    def sigma_points(self, x, P):
        """
        Computes the implex sigma points for an unscented Kalman filter
        given the mean (x) and covariance(P) of the filter.
        Returns tuple of the sigma points and weights.

        Works with both scalar and array inputs:
        sigma_points (5, 9, 2) # mean 5, covariance 9
        sigma_points ([5, 2], 9*eye(2), 2) # means 5 and 2, covariance 9I

        Parameters
        ----------

        x : An array-like object of the means of length n
            Can be a scalar if 1D.
            examples: 1, [1,2], np.array([1,2])

        P : scalar, or np.array
           Covariance of the filter. If scalar, is treated as eye(n)*P.

        Returns
        -------

        sigmas : np.array, of size (n, n+1)
            Two dimensional array of sigma points. Each column contains all of
            the sigmas for one dimension in the problem space.

            Ordered by Xi_0, Xi_{1..n}
        """

        if self.n != np.size(x):
            raise ValueError("expected size(x) {}, but size is {}".format(
                self.n, np.size(x)))

        n = self.n

        if np.isscalar(x):
            x = np.asarray([x])
        x = x.reshape(-1, 1)

        if np.isscalar(P):
            P = np.eye(n) * P
        else:
            P = np.atleast_2d(P)

        U = self.sqrt(P)

        lambda_ = n / (n + 1)
        Istar = np.array([[-1/np.sqrt(2*lambda_), 1/np.sqrt(2*lambda_)]])

        for d in range(2, n+1):
            row = np.ones((1, Istar.shape[1] + 1)) * 1. / np.sqrt(lambda_*d*(d + 1)) # pylint: disable=unsubscriptable-object
            row[0, -1] = -d / np.sqrt(lambda_ * d * (d + 1))
            Istar = np.r_[np.c_[Istar, np.zeros((Istar.shape[0]))], row] # pylint: disable=unsubscriptable-object

        I = np.sqrt(n)*Istar
        scaled_unitary = (U.T).dot(I)

        sigmas = self.subtract(x, -scaled_unitary)
        return sigmas.T


    def _compute_weights(self):
        """ Computes the weights for the scaled unscented Kalman filter. """

        n = self.n
        c = 1. / (n + 1)
        self.Wm = np.full(n + 1, c)
        self.Wc = self.Wm


    def __repr__(self):
        return '\n'.join([
            'SimplexSigmaPoints object',
            helpers.pretty_str('n', self.n),
            helpers.pretty_str('alpha', self.alpha),
            helpers.pretty_str('Wm', self.Wm),
            helpers.pretty_str('Wc', self.Wc),
            helpers.pretty_str('subtract', self.subtract),
            helpers.pretty_str('sqrt', self.sqrt)
            ])

class SigmaPointsBase():
    """
    Parent class when sigma points algorithms are constructed. All points are used to estimate mean and covariance of Y, where Y=f(X)
    """
    def __init__(self, n, sqrt_method=None):
        """
        Init

        Parameters
        ----------
        n : TYPE int
            DESCRIPTION. Dimension of x
        sqrt_method : TYPE, optional function
            DESCRIPTION. The default is None. Method to calculate the square root of a matrix. If None is supplied, scipy.linalg.cholesky is used

        Returns
        -------
        None.

        """
        self.n = n
        if sqrt_method is None:
            self.sqrt = cholesky 
        else:
            self.sqrt = sqrt_method
        
    def num_sigmas(self):
        """
        Returns the number of sigma points. Most algorithms return (2n+1) points, can be overwritten by child class

        Returns
        -------
        TYPE int
            DESCRIPTION. dim_sigma, number of sigma points

        """
        return 2*self.n + 1
    
    def is_matrix_pos_def(self, a_matrix):
        """
        Checks if a matrix is positive definite

        Parameters
        ----------
        a_matrix : TYPE np.array((n,n))
            DESCRIPTION. A matrix

        Returns
        -------
        TYPE bool
            DESCRIPTION. True if the matrix is pos def, else False

        """
        return np.all(np.linalg.eigvals(a_matrix) > 0)
class GenUTSigmaPoints(SigmaPointsBase):
    """
    Implement the sigma points as described by Ebeigbe. Distributions does NOT need to be symmetrical.
    
    @article{EbeigbeDonald2021AGUT,
abstract = {The unscented transform uses a weighted set of samples called sigma points to propagate the means and covariances of nonlinear transformations of random variables. However, unscented transforms developed using either the Gaussian assumption or a minimum set of sigma points typically fall short when the random variable is not Gaussian distributed and the nonlinearities are substantial. In this paper, we develop the generalized unscented transform (GenUT), which uses adaptable sigma points that can be positively constrained, and accurately approximates the mean, covariance, and skewness of an independent random vector of most probability distributions, while being able to partially approximate the kurtosis. For correlated random vectors, the GenUT can accurately approximate the mean and covariance. In addition to its superior accuracy in propagating means and covariances, the GenUT uses the same order of calculations as most unscented transforms that guarantee third-order accuracy, which makes it applicable to a wide variety of applications, including the assimilation of observations in the modeling of the coronavirus (SARS-CoV-2) causing COVID-19.},
journal = {ArXiv},
year = {2021},
title = {A Generalized Unscented Transformation for Probability Distributions},
language = {eng},
address = {United States},
author = {Ebeigbe, Donald and Berry, Tyrus and Norton, Michael M and Whalen, Andrew J and Simon, Dan and Sauer, Timothy and Schiff, Steven J},
issn = {2331-8422},
}


    """
    def __init__(self, n, sqrt_method=None):
        """
        Init

        Parameters
        ----------
        n : TYPE int
            DESCRIPTION. Dimension of x
        sqrt_method : TYPE, optional function
            DESCRIPTION. The default is None. Method to calculate the square root of a matrix. If None is supplied, scipy.linalg.cholesky is used

        Returns
        -------
        None.

        """
        super().__init__(n, sqrt_method = sqrt_method)
        self.dim_sigma = self.num_sigmas()
        
    def compute_scaling_and_weights(self, P, S, K, s1 = None):
        """
        Computes the scaling parameters s and the weights w

        Parameters
        ----------
        P : TYPE np.array(n,n)
            DESCRIPTION. Covariance of X
        S : TYPE np.array(n,)
            DESCRIPTION. 3rd central moment of X. Can be computed by scipy.stats.moments(data, moment=3)
        K : TYPE np.array(n,)
            DESCRIPTION. 4th central moment of X. Can be computed by scipy.stats.moments(data, moment=4)
        s1 : TYPE, optional np.array(n,)
            DESCRIPTION. The default is None. First part of scaling arrays. s1> 0 for every element. If None, algorithm computes the suggested values in the article.

        Raises
        ------
        ValueError
            DESCRIPTION. Dimension mismatch

        Returns
        -------
        s : TYPE np.array(2n,)
            DESCRIPTION. Scaling values
        w : TYPE np.array(dim_sigma,)
            DESCRIPTION. Weights for every sigma points.

        """
        
        sigma = np.sqrt(np.diag(P)) #standard deviation of each state
        S_std = np.divide(S, np.power(sigma, 3))
        K_std = np.divide(K, np.power(sigma, 4))
        
        if s1 is None: #create s (s.shape = (n,))
            s1 = self.select_s1_to_match_kurtosis(S_std, K_std)
        
        if (s1.shape[0] != S.shape[0]):
            raise ValueError("Dimension of s is wrong")
        
        #create the next values for s, total dim is 2n+1
        s2 = s1 + S_std
        w2 = np.divide(1, np.multiply(s2, (s1 + s2)))
        w1 = np.multiply(np.divide(s2, s1), w2)
        w = np.concatenate((np.array([0]), w1, w2))
        w[0] = 1 - np.sum(w[1:])
        s = np.concatenate((s1, s2))
        return s, w
    
    def select_s1_to_match_kurtosis(self, S_std, K_std):
        """
        Computes the first part of the scaling array by the method suggested in the paper.

        Parameters
        ----------
        S_std : TYPE np.array(n,)
            DESCRIPTION. Scaled 3rd central moment
        K_std : TYPE np.array(n,)
            DESCRIPTION. Scaled 4th central moment

        Returns
        -------
        s1 : TYPE np.array(n,)
            DESCRIPTION. First part of the scaling value array

        """
        s1 = .5*(-S_std + np.sqrt(4*K_std - 3*np.square(S_std)))
        return s1
    
    def sigma_points(self, mu, P, s):
        """
        Computes the sigma points

        Parameters
        ----------
        mu : TYPE np.array(n,)
            DESCRIPTION. Mean value of X 
        P : TYPE np.array(n,n)
            DESCRIPTION. Covariance matrix of X

        Raises
        ------
        ValueError
            DESCRIPTION. Shapes are wrong
        LinAlgError
            DESCRIPTION. P is not positiv definite and symmetric

        Returns
        -------
        sigmas : TYPE np.array(n, dim_sigma)
            DESCRIPTION. sigma points
        P_sqrt : TYPE np.array(n,n)
            DESCRIPTION. sqrt(P). Can be inspected if something goes wrong.

        """
        if not self.n == mu.shape[0]:
            raise ValueError(f" self.n = {self.n} while mu.shape = {mu.shape}. mu.shape[0] must match self.n!")
        
        if not ((self.n == P.shape[0]) and (self.n == P.shape[1])):
            raise ValueError(f"P.shape = {P.shape}, it must be ({self.n, self.n})")
        
        n = self.n
        dim_sigma = self.dim_sigma
        
        sigmas = np.zeros((n, dim_sigma))
        sigmas[:, 0] = mu
        
        try:
            P_sqrt = self.sqrt(P)
        except np.linalg.LinAlgError as LinAlgError:
            print(f"P is not positive definite. Current value is P = {P}")
            raise LinAlgError
        
        for i in range(n):
            sigmas[:, 1 + i] = mu - s[i]*P_sqrt[i, :]
            sigmas[:, 1 + n + i] = mu + s[n + i]*P_sqrt[i, :]
        
        return sigmas, P_sqrt

class ParametricUncertaintyGenUTSigmaPoints(GenUTSigmaPoints):
    """
    Uses the generic sigma points "GenUTSigmaPoints" in a special way, tailor made for the tuning of noise matrices by GenUT with P_par_x estimated.
    
    xa = np.array([par, x]), where x are the states I want to estimate and par are uncertain parameters determining the noise.
    """
    def __init__(self, par_cov, dim_x, cm3_par, cm4_par, sqrt_method = None, block_factorization = True):
        """
        

        Parameters
        ----------
        par_cov : TYPE np.array((dim_par, dim_par))
            DESCRIPTION. Covariance matrix
        dim_x : TYPE int
            DESCRIPTION. Number of states to be estimated
        cm3_par : TYPE np.array((dim_par,))
            DESCRIPTION. 3rd central moment ("diagonal" terms) of the parameters
        cm4_par : TYPE np.array((dim_par,))
            DESCRIPTION. 3rd central moment ("diagonal" terms) of the parameters
        sqrt_method : TYPE, optional function
            DESCRIPTION. The default is None. Matrix square root method. Default method of parent class is (upper) Cholesky factorization

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.

        """
       
        dim_par = par_cov.shape[0]
        if not ((dim_par == par_cov.shape[1]) and 
                (dim_par == cm3_par.shape[0]) and (dim_par == cm3_par.shape[0])):
            raise ValueError("Input dimension of covariance matrix, cm3_par or cm4_par are incorrect")
            
        dim_xa = dim_par + dim_x #augmented state dimension
        
        #factors required for block cholesky decomposition
        U_par_cov = scipy.linalg.cholesky(par_cov, lower = False) #want upper (also default choice, so the kwarg lower=False is redundant)
        
        U_par_cov_inv = np.linalg.inv(U_par_cov)
        
        #Cholesky: A = U_A.T @ U_A
        #Inverse rule: C = A@B ==> C^-1 = (A@B)^-1 = (B^-1)@(A^-1) for non-singular A,B- Hence, for the Cholesky decomposition: A^-1 = (U_A.T @ U_A)^-1 = (U_A)^-1 @ (U_A.T)^-1 = (U_A^-1) @ (U_A^-1).T 
        par_cov_inv = U_par_cov_inv @ U_par_cov_inv.T
        
        cm3_x = np.zeros(dim_x) #according to Gaussian assumption of x
        
        self.dim_par = dim_par
        self.dim_x = dim_x
        self.U_par_cov = U_par_cov
        self.par_cov_inv = par_cov_inv
        self.U_par_cov_inv = U_par_cov_inv
        self.cm3_par = cm3_par
        self.cm4_par = cm4_par
        self.cm3 = np.hstack((cm3_par, cm3_x)) #for the augmented state vector
        self.block_factorization = block_factorization #if True, use the proposed (fast) block factorization
        # zeros_block = np.zeros((self.dim_par, self.dim_x))
        # self.U_block = np.hstack((self.U_par_cov, zeros_block)) #constant block in block factorization method of finding P_sqrt
        self.P_sqrt = np.zeros((dim_xa, dim_xa))
        self.P_sqrt[:dim_par, :dim_par] = self.U_par_cov
        super().__init__(dim_xa, sqrt_method = sqrt_method) #__init__ in GenUTSigmaPoints
    
    def _compute_weights(self, P):
        self._compute_cm4(P)
        scaling, weights = self.compute_scaling_and_weights(P, self.cm3, self.cm4)
        self.s = scaling
        self.Wm = weights
        self.Wc = weights
    
    def sigma_points(self, mu, P):
        """
        Computes the generalized sigma points and weights, based on the "joint genUT" formulation. It uses that xa = [par, x].T and leverages that i) the distribution par is fixed and does not change ==> mean, covariance, cm3_par and cm4_par does not change and ii) x is assumed to be normal distribution ==> cm3_x=0 and cm4_x can be calculated by Isserli's theorem

        Parameters
        ----------
        mu : TYPE np.array(n,), where n == dim_xa
            DESCRIPTION. Mean value of X 
        P : TYPE np.array(n,n)
            DESCRIPTION. Covariance matrix of X

        Raises
        ------
        ValueError
            DESCRIPTION. Shapes are wrong
        LinAlgError
            DESCRIPTION. P is not positiv definite and symmetric

        Returns
        -------
        sigmas : TYPE np.array(n, dim_sigma)
            DESCRIPTION. sigma points
        P_sqrt : TYPE np.array(n,n)
            DESCRIPTION. sqrt(P). Can be inspected if something goes wrong.

        """
        if not self.n == mu.shape[0]:
            raise ValueError(f" self.n = {self.n} while mu.shape = {mu.shape}. mu.shape[0] must match self.n!")
        
        if not ((self.n == P.shape[0]) and (self.n == P.shape[1])):
            raise ValueError(f"P.shape = {P.shape}, it must be ({self.n, self.n})")
        
        n = self.n
        dim_sigma = self.dim_sigma
        
        self._compute_weights(P) #must be run here such that self.s is correct
        s = self.s
        
        sigmas = np.zeros((n, dim_sigma))
        sigmas[:, 0] = mu
        
        try:
            if self.block_factorization:
                #For notation, see Block Cholesky function
                P_xpar = P[self.dim_par:, :self.dim_par]
                # P_xpar = np.zeros(P_xpar.shape) #WRONG but can be useful if P_par is ill-conditioned (get numerical issues)
                P_xx = P[-self.dim_x:, -self.dim_x:]
                # P_sqrt = block_cholesky(self.U_par_cov,
                #                         self.U_par_cov_inv,
                #                         self.par_cov_inv,
                #                         P_xpar,
                #                         P_xx)
                
                S = P_xx - P_xpar @ self.par_cov_inv @ P_xpar.T #Schur complement
                # print(f"block cholesky in progress. dim_S = {S.shape}. Cond(S) = {np.linalg.cond(S)}")
                # U_S = scipy.linalg.cholesky(S, lower = False)
                # P_sqrt = np.block([[self.U_block],
                #                 [P_xpar @ self.U_par_cov_inv.T, U_S]])
                P_sqrt = self.P_sqrt
                P_sqrt[self.dim_par:, :self.dim_par] = P_xpar @ self.U_par_cov_inv.T
                P_sqrt[self.dim_par:, self.dim_par:] = scipy.linalg.cholesky(S, lower = False)
                
                # P1 = P_sqrt.T @ P_sqrt
                # P2 = P_sqrt2.T @ P_sqrt
                # print(f"norm P-P1: {np.linalg.norm(P-P1)}\n",
                #       f"norm P1-P2: {np.linalg.norm(P1-P2)}")
            else:
                # print(f"normal choelsky: dim_P = {P.shape}, cond(P) = {np.linalg.cond(P)}")
                # raise ValueError
                P_sqrt = self.sqrt(P)
            
        except scipy.linalg.LinAlgError as LinAlgError:
            if self.block_factorization:
                print(f"P or S is not positive definite. Condition number of P is {np.linalg.cond(P)} and S is {np.linalg.cond(S)}")
            else:
                print(f"P is not positive definite. Condition number of P is {np.linalg.cond(P)}")
            raise LinAlgError
        
        for i in range(n):
            sigmas[:, 1 + i] = mu - s[i]*P_sqrt[i, :]
            sigmas[:, 1 + n + i] = mu + s[n + i]*P_sqrt[i, :]
        
        return sigmas.T #JulierSigmaPoints/MerweScaledSigmaPoints returns this way. Parent class GenUTSigmaPoints returns the transpose
    
    
    def _compute_cm4(self, P):
        """
        Calculates the 4th central moment for the augmented state vector xa = [par, x].T. The 4th central moments for par is fixed and known, while the 4th central moment for x is calculated by Isserli's theorem which relates the 4th central moment to the covariance matrix for a normal distribution (can generally calculate any central moment for normal distribution, but the general formula is not implemented here) 

        Parameters
        ----------
        P : TYPE np.array((dim_xa, dim_xa))
            DESCRIPTION. Covariance matrix

        Returns
        -------
        None. Saves self.cm4

        """
        
        #get values for specifying normal distribution for x-variables
        P_xx = P[-self.dim_x:, -self.dim_x:]
        cm4_x = self.compute_cm4_isserli_for_multivariate_normal(P_xx)
        self.cm4 = np.hstack((self.cm4_par, cm4_x))
        
    def compute_cm4_isserli_for_multivariate_normal(self, P):
        """
        Calculate 4th central moment from Isserli's theorem based on Equation 2.42 in 
        Barfoot, T. (2017). State Estimation for Robotics. Cambridge: Cambridge University Press. doi:10.1017/9781316671528

        Parameters
        ----------
        P : TYPE np.array((dim_x, dim_x))
            DESCRIPTION. Covariance matrix

        Returns
        -------
        cm4 : TYPE np.array((dim_x,))
            DESCRIPTION. 4th central moment of an multivariate normal distribution ("diagonal" terms)
        """
        dim_x = P.shape[0]
        # print(Px.shape)
        I = np.eye(dim_x)
        cm4 = P @ (np.trace(P)*I + 2*P)
        return np.diag(cm4)
        
def block_cholesky(U_A, U_A_inv, A_inv, B, D):
    """
    Cholesky decomposition of full matrix C, by using available cholesky decomposition of it's block matrix A. See https://scicomp.stackexchange.com/questions/5050/cholesky-factorization-of-block-matrices.
    
    Have 
    
    C = [[A, B.T]
         [B, D]]
    Want to find U_C such that U_C.T @ U_C = C (upper Cholesky root) by using available information about A (it's cholesky decomposition and inverses)

    Parameters
    ----------
    U_A : TYPE np.array((dim_a, dim_a))
        DESCRIPTION. Cholesky decomposition of A. Have A = U_A.T @ U_A
    U_A_inv : TYPE np.array((dim_a, dim_a))
        DESCRIPTION. Inverse of U_A
    A_inv : TYPE np.array((dim_a, dim_a))
        DESCRIPTION. Inverse of A
    B : TYPE np.array((dim_d, dim_a))
        DESCRIPTION. Block matrix
    D : TYPE np.array((dim_d, dim_d))
        DESCRIPTION. Block matrix. Need to do Cholesky decomposition of this dimensionality to compute U_C. Note that dim_d < dim_c, so this is more efficienct than computing U_C directly.

    Returns
    -------
    U_C : TYPE np.array((dim_c, dim_c))
        DESCRIPTION. Upper Cholesky decomposition of the matrix C

    """
    
    S = D - B @ A_inv @ B.T #Schur complement
    # print(f"block cholesky in progress. dim_S = {S.shape}. Cond(S) = {np.linalg.cond(S)}")
    U_S = scipy.linalg.cholesky(S, lower = False) #upper is also default choice, so this is redundant

    return np.block([[U_A, np.zeros((U_A.shape[0], U_S.shape[1]))],
                    [B @ U_A_inv.T, U_S]])
