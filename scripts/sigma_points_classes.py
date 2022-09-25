# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 15:54:11 2021

@author: halvorak
"""

import numpy as np
import scipy.stats
# import matplotlib.pyplot as plt
# import colorcet as cc
# import pathlib
# import os
import scipy.linalg
# import matplotlib.patches as plt_patches
# import matplotlib.transforms as plt_transforms


class SigmaPoints():
    """
    Parent class when sigma points algorithms are constructed. All points tru to estimate mean and covariance of Y, where Y=f(X)
    """
    def __init__(self, n, sqrt_method = scipy.linalg.sqrtm):
        """
        Init

        Parameters
        ----------
        n : TYPE int
            DESCRIPTION. Dimension of x
        sqrt_method : TYPE, optional function
            DESCRIPTION. The default is scipy.linalg.sqrtm (principal matrix square root). Method to calculate the square root of a matrix. The other choice is typically np.linalg.cholesky

        Returns
        -------
        None.

        """
        self.n = n
        self.sqrt = sqrt_method
        
    def num_sigma_points(self):
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
        Checks if a matrix is positive definite by checking if all eigenvalues are positive

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
    

    
class JulierSigmaPoints(SigmaPoints):
    """
    Implement the sigma points as described by Julier's original paper. It assumes that the distribtions are symmetrical.
    
    @TECHREPORT{Julier96ageneral,
    author = {Simon Julier and Jeffrey K. Uhlmann},
    title = {A General Method for Approximating Nonlinear Transformations of Probability Distributions},
    institution = {},
    year = {1996}
}
    
    """
    def __init__(self, n, kappa = 0., sqrt_method = scipy.linalg.sqrtm):
        """
        Init

        Parameters
        ----------
        n : TYPE int
            DESCRIPTION. Dimension of x
        sqrt_method : TYPE, optional function
            DESCRIPTION. The default is scipy.linalg.sqrtm (principal matrix square root). Method to calculate the square root of a matrix. The other choice is typically np.linalg.cholesky
        kappa : TYPE, optional float
            DESCRIPTION. The default is 0. If set to (n-3), you minimize error in higher order terms.


        Returns
        -------
        None.

        """
        super().__init__(n, sqrt_method = sqrt_method)
        # print("WARNING: This class has NOT been verified yet")
        # raise ValueError("This class has NOT been verified yet!")
        if not (kappa == (3-n)):
            print(f"warning: kappa is not set to kappa = (3-n) = {3-n}, which minimizes the fourth order mismatch. Proceeding with a value of kappa = {kappa}")
        self.kappa = kappa
        self.dim_sigma = self.num_sigma_points()
        self.Wm = self.compute_weights()
        self.Wc = self.Wm.copy()
        
    # def compute_weights(self)
    def compute_sigma_points(self, mu, P):
        """
        Computes the sigma points based on Julier's paper

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
            DESCRIPTION. sqrt((n+kappa)P). Can be inspected if something goes wrong.

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
            sqrt_factor = np.sqrt(n+self.kappa)
            P_sqrt = self.sqrt(P)
            P_sqrt_weight = sqrt_factor*P_sqrt
        except np.linalg.LinAlgError as LinAlgError:
            print(f"(n+kappa)P is not positive definite. Current value is (n+kappa)P = {(n+self.kappa)*P}")
            raise LinAlgError
        
        for i in range(n):
            sigmas[:, 1 + i] = mu + P_sqrt_weight[:, i]
            sigmas[:, 1 + n + i] = mu - P_sqrt_weight[:, i]
        
        return sigmas, self.Wm, self.Wc, P_sqrt
        
    def compute_weights(self):
        """
        Computes the weights

        Returns
        -------
        weights : TYPE np.array(dim_sigma,)
            DESCRIPTION. Weights for every sigma points

        """
        n = self.n
        dim_sigma = self.dim_sigma
        
        weights = np.array([1/(2*(n + self.kappa)) for i in range(dim_sigma)])
        weights[0] = self.kappa/(n + self.kappa)
        return weights

class ScaledSigmaPoints(SigmaPoints):
    """
    From
    JULIER, S. J. The Scaled Unscented Transformation.  Proceedings of the American Control Conference, 2002 2002 Anchorage. 4555-4559 vol.6.

    """
    
    def __init__(self, n, alpha = 1e-3, beta = 2., kappa = 0., sqrt_method = scipy.linalg.sqrtm):
        """
        Init

        Parameters
        ----------
        n : TYPE int
            DESCRIPTION. Dimension of x
        sqrt_method : TYPE, optional function
            DESCRIPTION. The default is scipy.linalg.sqrtm (principal matrix square root). Method to calculate the square root of a matrix. The other choice is typically np.linalg.cholesky
        kappa : TYPE, optional float
            DESCRIPTION. The default is 0. If set to (n-3), you minimize error in higher order terms.


        Returns
        -------
        None.

        """
        super().__init__(n, sqrt_method = sqrt_method)
        # print("WARNING: This class has NOT been verified yet")
        # raise ValueError("This class has NOT been verified yet!")
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lam = self.calculate_lam()
        if not (kappa == (3-n)):
            print(f"warning: kappa is not set to kappa = (3-n) = {3-n}, which minimizes the fourth order mismatch. Proceeding with a value of kappa = {kappa}")
        self.dim_sigma = self.num_sigma_points()
        self.Wm, self.Wc = self.compute_weights()
    
    def calculate_lam(self):
        lam = (self.alpha**2)*(self.n + self.kappa) - self.n
        return lam
        
    def compute_weights(self):
        """
        Computes the weights

        Returns
        -------
        weights : TYPE np.array(dim_sigma,)
            DESCRIPTION. Weights for every sigma points

        """
        n = self.n
        dim_sigma = self.dim_sigma
        alpha = self.alpha
        beta = self.beta
        
        lam = self.calculate_lam()
        
        Wm = np.array([1/(2*(n + lam)) for i in range(dim_sigma)])
        Wc = Wm.copy()
        
        Wm[0] = lam/(lam + n)
        Wc[0] = lam/(lam + n) + (1 - alpha**2 + beta)
        return Wm, Wc
    
    def compute_sigma_points(self, mu, P):
        """
        Computes the sigma points based on Julier's paper

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
            DESCRIPTION. sqrt((n+kappa)P). Can be inspected if something goes wrong.

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
            sqrt_factor = np.sqrt(n+self.lam)
            P_sqrt = self.sqrt(P)
            P_sqrt_weight = sqrt_factor*P_sqrt
        except np.linalg.LinAlgError as LinAlgError:
            print(f"(n+kappa)P is not positive definite. Current value is (n+kappa)P = {(n+self.kappa)*P}")
            raise LinAlgError
        
        for i in range(n):
            sigmas[:, 1 + i] = mu + P_sqrt_weight[:, i]
            sigmas[:, 1 + n + i] = mu - P_sqrt_weight[:, i]
        
        return sigmas, self.Wm, self.Wc, P_sqrt
        
        
        
class GenUTSigmaPoints(SigmaPoints):
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
    def __init__(self, n, sqrt_method = scipy.linalg.sqrtm):
        """
        Init

        Parameters
        ----------
        n : TYPE int
            DESCRIPTION. Dimension of x
        sqrt_method : TYPE, optional function
            DESCRIPTION. The default is scipy.linalg.sqrtm (principal matrix square root). Method to calculate the square root of a matrix. The other choice is typically np.linalg.cholesky

        Returns
        -------
        None.

        """
        super().__init__(n, sqrt_method = sqrt_method)
        self.dim_sigma = self.num_sigma_points()
        
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
    
    # def compute_sigma_points(self, mu, P, s):
    def compute_sigma_points(self, mu, P, S = None, K = None, s1 = None, sqrt_method = None):
        """
        Computes the sigma points

        Parameters
        ----------
        mu : TYPE np.array(n,)
            DESCRIPTION. Mean value of X 
        P : TYPE np.array(n,n)
            DESCRIPTION. Covariance matrix of X
        S : TYPE np.array(n,) if None, assumed symmetric distribution (0)
            DESCRIPTION. 3rd central moment of X. Can be computed by scipy.stats.moments(data, moment=3)
        K : TYPE np.array(n,) If none, assumed Gaussian distribution
            DESCRIPTION. 4th central moment of X. Can be computed by scipy.stats.moments(data, moment=4)
        s1 : TYPE, optional np.array(n,)
            DESCRIPTION. The default is None. First part of scaling arrays. s1> 0 for every element. If None, algorithm computes the suggested values in the article.

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
        W : TYPE np.array(n,)
            DESCRIPTION. Weights for the sigma points
        P_sqrt : TYPE np.array(n,n)
            DESCRIPTION. sqrt(P). Can be inspected if something goes wrong.

        """
        if not self.n == mu.shape[0]:
            raise ValueError(f" self.n = {self.n} while mu.shape = {mu.shape}. mu.shape[0] must match self.n!")
        
        if not ((self.n == P.shape[0]) and (self.n == P.shape[1])):
            raise ValueError(f"P.shape = {P.shape}, it must be ({self.n, self.n})")
        
        if sqrt_method is None:
            sqrt_method = self.sqrt
        
        n = self.n #dimension of x
        
        if S is None: #assumed symmetric distribution
            S = np.zeros((n,))
        if K is None: #assume Gaussian distribution
            K = self.compute_cm4_isserlis_for_multivariate_normal(P)
        
        dim_sigma = self.dim_sigma
        
        sigmas = np.zeros((n, dim_sigma))
        sigmas[:, 0] = mu
        
        self.P_sqrt = sqrt_method(P)
        
        #compute scaling and weights
        self.s, Wm = self.compute_scaling_and_weights(P, S, K, s1 = s1)
        Wc = Wm.copy()
        
        for i in range(n):
            sigmas[:, 1 + i] = mu - self.s[i]*self.P_sqrt[:, i]
            sigmas[:, 1 + n + i] = mu + self.s[n + i]*self.P_sqrt[:, i]
        self.sigmas = sigmas
        self.Wm = Wm
        self.Wc = Wc
        return sigmas, Wm, Wc, self.P_sqrt
    
    def compute_cm4_isserlis_for_multivariate_normal(self, P):
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


#Sigma points are implemented in myfilter-directory

# class ParametricUncertaintyGenUTSigmaPoints(GenUTSigmaPoints):
    
#     def __init__(self, dim_par, dim_x, cm3_par, cm4_par, sqrt_method = None):
#         """
        

#         Parameters
#         ----------
#         dim_par : TYPE int
#             DESCRIPTION. Number of parameters in the distrbution
#         dim_x : TYPE int
#             DESCRIPTION. Number of states to be estimated
#         cm3_par : TYPE np.array((dim_par,))
#             DESCRIPTION. 3rd central moment ("diagonal" terms) of the parameters
#         cm4_par : TYPE np.array((dim_par,))
#             DESCRIPTION. 3rd central moment ("diagonal" terms) of the parameters
#         sqrt_method : TYPE, optional function
#             DESCRIPTION. The default is None. Matrix square root method. Default method of parent class is (upper) Cholesky factorization

#         Returns
#         -------
#         None.

#         """
#         dim_xa = dim_par + dim_x #augmented state dimension
#         self.dim_par = dim_par
#         self.dim_x = dim_x
#         self.cm3_par = cm3_par
#         self.cm4_par = cm4_par
#         super().__init__(dim_xa, sqrt_method = sqrt_method)
    
#     def _compute_weights(self, mu, P):
#         self._compute_cm3_cm4(mu, P)
#         scaling, weights = self.compute_scaling_and_weights(self.P, self.cm3, self.cm4)
#         self.s = scaling
#         self.Wm = weights
#         self.Wc = weights
    
#     def compute_sigma_points(self, mu, P):
#         self._compute_weights(mu, P) #must be run such that self.s is correct
#         sigmas, P_sqrt = super().compute_sigma_points(mu, P, self.s) #rewrite this later to get computational savings
#         return sigmas.T #Julier/Merwe returns this way. Original GenUTSigmaPoints returns the transpose
    
#     def _compute_cm3_cm4(self, mu, P, sample_size = int(5e3)):
#         """
#         Calculates 3rd and 4th central moment

#         Parameters
#         ----------
#         mu : TYPE
#             DESCRIPTION.
#         P : TYPE
#             DESCRIPTION.
#         sample_size : TYPE, optional
#             DESCRIPTION. The default is int(5e3).

#         Raises
#         ------
#         TypeError
#             DESCRIPTION.

#         Returns
#         -------
#         None.

#         """
#         if not isinstance(sample_size, int):
#             raise TypeError("Sample size must be an integer")
        
#         #get values for specifying normal distribution for x-variables
#         Px = P[-self.dim_x:, -self.dim_x:]
#         mu_x = mu[-self.dim_x:]
        
#         #Make samples and calculate the required central moments
#         samples_x = scipy.stats.multivariate_normal(mean = mu_x, cov = Px).rvs(size = sample_size)
#         cm3_x = scipy.stats.moment(samples_x, moment=3)
#         cm4_x = scipy.stats.moment(samples_x, moment=4)
#         self.cm3 = np.hstack((self.cm3_par, cm3_x))
#         self.cm4 = np.hstack((self.cm4_par, cm4_x))
        
        
        

# def unscented_transform(sigmas, w, fx = None):
#     """
#     Calculates mean and covariance of a nonlinear function by the unscented transform. Every sigma point is propagated through the function, and combined with their weights the mean and covariance is calculated.

#     Parameters
#     ----------
#     sigmas : TYPE np.ndarray(n, dim_sigma)
#         DESCRIPTION. Array of sigma points. Each column contains a sigma point
#     w : TYPE np.array(dim_sigma,)
#         DESCRIPTION. Weights of each sigma point.
#     fx : TYPE, optional function
#         DESCRIPTION. The default is None. The non-linear function which the RV is propagated through. If None is supplied, the identity function is used.

#     Returns
#     -------
#     mean : TYPE np.array(n,)
#         DESCRIPTION. Mean value of Y=f(X), X is a random variable (RV) 
#     Py : TYPE np.array(n,n)
#         DESCRIPTION. Covariance matrix, cov(Y) where Y=f(X)

#     """
#     if fx is None:
#         fx = lambda x: x
    
#     (n, dim_sigma) = sigmas.shape
    
#     yi = np.zeros(sigmas.shape)
#     for i in range(dim_sigma):
#         yi[:, i] = fx(sigmas[:, i])
#     mean = np.dot(yi, w)
    
#     Py = np.zeros((n,n))
#     for i in range(dim_sigma):
#         Py += w[i]*np.dot((yi[:, i] - mean).reshape(-1,1), 
#                           (yi[:, i] - mean).reshape(-1,1).T)
    
#     return mean, Py



