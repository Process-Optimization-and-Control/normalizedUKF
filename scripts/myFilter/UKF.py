# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 11:17:56 2022

@author: halvorak
"""

from . import unscented_transform

# from copy import deepcopy
import numpy as np
import scipy.linalg
from numba import jit


class UKFBase():
    r"""
    Base class for UKF implementations


    Parameters
    ----------

    dim_x : int
        Number of state variables for the filter.


    dim_y : int
        Number of of measurements


    hx : function(x,**hx_args)
        Measurement function. Converts state vector x into a measurement
        vector of shape (dim_y,).

    fx : function(x,**fx_args)
        Propagation of states from current time step to the next.

    points : class
        Class which computes the sigma points and weights for a UKF
        algorithm. You can vary the UKF implementation by changing this
        class. 

    sqrt_fn : callable(ndarray), default=scipy.linalg.sqrtm
        Defines how we compute the square root of a matrix, which has
        no unique answer. Principal matrix square root is the default choice. Typically the alternative is Cholesky decomposition. Different choices affect how the sigma points
        are arranged relative to the eigenvectors of the covariance matrix. Daid et al recommends principal matrix square root




    Attributes
    ----------

    x_prior : numpy.array(dim_x)
        Prior (predicted) state estimate. 

    P_prior : numpy.array(dim_x, dim_x)
        Prior (predicted) state covariance matrix. .

    x_post : numpy.array(dim_x)
        Posterior (updated) state estimate. .

    P_post : numpy.array(dim_x, dim_x)
        Posterior (updated) state covariance matrix. .

    R : numpy.array(dim_y, dim_y)
        measurement noise matrix

    Q : numpy.array(dim_x, dim_x)
        process noise matrix

    K : numpy.array
        Kalman gain

    y_res : numpy.array
        innovation residual


    """

    def __init__(self, fx, hx, points_x, Q, R, 
                 w_mean = None, v_mean = None,
                 sqrt_fn=scipy.linalg.sqrtm, name=None):
        """
        Create a Kalman filter. IMPORTANT: Additive white noise is assumed!

        """
        dim_w = Q.shape[0]
        dim_v = R.shape[0]
        Q = np.atleast_2d(Q)
        R = np.atleast_2d(R)
        
        # check inputs
        assert ((dim_w, dim_w) == Q.shape)
        assert ((dim_v, dim_v) == R.shape)
        
        if w_mean is None:
            w_mean = np.zeros((dim_w,))
        
        if v_mean is None:
            v_mean = np.zeros((dim_v,))

        self._dim_w = dim_w
        self._dim_v = dim_v
        # self.x_prior = np.zeros((dim_x,))
        # self.P_prior = np.eye(dim_x)
        # self.x_post = self.x_prior.copy()
        # self.P_post = self.P_prior.copy()
        self.w_mean = w_mean
        self.Q = Q
        self.v_mean = v_mean
        self.R = R
        # self._dim_x = dim_x
        # self._dim_y = dim_y
        self.points_fn_x = points_x
        self._num_sigmas_x = points_x.num_sigma_points()
        self.hx = hx
        self.fx = fx
        self.msqrt = sqrt_fn
        self._name = name  # object name, handy when printing from within class

        # # create sigma-points
        # self.sigmas_raw_fx = np.zeros((self._dim_x, self._num_sigmas_x))
        
        # # sigma-points propagated through fx to form prior distribution
        # self.sigmas_prop = np.zeros((self._dim_x, self._num_sigmas_x))
        
        # # sigma-points based on prior distribution
        # self.sigmas_raw_hx = np.zeros((self._dim_x, self._num_sigmas_x))
        
        # # sigma-points propagated through measurement equation. Form posterior distribution
        # self.sigmas_meas = np.zeros((self._dim_y, self._num_sigmas_x))

        # self.y_res = np.zeros((dim_y, 1))           # residual
        # self.y = np.array([[None]*dim_y]).T  # measurement

    def compute_transformed_sigmas(self, sigmas_in, func, **func_args):
        """
        Send sigma points through a nonlinear function. Call general distribution z, dimension of this variable is dim_z

        Parameters
        ----------
        sigmas_in : TYPE np.array((dim_z, dim_sigmas))
            DESCRIPTION. Sigma points to be propagated
        func : TYPE function(np.array(dim_z,), **func_args). F(dim_z)=>dim_q, q output dimension
            DESCRIPTION. function the sigma points are propagated through
        **func_args : TYPE list, optional
            DESCRIPTION. Additional inputs to func

        Returns
        -------
        sigmas_out : TYPE np.array((dim_q, dim_sigmas))
            DESCRIPTION. Propagated sigma points

        """
        sigmas_out = map(func, sigmas_in.T)
        sigmas_out = np.array(list(sigmas_out)).T
        return sigmas_out

    def cross_covariance(self, x_mean, y_mean, sigmas_x, sigmas_y, W_c):
        """
        Cross-covariance between two probability distribution x,y

        Parameters
        ----------
        x_mean : TYPE np.array(dim_x,)
            DESCRIPTION. Mean of the distribution x
        y_mean : TYPE np.array(dim_y,)
            DESCRIPTION. Mean of the distribution y
        sigmas_x : TYPE np.array((dim_x, dim_sigmas))
            DESCRIPTION. Sigma-points created from the x-distribution
        sigmas_y : TYPE np.array((dim_y, dim_sigmas))
            DESCRIPTION. Sigma-points created from the y-distribution
        W_c : TYPE np.array(dim_sigmas,)
            DESCRIPTION. Weights to compute the covariance

        Returns
        -------
        P_xy : TYPE np.array((dim_x, dim_y))
            DESCRIPTION. Cross-covariance between x and y

        """
        try:
            (dim_x, dim_sigmas_x) = sigmas_x.shape
        except ValueError:  # sigmas_x is 1D
            sigmas_x = np.atleast_2d(sigmas_x)
            (dim_x, dim_sigmas_x) = sigmas_x.shape
            assert dim_sigmas_x == W_c.shape[0], "Dimensions are wrong"
        try:
            (dim_y, dim_sigmas_y) = sigmas_y.shape
        except ValueError:  # sigmas_y is 1D
            sigmas_y = np.atleast_2d(sigmas_y)
            (dim_y, dim_sigmas_y) = sigmas_y.shape
            assert dim_sigmas_y == dim_sigmas_x, "Dimensions are wrong"
        # dim_x, dim_sigmas_x = sigmas_x.shape
        # dim_y, dim_sigmas_y = sigmas_y.shape
        # assert dim_sigmas_x == dim_sigmas_y, f"dim_sigmas_x != dim_sigmas_y: {dim_sigmas_x} != {dim_sigmas_y}"

        P_xy = np.zeros((dim_x, dim_y))
        # print(f"P_xy: {P_xy}")
        for i in range(dim_sigmas_x):
            P_xy += W_c[i]*((sigmas_x[:, i] - x_mean.flatten()).reshape(-1, 1)
                            @ (sigmas_y[:, i] - y_mean.flatten()).reshape(-1, 1).T)
            # print(P_xy)
            # print(f"i={i}")
        return P_xy
    
    def correlation_from_covariance(self, cov):
        """
        Calculate correlation matrix from a covariance matrix

        Parameters
        ----------
        cov : TYPE np.array((dim_p, dim_p))
            DESCRIPTION. Covariance matrix

        Returns
        -------
        corr : TYPE np.array((dim_p, dim_p))
            DESCRIPTION. Correlation matrix

        """
        sigmas = np.sqrt(np.diag(cov))
        dim_p = sigmas.shape[0]
        
        #Create sigma_mat = [[s1, s1 ,.., s1],
        # [s2, s2,...,s2],
        # [s_p, s_p,..,s_p]]
        sigma_mat = np.tile(sigmas.reshape(-1,1), dim_p)
        sigma_cross_mat = np.multiply(sigma_mat, sigma_mat.T)
        # print(f"sigmas: {sigmas}\n",
        #       f"sigma_mat: {sigma_mat}\n",
        #       f"sigma_cross_mat: {sigma_cross_mat}")
        corr = np.divide(cov, sigma_cross_mat) #element wise division
        return corr, sigmas
    
class UnscentedKalmanFilter(UKFBase):
    
    def __init__(self, fx, hx, points_x, Q, R, 
                 w_mean = None, v_mean = None,
                 sqrt_fn=scipy.linalg.sqrtm, name=None):
        """
        Create a Kalman filter. IMPORTANT: Additive white noise is assumed!

        """
        super().__init__(fx, hx, points_x, Q, R, 
                     w_mean = None, v_mean = None,
                     sqrt_fn=scipy.linalg.sqrtm, name=None)
        
        #as the noise is additive, dim_x = dim_w
        dim_x = self._dim_w
        dim_y = self._dim_v
    
        self.x_prior = np.zeros((dim_x,))
        self.P_prior = np.eye(dim_x)
        self.x_post = self.x_prior.copy()
        self.P_post = self.P_prior.copy()
        self._dim_x = dim_x
        self._dim_y = dim_y

        # create sigma-points
        self.sigmas_raw_fx = np.zeros((self._dim_x, self._num_sigmas_x))
        
        # sigma-points propagated through fx to form prior distribution
        self.sigmas_prop = np.zeros((self._dim_x, self._num_sigmas_x))
        
        # sigma-points based on prior distribution
        self.sigmas_raw_hx = np.zeros((self._dim_x, self._num_sigmas_x))
        
        # sigma-points propagated through measurement equation. Form posterior distribution
        self.sigmas_meas = np.zeros((self._dim_y, self._num_sigmas_x))

        self.y_res = np.zeros((dim_y, 1))           # residual
        self.y = np.array([[None]*dim_y]).T  # measurement
    

    def predict(self, UT=None, kwargs_sigma_points={}, fx=None, w_mean = None, Q = None, **fx_args):
        r"""
        Performs the predict step of the UKF. On return, self.x_prior and
        self.P_prior contain the predicted state (x) and covariance (P). '

        Important: this MUST be called before update() is called for the first
        time.
        
        
        Solves the equation
        wk = fx(x, p) - fx(x_post, E[p])
        fx(x,p) = fx(x_post, E[p]) + wk

        Parameters
        ----------

        fx : callable f(x, **fx_args), optional
            State transition function. If not provided, the default
            function passed in during construction will be used.

        UT : function(sigmas, Wm, Wc, kwargs_sigma_points), optional
            Optional function to compute the unscented transform for the sigma
            points passed. If the points are GenUT, you can pass 3rd and 4th moment through kwargs_sigma_points (see description of sigma points class for details)

    

        **fx_args : keyword arguments
            optional keyword arguments to be passed into f(x).
        """

        if fx is None:
            fx = self.fx
        
        if w_mean is None:
            w_mean = self.w_mean
        
        if Q is None:
            Q = self.Q
        elif np.isscalar(Q):
            Q = np.eye(self._dim_x) * Q
        

        if UT is None:
            UT = unscented_transform.unscented_transformation_gut

        # calculate sigma points for given mean and covariance for the states
        self.sigmas_raw_fx, self.Wm_x, self.Wc_x, P_sqrt = self.points_fn_x.compute_sigma_points(
            self.x_post, self.P_post, **kwargs_sigma_points)

        # propagate all the sigma points through fx
        self.sigmas_prop = self.compute_transformed_sigmas(
            self.sigmas_raw_fx, fx, **fx_args)

       

        # pass the propagated sigmas of the states (not the augmented states) through the unscented transform to compute prior
        self.x_prior, self.P_prior = UT(
            self.sigmas_prop, self.Wm_x, self.Wc_x)
        
        #add process noise
        self.x_prior += w_mean
        self.P_prior += Q

    def update(self, y, R=None, v_mean = None, UT=None, hx=None, kwargs_sigma_points={}, **hx_args):
        """
        Update the UKF with the given measurements. On return,
        self.x and self.P contain the new mean and covariance of the filter.

        Parameters
        ----------

        y : numpy.array of shape (dim_y)
            measurement vector

        R : numpy.array((dim_y, dim_y)), optional
            Measurement noise. If provided, overrides self.R for
            this function call.

        UT : function(sigmas, Wm, Wc, noise_cov), optional
            Optional function to compute the unscented transform for the sigma
            points passed through hx. 

        hx : callable h(x, **hx_args), optional
            Measurement function. If not provided, the default
            function passed in during construction will be used.

        **hx_args : keyword argument
            arguments to be passed into h(x) after x -> h(x, **hx_args)
        """

        if y is None:
            self.y = np.array([[None]*self._dim_y]).T
            self.x_post = self.x_prior.copy()
            self.P_post = self.P_prior.copy()
            return

        if hx is None:
            hx = self.hx

        if UT is None:
            UT = unscented_transform.unscented_transformation_gut

        if v_mean is None:
            v_mean = self.v_mean
            
        if R is None:
            R = self.R
        elif np.isscalar(R):
            R = np.eye(self._dim_y) * R

        # recreate sigma points
        (self.sigmas_raw_hx,
         self.Wm, self.Wc,
         P_sqrt) = self.points_fn_x.compute_sigma_points(self.x_prior,
                                                       self.P_prior,
                                                       **kwargs_sigma_points
                                                       )

        # send sigma points through measurement equation
        self.sigmas_meas = self.compute_transformed_sigmas(
            self.sigmas_raw_hx, hx, **hx_args)

        # compute mean and covariance of the predicted measurement
        y_pred, Py_pred = UT(self.sigmas_meas, self.Wm, self.Wc)
        
        # add measurement noise
        y_pred += v_mean
        Py_pred += R 
        self.y_pred = y_pred
        self.Py_pred = Py_pred.copy()

        # Innovation term of the UKF
        self.y_res = y - y_pred
        
        #Kalman gain. Start with cross_covariance
        Pxy = self.cross_covariance(
            self.x_prior, y_pred, self.sigmas_raw_hx, self.sigmas_meas, self.Wc)
        self.Pxy = Pxy

        # Kalman gain
        # solve K@Py_pred = P_xy <=> PY_pred.T @ K.T = P_xy.T
        self.K = np.linalg.solve(Py_pred.T, Pxy.T).T
        # self.K = np.linalg.lstsq(Py_pred.T, Pxy.T)[0].T #also an option
        assert self.K.shape == (self._dim_x, self._dim_y)

        # calculate posterior
        self.x_post = self.x_prior + self.K @ self.y_res
        self.P_post = self.P_prior - self.K @ Py_pred @ self.K.T



        
        
        