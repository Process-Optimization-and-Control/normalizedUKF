# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 11:51:40 2022

@author: halvorak
"""


import numpy as np
import scipy.stats
import scipy.integrate
import matplotlib.pyplot as plt
import scipy.linalg
import copy

# Did some modification to these packages
from myFilter import UKF
from myFilter import sigma_points as ukf_sp
from myFilter import unscented_transform as mf_ut
# from myFilter import UKF_constrained

#Self-written modules
import sigma_points_classes as spc
import unscented_transformation as ut
# import utils_falling_body as utils_fb


def fx(x):
    # xk = np.zeros(x.shape[0])
    # xk[0] = (x[0]/10)**3 - np.sqrt(x[1])
    # xk[1] = x[1]*.95
    B = np.eye(x.shape[0])*2
    xk = B@x
    return x
    # return xk

def parabola(x, a, b):
    """Return y = a + b*x**2."""
    # return a + b*x[0]**2+np.sqrt(x[1])
    # return x[0] + x[1]
    B = np.eye(x.shape[0])*2
    return B@x

x0 = np.array([2,10])
P0 = np.array([[1, .1],
               [.1, 2]])
# x0 = np.array([12.3, 7.6])
# P0 = np.array([[1.44, 0],
#                [0, 2.89]])

dim_x = x0.shape[0]
B = np.eye(dim_x)*2

#%% Monte Carlo
N_mc = int(1e5)
# N_mc = int(5e4)
x0_s = np.random.multivariate_normal(x0, P0, size = N_mc)

x1_s = np.array(list(map(fx, x0_s)))
x1_m = x1_s.mean(axis = 0)
P1 = np.cov(x1_s.T)

x1_s_prior = np.random.multivariate_normal(x1_m, P1, size = N_mc)

a=4
b=3
hx = lambda x: parabola(x, a, b)

y_samples = np.array(list(map(hx, x1_s_prior)))
y_mean = y_samples.mean(axis = 0)
y_var = y_samples.var()
y_var = np.cov(y_samples.T)
dim_y = y_mean.shape[0]

#%%Square-root method
sqrt_fn = np.linalg.cholesky
sqrt_fn = lambda P: scipy.linalg.cholesky(P, lower = True) #NB: ONLY LOWER Cholesky works!
sqrt_fn = lambda P: scipy.linalg.cholesky(P, lower = False) #WONT WORK 
# sqrt_fn = scipy.linalg.sqrtm
#%% UT - spc
Q = np.zeros((dim_x, dim_x))
R = np.zeros((dim_y, dim_y))
points_x = spc.JulierSigmaPoints(dim_x,
                                 kappa = 3-dim_x, 
                                 sqrt_method = sqrt_fn)
ukf = UKF.UKF_additive_noise(x0, P0, fx, hx, points_x, Q, R)
# ukf.x_post = x0
# ukf.P_post = P0
ukf.predict()
ut_x_prior = ukf.x_prior
ut_P_prior = ukf.P_prior
ukf.x_prior = x1_m #to make sure that we have excatly the same starting point when comparing MC and UT
ukf.P_prior = P1.copy()
ukf.update(y_mean)

points_x2 = ukf_sp.JulierSigmaPoints(dim_x,
                                 kappa = 3-dim_x, 
                                 sqrt_method = sqrt_fn)
sigmas = points_x2.sigma_points(x0, P0)
points_x2._compute_weights()
Wm2 = points_x2.Wm
Wc2 = points_x2.Wc
mean, cov = ut.unscented_transformation(sigmas.T, Wm2)
# ukf2 = UKF.UnscentedKalmanFilter(fx, hx, points_x2, Q, R)
# ukf2.x_post = x0
# ukf2.P_post = P0
# ukf2.predict()
# ut2_x_prior = ukf2.x_prior
# ut2_P_prior = ukf2.P_prior
# ukf2.x_prior = x1_m #to make sure that we have excatly the same starting point when comparing MC and UT
# ukf2.P_prior = P1.copy()
# ukf2.update(y_mean)

print(f"MC, x_prior_m: {x1_m}\n",
      f"UT, x_prior_m: {ut_x_prior}\n",
      # f"UT2, x_prior_m: {ut2_x_prior}\n",
      f"MC, P_prior_m: {P1}\n",
      f"UT, P_prior_m: {ut_P_prior}\n",
      # f"UT2, P_prior_m: {ut2_P_prior}\n",
      "---------------\n",
      f"MC, y_m: {y_mean}\n",
      f"UT, y_m: {ukf.y_pred}\n",
      f"MC, y_var: {y_var}\n",
      f"UT, P_y: {ukf.Py_pred}\n",
      )

points_scaled = spc.ScaledSigmaPoints(dim_x)

sigmas_scaled, Wm_s, Wc_s, P_sqrt_s = points_scaled.compute_sigma_points(x0, P0)

mean_s, cov_s = mf_ut.unscented_transformation_gut(sigmas_scaled, Wm_s, Wc_s)

points_scaled2 = ukf_sp.MerweScaledSigmaPoints(dim_x, alpha = points_scaled.alpha, beta = points_scaled.beta, kappa = points_scaled.kappa)
sigmas_s2 = points_scaled2.sigma_points(x0, P0)
Wm2, Wc2 = (points_scaled2.Wm, points_scaled2.Wc) 
mean_s2, cov_s2 = mf_ut.unscented_transformation_gut(sigmas_s2.T, Wm2, Wc2)

