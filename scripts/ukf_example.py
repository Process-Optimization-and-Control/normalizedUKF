# -*- coding: utf-8 -*-
"""
Created on Fri May  7 13:23:40 2021

@author: halvorak
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
# sys.path.append("\filterpy\filterpy\kalman")
# from filterpy.filterpy.kalman.UKF import UnscentedKalmanFilter
# from filterpy.filterpy.kalman.sigma_points import JulierSigmaPoints
from myFilter import UKF
# from myFilter import sigma_points
import sigma_points_classes as sigma_points

# Simple example of a linear order 1 kinematic filter in 2D. There is no
# need to use a UKF for this example, but it is easy to read.

# >>> def fx(x, dt):
# >>>     # state transition function - predict next state based
# >>>     # on constant velocity model x = vt + x_0
# >>>     F = np.array([[1, dt, 0, 0],
# >>>                   [0, 1, 0, 0],
# >>>                   [0, 0, 1, dt],
# >>>                   [0, 0, 0, 1]], dtype=float)
# >>>     return np.dot(F, x)
# >>>
# >>> def hx(x):
# >>>    # measurement function - convert state into a measurement
# >>>    # where measurements are [x_pos, y_pos]
# >>>    return np.array([x[0], x[2]])
# >>>
# >>> dt = 0.1
# >>> # create sigma points to use in the filter. This is standard for Gaussian processes
# >>> points = MerweScaledSigmaPoints(4, alpha=.1, beta=2., kappa=-1)
# >>>
# >>> kf = UnscentedKalmanFilter(dim_x=4, dim_z=2, dt=dt, fx=fx, hx=hx, points=points)
# >>> kf.x = np.array([-1., 1., -1., 1]) # initial state
# >>> kf.P *= 0.2 # initial uncertainty
# >>> z_std = 0.1
# >>> kf.R = np.diag([z_std**2, z_std**2]) # 1 standard
# >>> kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=0.01**2, block_size=2)
# >>>
# >>> zs = [[i+randn()*z_std, i+randn()*z_std] for i in range(50)] # measurements
# >>> for z in zs:
# >>>     kf.predict()
# >>>     kf.update(z)
# >>>     print(kf.x, 'log-likelihood', kf.log_likelihood)

def fx(x, dt):
    """
    State transition function, to calculate x_k+1 = f(x_k, dt)

    Parameters
    ----------
    x : TYPE np.array
        DESCRIPTION. State vector
    dt : TYPE float
        DESCRIPTION. Time step

    Returns
    -------
    None.

    """
    nx = x.shape[0]
    F = np.identity(nx)
    F[0, 2] = dt
    F[1, nx-1] = dt
    x_kp1 = np.dot(F, x)
    return x_kp1

def hx(x, param):
    N1 = param["N1"]
    E1 = param["E1"]
    N2 = param["N2"]
    E2 = param["E2"]
    
    n = x[0]
    e = x[1]
    # ndot = x[2]
    # edot = x[3]
    
    y = np.zeros((2,))
    y[0] = np.sqrt((n - N1)**2 + (e - E1)**2)
    y[1] = np.sqrt((n - N2)**2 + (e - E2)**2)
    
    return y 

x0 = np.array([0, 0, 50, 50])
# x0 = np.expand_dims(x0, axis = 1)


dt = .01 # [s]
tspan = 60 #s
param = {"N1": 20,
         "E1": 0,
         "N2": 0,
         "E2": 20
         }
# y0 = np.zeros((2, 1))
y0 = hx(x0, param)

dim_x = x0.shape[0]
dim_y = y0.shape[0]
t = np.linspace(0, tspan, int(tspan/dt))
# t2 = np.arange(0, 60+.9*dt, dt)

points = sigma_points.JulierSigmaPoints(dim_x, kappa = dim_x-3)

f = lambda x: fx(x, dt)
h = lambda x: hx(x, param)

R = np.diag([1., 1.])
Q = np.diag([1e-5, 1e-5, 4., 4.])
kf = UKF.UnscentedKalmanFilter(f, h, points, Q, R)
kf.x_post = x0.flatten()
kf.P_post = np.eye(dim_x)*1e-3

#The true system (real plant)
x_true = np.zeros((x0.shape[0], len(t)))
x_true[:, 0] = x0
y_true = np.zeros((y0.shape[0], len(t)))
y_true[:, 0] = y0

#The predictions by UKF
x_pred = np.zeros((x0.shape[0], len(t)))
x_pred[:, 0] = x0
# y_meas = np.zeros((y0.shape[0], len(t)))
# y_meas[:, 0] = y0
likelihood = np.zeros(t.shape)
P_tensor = np.zeros((len(t), dim_x, dim_x))
P_tensor[0] = kf.P_post
P_trace = np.zeros(t.shape)
K_kalman = np.zeros((len(t), dim_x, dim_y))

P_trace[0] = np.trace(kf.P_post)
# kf.predict()
# kf.update(y0)


for i in range(1, len(t)):
    #simulate the real plant
    x_true[:, i] = fx(x_true[:, i-1], dt)
    y_true[:, i] = hx(x_true[:, i], param) + np.random.multivariate_normal(mean = np.array([0, 0]), cov = kf.R)
    
    #kalman filter estimate
    kf.predict()
    kf.update(y_true[:, i])
    
    #save the estimates
    x_pred[:, i] = kf.x_post
    # likelihood[i] = kf.log_likelihood
    P_tensor[i] = kf.P_post
    P_trace[i] = np.trace(kf.P_post)
    K_kalman[i] = kf.K
    

x_error = x_true - x_pred
    
(fig , [[ax1, ax2], [ax3, ax4]])= plt.subplots(2,2)
ax = ax1
line_x0 = ax1.plot(t, x_error[0, :])
line_x1 = ax2.plot(t, x_error[1, :])
line_x2 = ax3.plot(t, x_error[2, :])
line_x3 = ax4.plot(t, x_error[3, :])
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("North error")
ax2.set_ylabel("East error")
ax3.set_ylabel("North velocity error")
ax4.set_ylabel("East velocityerror")

fig2 = plt.figure()
ax = plt.gca()
ax.plot(t, likelihood)
ax.set_ylabel("Likelihood")
ax.set_xlabel("Time")

fig3 = plt.figure()
ax = plt.gca()
ax.plot(t, P_trace)
ax.set_ylabel("Tr(P)")
ax.set_xlabel("Time")

fig = plt.figure()
plt.plot(t, K_kalman[:, 0, 0])
plt.ylabel("K [0,0]")    
    