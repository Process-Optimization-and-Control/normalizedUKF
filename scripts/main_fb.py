# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 19:14:40 2021

Falling body estimation. Example 14.2 in Dan Simon's book "Optimal State Estimation"

@author: halvorak
"""


import numpy as np
import scipy.stats
import scipy.integrate
import matplotlib
# matplotlib.use("qtagg") #shold change backend? Get issues with editing axis on plots
import matplotlib.pyplot as plt
import pathlib
import os
import scipy.linalg
# import matplotlib.patches as plt_patches
# import matplotlib.transforms as plt_transforms
import copy

# Did some modification to these packages
from myFilter import UKF
# from myFilter import sigma_points as ukf_sp
# from myFilter import UKF_constrained

#Self-written modules
import sigma_points_classes as spc
import unscented_transformation as ut
import utils_falling_body as utils_fb


#%% For running the sim N times
N = 1 #this is how many times to repeat each iteration
dim_x = 3
j_valappil = np.zeros((dim_x, N))
j_valappil_norm = np.zeros((dim_x, N))
Ni = 0
rand_seed = 1234
while Ni < N:
    try:
        np.random.seed(rand_seed) #to get reproducible results. rand_seed updated in every iteration
        #%% Import the distributions of the parameters for the fx equations (states)
        #Modes of the dists are used for the true system, and mean of the dists are the parameters for the UKF
        
        # utils_fb.uncertainty_venturi2()
        x0, P0, par_true_fx, par_true_hx, Q_nom, R_nom = utils_fb.get_literature_values()
        par_kf_fx = par_true_fx.copy()
        par_true_hx["y_rep"] = R_nom
        par_kf_hx = par_true_hx.copy()
        
       
        
        #%% Define dimensions and initialize arrays
        
        # x0_kf = copy.deepcopy(x0) + np.sqrt(np.diag(P0)) #+ the standard deviation
        # x0_kf = copy.deepcopy(x0) + 2*np.sqrt(np.diag(P0)) #+ the standard deviation
        x0_kf = np.random.multivariate_normal(x0, P0) #random starting point
        # x0_kf = x0.copy()
        
        
        dim_x = x0.shape[0]
        dt_y = .5 # [s] <=> 50 ms. Measurement frequency
        dt_int = 1e-3 #[s] - discretization time for integration
        
        t_end = 30
        t = np.linspace(0, t_end, int(t_end/dt_y))
        dim_t = t.shape[0]
        # t = np.linspace(t_y[0],t_y[0+1], int(dt_y/dt), endpoint = True)
        
        y0 = utils_fb.hx(x0, par_true_hx)
        dim_y = y0.shape[0]
        y = np.zeros((dim_y, dim_t))
        y[:, 0] = y0
        
        x_true = np.zeros((dim_x, dim_t)) #[[] for _ in range(dim_t-1)] #make a list of list
        x_ol = np.zeros((dim_x, dim_t)) #Open loop simulation - same starting point and param as UKF
        x_post = np.zeros((dim_x, dim_t))
        x_post_norm = np.zeros((dim_x, dim_t))
        P_post = np.zeros((dim_x, dim_t))
        P_post_norm = np.zeros((dim_x, dim_t))
        
        x_true[:, 0] = x0
        x_ol[:, 0] = x0_kf.copy()
        x_post[:, 0] = x0_kf.copy()
        x_post_norm[:, 0] = x0_kf.copy()
        P_post[:, 0] = np.diag(P0)
        P_post_norm[:, 0] = np.diag(P0)
        
        t_span = (t[0],t[1])
        
        #%% Square-root method
        sqrt_fn = np.linalg.cholesky
        # sqrt_fn = scipy.linalg.cholesky
        # sqrt_fn = lambda P: scipy.linalg.cholesky(P, lower = True)
        
        # sqrt_fn = scipy.linalg.sqrtm #principal matrix square root
        
        args_ode_solver = dict(atol = 1e-13, rtol = 1e-10)
        
        #%% Define UKF with adaptive Q, R from UT
        points = spc.JulierSigmaPoints(dim_x,kappa = 3-dim_x, sqrt_method = sqrt_fn)
        # points = spc.ScaledSigmaPoints(dim_x,sqrt_method = sqrt_fn)
        fx_ukf = lambda x: utils_fb.fx_ukf_ode(utils_fb.ode_model_plant, 
                                                     t_span, 
                                                     x,
                                                     args_ode = (w_noise_kf,
                                                                 par_kf_fx),
                                                     args_solver = args_ode_solver)
        
        hx_ukf = lambda x_in: utils_fb.hx(x_in, par_kf_hx)#.reshape(-1,1)
        
        #kf is where Q adapts based on UT of parametric uncertainty
        kf = UKF.UnscentedKalmanFilter(fx = fx_ukf, hx = hx_ukf,
                                       points_x = points,
                                       Q = Q_nom, R = R_nom,
                                       sqrt_fn = sqrt_fn)
        kf.x_post = x_post[:, 0]
        kf.P_post = copy.deepcopy(P0)
        # kf.Q = Q_nom #to be updated in a loop
        # kf.R = R_nom #to be updated in a loop
        
        #%% Define UKF with adaptive Q, R from LHS/MC
        # points_norm = spc.JulierSigmaPoints(dim_x,
        #                                   kappa = 3-dim_x,
        #                                   sqrt_method = sqrt_fn)
        # # points_norm = spc.ScaledSigmaPoints(dim_x,sqrt_method = sqrt_fn)
        # fx_ukf_norm = lambda x, dt_kf: utils_fb.fx_ukf_ode(utils_fb.ode_model_plant, 
        #                                              t_span, 
        #                                              x,
        #                                              args_ode = (w_noise_kf,
        #                                                          par_kf_fx.copy()),
        #                                              args_solver = args_ode_solver)
        
        # hx_ukf_norm = lambda x_in: utils_fb.hx(x_in, par_kf_hx.copy())#.reshape(-1,1)
        
        # #kf is where Q adapts based on UT of parametric uncertainty
        # kf_norm = UKF.UnscentedKalmanFilter(fx = fx_ukf, hx = hx_ukf,
        #                                points_x = points_norm,
        #                                Q = Q_nom, R = R_nom,
        #                                sqrt_fn = sqrt_fn)
        # kf_norm.x_post = x_post_norm[:, 0]
        # kf_norm.P_post = copy.deepcopy(P0)
        # # kf_norm.Q = Q_nom #to be updated in a loop
        # # kf_norm.R = R_nom #to be updated in a loop
        
       
        #%% Create noise
        # w_plant = np.zeros((dim_t, dim_x))
        w_mean = np.zeros(dim_x)
        w_plant = np.random.multivariate_normal(w_mean, Q_nom, size = dim_t)
        w_noise_kf = np.zeros(dim_x)
        v_noise = np.random.multivariate_normal(kf.v_mean, kf.R, size = dim_t)
        
        #%% Simulate the plant and UKF
        for i in range(1,dim_t):
            t_span = (t[i-1], t[i])
            w_plant_i = w_plant[i, :]
            res = scipy.integrate.solve_ivp(utils_fb.ode_model_plant, 
                                            t_span,#(t_y[i-1],t_y[i]), 
                                            x_true[:, i-1],
                                            **args_ode_solver,
                                            # rtol = 1e-10,
                                            # atol = 1e-13,
                                            args = (w_plant_i, par_true_fx)
                                            )
            x_true[:, i] = res.y[:, -1] #add the interval to the full list
            #Make a new measurement and add measurement noise
            y[:, i] = utils_fb.hx(x_true[:, i], par_true_hx) + v_noise[i, :] 
            
            # Solve the open loop model prediction, based on the same info as UKF has (no measurement)
            res_ol = scipy.integrate.solve_ivp(utils_fb.ode_model_plant, 
                                               t_span,#(t_y[i-1],t_y[i]), 
                                               x_ol[:, i-1], 
                                               # rtol = 1e-10,
                                               # atol = 1e-13
                                               args = (w_noise_kf, par_kf_fx)
                                               )
            
            x_ol[:, i] = res_ol.y[:, -1] #add the interval to the full list
          
            # #Prediction step of each UKF
            kf.predict()
            # kf_norm.predict()
           
            #Correction step of UKF
            kf.update(y[:, i])
            # kf_norm.update(y[:, i])

            # # Save the estimates
            x_post[:, i] = kf.x_post
            # x_post_norm[:, i] = kf_norm.x_post
            P_post[:, i] = np.diag(kf.P_post)
            # P_post_norm[:, i] = np.diag(kf_norm.P_post)
        
        y[:, 0] = np.nan #the 1st measurement is not real, just for programming convenience
        
        
        
        #%% Compute performance index
        j_valappil[:, Ni] = utils_fb.compute_performance_index_valappil(x_post, 
                                                                      x_ol, 
                                                                      x_true)
        j_valappil_norm[:, Ni] = utils_fb.compute_performance_index_valappil(x_post_norm, 
                                                                          x_ol, 
                                                                          x_true)
    
        
        Ni += 1
        rand_seed += 1
        if (Ni%5 == 0): #print every 5th iteration                                                               
            print(f"End of iteration {Ni}/{N}")
    except BaseException as e:
        # print(e)
        raise e
        continue

#%% Plot
plot_it = True
if plot_it:
    # ylabels = [r"$x_1 [ft]$", r"$x_2 [ft/s]$", r"$x_3 [ft^3$/(lb-$s^2)]$", "$y [ft]$"]#
    # ylabels = [r"$x_1$ [ft]", r"$x_2$ [ft/s]", r"$x_3$ [*]", "$y$ [ft]"]#
    ylabels = [r"$x_1$ [m]", r"$x_2$ [m/s]", r"$x_3$ [*]", "$y$ [m]"]#
    kwargs_fill = dict(alpha = .2)    
    fig1, ax1 = plt.subplots(dim_x + 1, 1, sharex = True)
    for i in range(dim_x): #plot true states and ukf's estimates
        ax1[i].plot(t, x_true[i, :], label = "True")
        # ax1[i].plot([np.nan, np.nan], [np.nan, np.nan], color='w', alpha=0, label=' ')
        l_post = ax1[i].plot(t, x_post[i, :], label = r"$UKF$")[0]
        l_post_norm = ax1[i].plot(t, x_post_norm[i, :], label = r"$UKF_{norm}$")[0]
        ax1[i].plot(t, x_ol[i, :], label = "OL")
        
        
        if True:
            #Standard UKF
            ax1[i].fill_between(t, 
                                x_post[i, :] + 2*np.sqrt(P_post[i,:]),
                                x_post[i, :] - 2*np.sqrt(P_post[i,:]),
                                **kwargs_fill,
                                color = l_post.get_color())
            ax1[i].fill_between(t, 
                                x_post[i, :] + 1*np.sqrt(P_post[i,:]),
                                x_post[i, :] - 1*np.sqrt(P_post[i,:]),
                                **kwargs_fill,
                                color = l_post.get_color())
            
            #Normalized UKF
            ax1[i].fill_between(t, 
                                x_post_norm[i, :] + 2*np.sqrt(P_post_norm[i,:]),
                                x_post_norm[i, :] - 2*np.sqrt(P_post_norm[i,:]),
                                **kwargs_fill,
                                color = l_post_norm.get_color())
            ax1[i].fill_between(t, 
                                x_post_norm[i, :] + 1*np.sqrt(P_post_norm[i,:]),
                                x_post_norm[i, :] - 1*np.sqrt(P_post_norm[i,:]),
                                **kwargs_fill,
                                color = l_post_norm.get_color())
            
            
        
        ax1[i].set_ylabel(ylabels[i])
    ax1[-1].set_xlabel("Time [s]")
    #Plot measurements
    ax1[-1].plot(t, y[0,:], marker = "x", markersize = 3, linewidth = 0, label = "y")
    ax1[-1].set_ylabel(ylabels[-1])
    # ax1[0].legend()        
    ax1[0].legend(ncol = 3,
                  frameon = False)      
    plt.tight_layout()

# print("Median value of cost function is\n")
# for i in range(dim_x):
#     print(f"{ylabels[i]}: Q-UT = {np.median(j_valappil[i]): .3f}, Q-LHS-{N_lhs_dist} = {np.median(j_valappil_lhs[i]): .3f}, Q-MC-{N_mc_dist} = {np.median(j_valappil_mc[i]): .3f}, Q-MCm-{N_mcm_dist} = {np.median(j_valappil_mcm[i]): .3f} and Q-fixed = {np.median(j_valappil_qf[i]): .3f}")

