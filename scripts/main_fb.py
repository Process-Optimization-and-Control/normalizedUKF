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
import pandas as pd
import seaborn as sns
# import matplotlib.patches as plt_patches
# import matplotlib.transforms as plt_transforms
import copy

# Did some modification to these packages
from myFilter import UKF
# from myFilter import sigma_points as ukf_sp

#Self-written modules
import sigma_points_classes as spc
import utils_falling_body as utils_fb

font = {'size': 18}
matplotlib.rc('font', **font)


#%% For running the sim N times
N = int(100) #this is how many times to repeat each iteration
dim_x = 3
cost_func = np.zeros((dim_x, N))
cost_func_norm = np.zeros((dim_x, N))
kappa_max = np.zeros((3, N)) #for P_post, P_prior and K
kappa_norm_max = np.zeros((3, N))

#save trajectories for condition number
kappa_trajectories = [[] for i in range(N)]
kappa_norm_trajectories = [[] for i in range(N)]

#save trajectories for std_dev and correlation
corr_trajectories = [[] for i in range(N)]
corr_prior_trajectories = [[] for i in range(N)]
corr_y_trajectories = [[] for i in range(N)]
corr_xy_trajectories = [[] for i in range(N)]
std_dev_x_post_trajectories = [[] for i in range(N)]
std_dev_x_prior_trajectories = [[] for i in range(N)]
std_dev_y_trajectories = [[] for i in range(N)]

Ni = 0
rand_seed = 6969
rand_seed_div = [7695, 7278] #list of simulations which diverge
# rand_seed = rand_seed_div[1]

run_ukf = True
run_ukf_norm = True
calc_RMSE = True
calc_condition_number = True
save_corr = True #save correlation matrix trajectory

diverged_sim = []
diverged_sim_norm = []
crashed_sim = []

#define colors for plots
color_std = '#1f77b4'
color_norm = '#ff7f0e'
while Ni < N:
    try:
        np.random.seed(rand_seed) #to get reproducible results. rand_seed updated in every iteration
        #%% Import the distributions of the parameters for the fx equations (states)
        x0, P0, par_true_fx, par_true_hx, Q_nom, R_nom = utils_fb.get_literature_values()
        par_kf_fx = par_true_fx.copy()
        par_kf_hx = par_true_hx.copy()
        
        x0_kf = np.random.multivariate_normal(x0, P0) #random starting point
        # print(f"{np.linalg.cond(P0):.2e}")
        #%% Define dimensions and initialize arrays
        
        dim_x = x0.shape[0]
        dt_y = .5 # [s] <=> 50 ms. Measurement frequency
        dt_int = 1e-3 #[s] - discretization time for integration
        
        t_end = 30 #[s]
        t = np.linspace(0, t_end, int(t_end/dt_y))
        dim_t = t.shape[0]
        
        y0 = utils_fb.hx(x0, par_true_hx)
        dim_y = y0.shape[0]
        y = np.zeros((dim_y, dim_t))
        y[:, 0] = y0
        
        x_true = np.zeros((dim_x, dim_t)) #[[] for _ in range(dim_t-1)] #make a list of list
        x_ol = np.zeros((dim_x, dim_t)) #Open loop simulation - same starting point and param as UKF
        x_post = np.zeros((dim_x, dim_t))
        x_post_norm = np.zeros((dim_x, dim_t))
        P_post = np.zeros((dim_x, dim_t))
        # P_post_norm = np.zeros((dim_x, dim_t))
        
        #condition numbers
        kappa = np.zeros((3, dim_t)) #P_post, P_prior and Py_pred
        kappa_norm = np.zeros((3,dim_t)) #corr_post, corr_prior and corr_y
        
        #correlation
        corr_post = np.zeros((dim_x, dim_x, dim_t))
        corr_prior = np.zeros((dim_x, dim_x, dim_t))
        corr_y = np.zeros((dim_y, dim_y, dim_t))
        corr_xy = np.zeros((dim_x, dim_y, dim_t))
        
        #standard deviation
        std_dev_x_post = np.zeros((dim_x, dim_t))
        std_dev_x_prior = np.zeros((dim_x, dim_t))
        std_dev_y = np.zeros((dim_y, dim_t))
        
        x_true[:, 0] = x0
        x_ol[:, 0] = x0_kf.copy()
        x_post[:, 0] = x0_kf.copy()
        x_post_norm[:, 0] = x0_kf.copy()
        P_post[:, 0] = np.diag(P0)
        std_dev_x_post[:, 0] = np.sqrt(np.diag(P0))
        std_dev_x_prior[:, 0] = np.sqrt(np.diag(P0))
        
        t_span = (t[0],t[1])
        
        #%% Square-root method
        # sqrt_fn = np.linalg.cholesky
        # sqrt_fn = scipy.linalg.cholesky
        sqrt_fn = lambda P: scipy.linalg.cholesky(P, lower = True)
        
        # sqrt_fn = scipy.linalg.sqrtm #principal matrix square root
        
        args_ode_solver = {}
        # args_ode_solver = dict(atol = 1e-13, rtol = 1e-10)
        
        #%% Def standard UKF
        # points = spc.JulierSigmaPoints(dim_x,kappa = 3-dim_x, sqrt_method = sqrt_fn)
        points = spc.ScaledSigmaPoints(dim_x,sqrt_method = sqrt_fn)
        fx_ukf = lambda x: utils_fb.fx_ukf_ode(utils_fb.ode_model_plant, 
                                                     t_span, 
                                                     x,
                                                     args_ode = (w_noise_kf,
                                                                 par_kf_fx),
                                                     args_solver = args_ode_solver)
        
        hx_ukf = lambda x_in: utils_fb.hx(x_in, par_kf_hx)#.reshape(-1,1)
        
        kf = UKF.UKF_additive_noise(x0 = x_post[:, 0], P0 = P0.copy(), 
                                    fx = fx_ukf, hx = hx_ukf, 
                                    points_x = points, Q = Q_nom, 
                                    R = R_nom)
        
        #%% Def normalized UKF
        # points_norm = spc.JulierSigmaPoints(dim_x,
        #                                   kappa = 3-dim_x,
        #                                   sqrt_method = sqrt_fn)
        
        # corr_post_lim = 0.95
        corr_post_lim = np.inf
        corr_prior_lim = copy.copy(corr_post_lim)
        corr_y_lim = np.inf#.97
        corr_xy_lim = np.inf
        # corr_xy_lim = copy.copy(corr_y_lim)
        
        points_norm = spc.ScaledSigmaPoints(dim_x,sqrt_method = sqrt_fn)
        
        #kf is where Q adapts based on UT of parametric uncertainty
        kf_norm = UKF.Normalized_UKF_additive_noise_corr_lim(x0 = x_post_norm[:, 0], P0 = P0, fx = fx_ukf, hx = hx_ukf,
                                        points_x = points_norm,
                                        Q = Q_nom, R = R_nom,
                                        corr_post_lim = corr_post_lim,
                                        corr_prior_lim = corr_prior_lim,
                                        corr_y_lim = corr_y_lim,
                                        corr_xy_lim = corr_xy_lim
                                        )
        
        #%% Create noise
        # w_plant = np.zeros((dim_t, dim_x))
        w_mean = np.zeros(dim_x)
        w_plant = np.random.multivariate_normal(w_mean, Q_nom, size = dim_t)
        w_noise_kf = np.zeros(dim_x)
        v_noise = np.random.multivariate_normal(kf.v_mean, kf.R, size = dim_t)
        
        # kappa[0, 0] = np.linalg.cond(kf.P_post)
        # kappa_norm[0, 0] = np.linalg.cond(kf_norm.corr_post)
        
        kappa[:, 0] = np.array([np.linalg.cond(kf.P_post), np.nan, np.nan]) #prior and measurement is not defined for time 0
        kappa_norm[:, 0] = np.array([np.linalg.cond(kf_norm.corr_post), np.nan, np.nan])
        std_dev_y[:, 0] = np.nan
        
        if save_corr:
            corr_post[:,:,0] = kf_norm.corr_post
            corr_prior[:,:,0] = kf_norm.corr_post #dummy
            corr_y[:,:,0] = np.eye(dim_y)
        
        eps = 1e-5 # lowest limit for x3
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
            if x_true[-1, i] <= eps:
               x_true[-1, i] = eps
               try:
                   if w_plant[i+1,-1] < 0:
                       w_plant[i+1,-1] = -w_plant[i+1,-1]
               except IndexError: #we are already at the last time step, don't need to do sth
                   pass    
                   
            #Make a new measurement and add measurement noise
            y[:, i] = utils_fb.hx(x_true[:, i], par_true_hx) + v_noise[i, :] 
            for j in range(dim_y):
                if y[j,i] < 0:
                    y[j,i] = eps**2
            # if y[0,i] < 0:
                
            
            if not calc_RMSE:
                # Solve the open loop model prediction, based on the same info as UKF has (no measurement)
                res_ol = scipy.integrate.solve_ivp(utils_fb.ode_model_plant, 
                                                   t_span,#(t_y[i-1],t_y[i]), 
                                                   x_ol[:, i-1], 
                                                   # rtol = 1e-10,
                                                   # atol = 1e-13
                                                   args = (w_noise_kf, par_kf_fx)
                                                   )
                
                x_ol[:, i] = res_ol.y[:, -1] #add the interval to the full list
          
            #Prediction and correction step of UKF. Calculate condition numbers
            if run_ukf:
                kf.predict()
                kf.update(y[:, i])
                if calc_condition_number:
                    kappa[0, i] = np.linalg.cond(kf.P_post)
                    kappa[1, i] = np.linalg.cond(kf.P_prior)
                    kappa[2, i] = np.linalg.cond(kf.Py_pred)
           
            #Prediction and correction step of normalized UKF. Calculate condition numbers
            if run_ukf_norm:
                kf_norm.predict()
                kf_norm.update(y[:, i])
                if calc_condition_number:
                    kappa_norm[0, i] = np.linalg.cond(kf_norm.corr_post)
                    kappa_norm[1, i] = np.linalg.cond(kf_norm.corr_prior)
                    kappa_norm[2, i] = np.linalg.cond(kf_norm.corr_y)
                if save_corr:
                    corr_post[:, :, i] = kf_norm.corr_post
                    corr_prior[:, :, i] = kf_norm.corr_prior
                    corr_y[:, :, i] = kf_norm.corr_y

            # Save the estimates
            x_post[:, i] = kf.x_post
            x_post_norm[:, i] = kf_norm.x_post
            P_post[:, i] = np.diag(kf.P_post)
            std_dev_x_post[:, i] = np.diag(kf_norm.std_dev_post)
            std_dev_x_prior[:, i] = np.diag(kf_norm.std_dev_prior)
            std_dev_y[:, i] = np.diag(kf_norm.std_dev_y)
            
        y[:, 0] = np.nan #the 1st measurement is not real, just for programming convenience
        
        #%% Compute performance index and condition numbers
        cost_func[:, Ni] = utils_fb.compute_performance_index_valappil(x_post, x_ol, x_true, RMSE = calc_RMSE)
        cost_func_norm[:, Ni] = utils_fb.compute_performance_index_valappil(x_post_norm, x_ol, x_true, RMSE = calc_RMSE)
        
        #condition numbers
        kappa_max[:, Ni] = np.max(kappa, axis = 1)
        kappa_norm_max[:, Ni] = np.max(kappa_norm, axis = 1)
        kappa_trajectories[Ni] = kappa
        kappa_norm_trajectories[Ni] = kappa_norm
        
        #standard deviations
        std_dev_x_post_trajectories[Ni] = std_dev_x_post
        std_dev_x_prior_trajectories[Ni] = std_dev_x_prior
        std_dev_y_trajectories[Ni] = std_dev_y
        
        if save_corr:
            corr_trajectories[Ni] = corr_post
            corr_prior_trajectories[Ni] = corr_prior
            corr_y_trajectories[Ni] = corr_y
            
            
        #check for divergence in the simulation
        x_post_max = x_post.max(axis = 1)
        x_post_norm_max = x_post_norm.max(axis = 1)
        # std_dev_post_max = std_dev_x_post.max(axis = 1)
        if (x_post_norm_max > [1e7, 1e3, 1e1]).any():
            diverged_sim_norm.append(rand_seed)
        if (x_post_max > [1e7, 1e3, 1e1]).any():
            diverged_sim.append(rand_seed)
    
        Ni += 1
        rand_seed += 1
        if (Ni%5 == 0): #print every 5th iteration                                                               
            print(f"End of iteration {Ni}/{N}")
    except BaseException as e:
        # print(e)
        
        #save some information
        kappa_max[:, Ni] = np.max(kappa, axis = 1)
        kappa_norm_max[:, Ni] = np.max(kappa_norm, axis = 1)
        kappa_trajectories[Ni] = kappa
        kappa_norm_trajectories[Ni] = kappa_norm
        corr_trajectories[Ni] = corr_post
        corr_prior_trajectories[Ni] = corr_prior
        corr_y_trajectories[Ni] = corr_y
        
        crashed_sim.append(rand_seed)
        rand_seed += 1
        # raise e
        continue

print(f"# crashed sim: {len(crashed_sim)}\n",
      f"# diverged sim, std: {len(diverged_sim)}\n",
      f"# diverged sim, norm: {len(diverged_sim_norm)}")


#%% Plot single trajectory
plot_it = False
plot_it = True
if plot_it:
    # ylabels = [r"$x_1 [ft]$", r"$x_2 [ft/s]$", r"$x_3 [ft^3$/(lb-$s^2)]$", "$y [ft]$"]#
    # ylabels = [r"$x_1$ [ft]", r"$x_2$ [ft/s]", r"$x_3$ [*]", "$y$ [ft]"]#
    ylabels = [r"$x_1$ [m]", r"$x_2$ [m/s]", r"$x_3$ [*]", "$y_1$ [m]", "$y_2$ [Pa]"]#
    kwargs_fill = dict(alpha = .2)    
    fig1, ax1 = plt.subplots(dim_x + 1 +1, 1, sharex = True, layout = "constrained")
    for i in range(dim_x): #plot true states and ukf's estimates
        ax1[i].plot(t, x_true[i, :], label = "True")
        # ax1[i].plot([np.nan, np.nan], [np.nan, np.nan], color='w', alpha=0, label=' ')
        if run_ukf:
            l_post = ax1[i].plot(t, x_post[i, :], label = r"UKF")[0]
        if run_ukf_norm:
            l_post_norm = ax1[i].plot(t, x_post_norm[i, :], label = "NUKF")[0]
            # l_post_norm = ax1[i].plot(t, x_post_norm[i, :], label = r"$UKF_{norm}$")[0]
        if not calc_RMSE:
            ax1[i].plot(t, x_ol[i, :], label = "OL")
        
        
        if True:
            #Standard UKF
            if run_ukf:
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
            if run_ukf_norm:
                ax1[i].fill_between(t, 
                                    x_post_norm[i, :] + 2*std_dev_x_post[i, :],
                                    x_post_norm[i, :] - 2*std_dev_x_post[i, :],
                                    **kwargs_fill,
                                    color = l_post_norm.get_color())
                ax1[i].fill_between(t, 
                                    x_post_norm[i, :] + 1*std_dev_x_post[i, :],
                                    x_post_norm[i, :] - 1*std_dev_x_post[i, :],
                                    **kwargs_fill,
                                    color = l_post_norm.get_color())
            
            
        
        ax1[i].set_ylabel(ylabels[i])
    ax1[-1].set_xlabel("Time [s]")
    #Plot measurements
    ax1[-2].plot(t, y[0,:], marker = "x", markersize = 3, linewidth = 0, label = ylabels[-2])
    ax1[-1].plot(t, y[1,:], marker = "x", markersize = 3, linewidth = 0, label = ylabels[-1])
    ax1[-2].set_ylabel(ylabels[-2])
    ax1[-1].set_ylabel(ylabels[-1])
    # ax1[0].legend()        
    ax1[0].legend(ncol = 2,
                  frameon = True)      
    
#%% std_dev_y
plt_kwargs = dict(linewidth = .7)
if plot_it:
    fig_sy, ax_sy = plt.subplots(dim_y, 1, sharex = True, layout = "constrained")
    
    for Ni in range(N):
        for i in range(dim_y):
            ax_sy[i].plot(t, std_dev_y_trajectories[Ni][i,:], color = color_std, **plt_kwargs)
    ax_sy[0].set_ylabel(r"$\sigma_{y_1} [m]$")
    ax_sy[1].set_ylabel(r"$\sigma_{y_2} [Pa]$")
    ax_sy[1].set_xlabel(r"$t$ [s]")
            
#%% std_dev_x_post/prior
if plot_it:
    fig_sx, ax_sx = plt.subplots(dim_x, 1, sharex = True, layout = "constrained")
    
    for Ni in range(N):
        for i in range(dim_x):
            ax_sx[i].plot(t, std_dev_x_post_trajectories[Ni][i,:], color = color_std, **plt_kwargs)
            ax_sx[i].plot(t, std_dev_x_prior_trajectories[Ni][i,:], color = color_norm, **plt_kwargs)
    
    ax_sx[0].set_ylabel(r"$\sigma_{x_1} [m]$")
    ax_sx[1].set_ylabel(r"$\sigma_{x_2} [m/s]$")
    ax_sx[2].set_ylabel(r"$\sigma_{x_3} [*]$")
    ax_sx[-1].set_xlabel(r"$t$ [s]")
    
    #add legend
    custom_lines = [matplotlib.lines.Line2D([0], [0], color=color_std, lw=3),
                    matplotlib.lines.Line2D([0], [0], color=color_norm, lw=3)
                    ]
    ax_sx[0].legend(custom_lines, [r"$\sigma^+$", r"$\sigma^-$"])

#%% Violin plot of cost function and condition numbers for selected matrices
if N >= 5: #only plot this if we have done some iterations
    cols_x = [r"$x_1$", r"$x_2$", r"$x_3$"]
    cols_x = ["x1", "x2", "x3"]
    
    cost_diff = cost_func_norm - cost_func
    df_j_diff = pd.DataFrame(columns = [cols_x], data = cost_diff.T)
    
    df_cost = pd.DataFrame(columns = [cols_x], data = cost_func_norm.T)
    # df_cost["Filter"] = "sigmaRho"
    
    df_cost2 = pd.DataFrame(columns = [cols_x], data = cost_func.T)
    df_cost2["Filter"] = "Standard"
    
    fig_j, ax_j = plt.subplots(3,1)
    y_label = [r"$x_1$", r"$x_2$", r"$x_3$"]
    for i in range(dim_x):
        ax_j[i].scatter(range(N), cost_diff[i,:])
        ax_j[i].set_ylabel(y_label[i])
        x_lims = ax_j[i].get_xlim()
        ax_j[i].plot(x_lims, [0,0])
        ax_j[i].set_xlim(x_lims)
    
    # df_cost = pd.concat([df_cost, df_cost2])
    # del df_cost2
    
    # ax_cost = sns.violinplot(df_j_diff)
    # ax_cost.set_ylabel(r"$RMSE_{norm}-RMSE_{UKF}$")
    
    cols_kappa = [r"$(P^+,\rho^+)$", r"$(P^-,\rho^-)$", r"$(P_y, \rho_y)$"]
    # cols_kappa = ["(P^+,\rho^+)", "(P_y, \rho_y)"]
    ylabel_kappa = r"$\kappa_{max}$"
    df_kappa = pd.DataFrame(columns = [ylabel_kappa], data = kappa_max[0,:].T)
    df_kappa["Matrix"] = cols_kappa[0]
    df_kappa["Method"] = "UKF"
    
    df_kappa2 = pd.DataFrame(columns = [ylabel_kappa], data = kappa_max[1,:].T)
    df_kappa2["Matrix"] = cols_kappa[1]
    df_kappa2["Method"] = "UKF"
    
    df_kappa3 = pd.DataFrame(columns = [ylabel_kappa], data = kappa_max[2,:].T)
    df_kappa3["Matrix"] = cols_kappa[2]
    df_kappa3["Method"] = "UKF"
    
    sr_ukf_name = r"$\sigma\rho-UKF$"
    
    df_kappa_norm = pd.DataFrame(columns = [ylabel_kappa], data = kappa_norm_max[0,:].T)
    df_kappa_norm["Matrix"] = cols_kappa[0]
    df_kappa_norm["Method"] = sr_ukf_name
    
    df_kappa_norm2 = pd.DataFrame(columns = [ylabel_kappa], data = kappa_norm_max[1,:].T)
    df_kappa_norm2["Matrix"] = cols_kappa[1]
    df_kappa_norm2["Method"] = sr_ukf_name
    
    df_kappa_norm3 = pd.DataFrame(columns = [ylabel_kappa], data = kappa_norm_max[2,:].T)
    df_kappa_norm3["Matrix"] = cols_kappa[2]
    df_kappa_norm3["Method"] = sr_ukf_name
    
    df_kappa = pd.concat([df_kappa, df_kappa2, df_kappa3, df_kappa_norm, df_kappa_norm2, df_kappa_norm3], ignore_index = True)
    
    del df_kappa2, df_kappa3, df_kappa_norm, df_kappa_norm2, df_kappa_norm3
    
    fig_kappa_hist, ax_kappa_hist = plt.subplots(1,1, layout = "constrained")
    ax_kappa_hist = sns.stripplot(data = df_kappa, x = "Matrix", y = ylabel_kappa, ax = ax_kappa_hist, hue = "Method")
    ax_kappa_hist.set_yscale("log")
    
    # fig_kappa_hist2, ax_kappa_hist2 = plt.subplots(1,1)
    # df_kappa2 = df_kappa.copy()
    # df_kappa2[ylabel_kappa] = np.log(df_kappa2[ylabel_kappa].values)
    # df_kappa2 = df_kappa2.rename(columns = {ylabel_kappa: r"log($\kappa_{max}$)"})
    # ax_kappa_hist2 = sns.stripplot(data = df_kappa2, x = "Matrix", y = r"log($\kappa_{max}$)", ax = ax_kappa_hist2, hue = "Method")
    # plt.tight_layout()
    
#%% condition number trajectories - Monte Carlo
fig_kt, ax_kt = plt.subplots(3, 1, sharex = True, layout = "constrained")
ylabels = [r"$\kappa(P^+,\rho^+)$", r"$\kappa(P^-,\rho^-)$", r"$\kappa(P_y, \rho_y)$"]
# plt_kwargs = dict()

for i in range(dim_x):
    for Ni in range(N):
        if ((kappa_trajectories[Ni] < 1e-1).any() and run_ukf):
            print(Ni)
        # if Ni == 1: #plot with label
        #     ax_kt[i].plot(t, kappa_trajectories[Ni][i,:], label = "UKF", color = color_std, **plt_kwargs)
        #     ax_kt[i].plot(t, kappa_norm_trajectories[Ni][i,:], label = r"$\sigma\rho-UKF$", color = color_norm, **plt_kwargs)
        # else: #without label
        ax_kt[i].plot(t, kappa_trajectories[Ni][i,:], color = color_std, **plt_kwargs)
        ax_kt[i].plot(t, kappa_norm_trajectories[Ni][i,:], color = color_norm, **plt_kwargs)
            
    ax_kt[i].set_ylabel(ylabels[i])
    ax_kt[i].set_yscale("log")
ax_kt[-1].set_xlabel("Time [s]")

#custom legend
# from matplotlib.lines import Line2D
custom_lines = [matplotlib.lines.Line2D([0], [0], color=color_std, lw=3),
                matplotlib.lines.Line2D([0], [0], color=color_norm, lw=3)
                ]
ax_kt[0].legend(custom_lines, ["UKF", "NUKF"])

#compute mean values
kappa_np = np.hstack(kappa_trajectories)
kappa_np_mean = np.nanmean(kappa_np, axis = 1)
kappa_norm_np = np.hstack(kappa_norm_trajectories)
kappa_norm_np_mean = np.nanmean(kappa_norm_np, axis = 1)
   
#%% corr plot
if save_corr:
    fig_corr_post, ax_corr_post = plt.subplots(dim_x, dim_x, sharex = True, sharey = True, layout = "constrained")
    
    #remove redundant axes
    for r in range(dim_x):
        for c in range(r, dim_x):
            if r == c:
                pass
            else:
                ax_corr_post[r,c].remove()
    
    plt_prior_post_same = True
    if plt_prior_post_same:
        pass
    
    for Ni in range(N):
        for r in range(dim_x):
            for c in range(r):
                ax_corr_post[r, c].plot(t, corr_trajectories[Ni][r, c, :], color = color_norm, **plt_kwargs)
                ax_corr_post[r, c].plot(t, corr_prior_trajectories[Ni][r, c, :], color = color_std, **plt_kwargs)
                
                #plot some limits
                xlims = ax_corr_post[r,c].get_xlim()
                ax_corr_post[r, c].plot(xlims, [0,0], "k", linewidth = .5) #zero
                #upper and lower correlation limits
                ax_corr_post[r, c].plot(xlims, [corr_post_lim,corr_post_lim], "r", linewidth = .5)
                ax_corr_post[r, c].plot(xlims, [-corr_post_lim,-corr_post_lim], "r", linewidth = .5)
                ax_corr_post[r,c].set_xlim(xlims)
    
    #set limits on scales etc
    for r in range(dim_x):
        ax_corr_post[r,c].set_ylim([-1,1])
        for c in range(r):
            ax_corr_post[r,c].set_ylim([-1,1])
            
    for c in range(dim_x):
        ax_corr_post[-1,c].set_xlabel("Time [s]")
    # fig_corr_post.suptitle(r"$\rho^+$-trajectories, $N_{MC}$ = " + f"{N}")
    # fig_corr_post.suptitle(r"$\rho$-trajectories, $N_{MC}$ = " + f"{N}")
    
    #set \rho[r,c] as a textbox
    from matplotlib.offsetbox import AnchoredText
    for r in range(dim_x):
        for c in range(dim_x):
            text_box = AnchoredText(r"$\rho$" + f"[{r+1},{c+1}]", frameon=True, loc="upper left", pad=0.5)
            plt.setp(text_box.patch, facecolor='white', alpha=0.5)
            ax_corr_post[r,c].add_artist(text_box)
            
            
    fig_corr_post.suptitle(r"correlation-trajectories, $N_{MC}$ = " + f"{N}")
    
    #add legend
    custom_lines = [matplotlib.lines.Line2D([0], [0], color=color_norm, lw=3),
                    matplotlib.lines.Line2D([0], [0], color=color_std, lw=3),
                    matplotlib.lines.Line2D([0], [0], color='r', lw=3)
                    ]
    ax_corr_post[0,0].legend(custom_lines, [r"$\rho^+$", r"$\rho^-$", r"$\rho_{lim}=\pm$ " + f"{corr_post_lim}" ])
    
    
                
#%% corr_y plot
if save_corr:
    fig_corr_y, ax_corr_y = plt.subplots(dim_y, dim_y, sharex = True, sharey = True, layout = "constrained")
    
    #remove redundant axes
    for r in range(dim_y):
        for c in range(r, dim_y):
            if r == c:
                pass
            else:
                ax_corr_y[r,c].remove()
                
    for Ni in range(N):
        for r in range(dim_y):
            for c in range(r):
                ax_corr_y[r, c].plot(t, corr_y_trajectories[Ni][r, c, :], color = color_norm, **plt_kwargs)
                
                #plot some limits
                xlims = ax_corr_y[r,c].get_xlim()
                ax_corr_y[r, c].plot(xlims, [0,0], "k", linewidth = .5) #zero
                #upper and lower correlation limits
                ax_corr_y[r, c].plot(xlims, [corr_y_lim,corr_y_lim], "r", linewidth = .5)
                ax_corr_y[r, c].plot(xlims, [-corr_y_lim,-corr_y_lim], "r", linewidth = .5)
                ax_corr_y[r,c].set_xlim(xlims)
    
    #set limits on scales etc
    for r in range(dim_y):
        ax_corr_y[r,0].set_ylim([-1,1])
        # for c in range(r):
        #     ax_corr_y[r,c].set_ylim([-1,1])
        
    fig_corr_y.suptitle(r"$\rho_y$-trajectories, $N_{MC}$ = " + f"{N}")
            
            
