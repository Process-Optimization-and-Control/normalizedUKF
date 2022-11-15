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
import pandas as pd
# import matplotlib.patches as plt_patches
# import matplotlib.transforms as plt_transforms
import copy

# Did some modification to these packages
from myFilter import UKF
# from myFilter import sigma_points as ukf_sp

#Self-written modules
import sigma_points_classes as spc
import unscented_transformation as ut
import utils_wp

#%% Define directories
dir_project = pathlib.Path(__file__).parent.parent 
dir_data = os.path.join(dir_project, "data")


#%% For running the sim N times
N = int(10) #this is how many times to repeat each iteration (Monte Carlo simulation)
dim_x = 5 #for pre-allocation matrices where solutions will be stored
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

#%% Decide which UKF to run
run_ukf = True
run_ukf_norm = True
calc_RMSE = True
calc_condition_number = True
save_corr = True #save correlation matrix trajectory
noise_case = ["additive", "random_par_init", "random_par"]
noise_used = noise_case[0]

diverged_sim = []
diverged_sim_norm = []
crashed_sim = []

#create paths

#import reference points about the system
df_reservoir = pd.read_csv(os.path.join(dir_data, "reservoir_data.csv"), delimiter = ";", decimal = ",")
df_pump_chart = pd.read_csv(os.path.join(dir_data, "pump_data.csv"), delimiter = ";", decimal = ",")

std_dev_repeatability, accuracy, drift_rate = utils_wp.get_sensor_data()


#define colors for plots
colors_dict = matplotlib.colors.TABLEAU_COLORS
color_true = list(colors_dict.values())[0]
color_std = list(colors_dict.values())[1]
color_norm = list(colors_dict.values())[2]
color_ol = list(colors_dict.values())[3]
while Ni < N:
    try:
        np.random.seed(rand_seed) #to get reproducible results. rand_seed updated in every iteration
        #%% Import the distributions of the parameters for the fx equations (states)
        x0, P0, par_mean_fx, par_true_hx, P_par_fx, Q_nom, R_nom = utils_wp.get_literature_values(df_reservoir, df_pump_chart)
        
        par_mean_fx["theta_p1_0"] += 50
        par_mean_fx["theta_p3_0"] += 50
        par_mean_fx["theta_rho_0"] += 130
        x0[0] = par_mean_fx["theta_p1_0"]
        x0[1] += 55
        x0[2] = par_mean_fx["theta_p3_0"]
        x0[-1] = par_mean_fx["theta_rho_0"]
        
        par_kf_fx = par_mean_fx.copy()
        par_kf_hx = par_true_hx.copy()
        
        if noise_used == noise_case[0]:
            par_true_fx = par_mean_fx.copy()
        else: #random parameter for the system
            par_true_fx_val = np.random.multivariate_normal(list(par_mean_fx.values()), P_par_fx)
            par_true_fx = {key:val for key, val in zip(par_mean_fx.keys(), par_true_fx_val)}
        
        x0_kf = x0.copy()
        x0_true = np.random.multivariate_normal(x0, P0) #random starting point
        del x0
        # print(f"{np.linalg.cond(P0):.2e}")
        
        #%% Casadi functions
        F,jac_p,_,_,_,_,x_var,z_var,u_var,p_var,_,_,_, res, p_aug_var, Qk_lin = utils_wp.ode_model_plant(P_par_fx)
        hx = utils_wp.hx_cd()
        # x0_t = np.array([ 50, 70, 60, 100, 600])
        # u0_t = np.array([3000, .65])
        # # z0_t = np.array([70, 100])
        # t_span = (0,1)
        # xk = utils_wp.integrate_dae(F, x0_t, u0_t, par_kf_fx, t_span)
        
        
        #%% Define dimensions and initialize arrays
        
        dim_x = x0_true.shape[0]
        dt_y = .1 # [s,min,h] <=> measurement frequency 
        # dt_int = 1e-3 #[s,min,h] - discretization time
        
        t_end = 10 #[y]
        t = np.linspace(0, t_end, int(t_end/dt_y))
        dim_t = t.shape[0]
        
        y0 = utils_wp.eval_hx(hx, x0_true, par_true_hx)
        dim_y = y0.shape[0]
        y = np.zeros((dim_y, dim_t))
        y[:, 0] = y0
        
        u0 = utils_wp.get_u0()
        dim_u = u0.shape[0]
        u = np.zeros((dim_u, dim_t))
        u[:,0] = u0
        u[0, :] = u0[0]
        
        #compute initial state (dae equation ==> need correct algebraic solution, which is determined by u0 and parameters)
        x0_true = utils_wp.integrate_dae(F, x0_true, u0, par_true_fx, (0,1e-6))
        x0_kf = utils_wp.integrate_dae(F, x0_kf, u0, par_kf_fx, (0,1e-6))
        
        #initialize arrays
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
        
        x_true[:, 0] = x0_true
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
        
        
        #%% Def standard UKF
        # points = spc.JulierSigmaPoints(dim_x,kappa = 3-dim_x, sqrt_method = sqrt_fn)
        points = spc.ScaledSigmaPoints(dim_x,sqrt_method = sqrt_fn)
        
        fx_ukf = None #must be updated "on the fly" due to different uk, tk
        hx_ukf = lambda x: utils_wp.eval_hx(hx, x, par_kf_hx)
        
        if noise_used == "additive":
            Q_kf = Q_nom.copy()
            mult_w_real = 1. #multiplied with w_realization
        else:
            Q_kf = None
            mult_w_real = 0 #
            
        
        kf = UKF.UKF_additive_noise(x0 = x_post[:, 0], P0 = P0.copy(), 
                                    fx = fx_ukf, hx = hx_ukf, 
                                    points_x = points, Q = Q_kf, 
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
                                        Q = Q_kf, R = R_nom,
                                        corr_post_lim = corr_post_lim,
                                        corr_prior_lim = corr_prior_lim,
                                        corr_y_lim = corr_y_lim,
                                        corr_xy_lim = corr_xy_lim
                                        )
        
        #%% Create noise
        # w_plant = np.zeros((dim_t, dim_x))
        w_mean = np.zeros(dim_x)
        w_plant = np.random.multivariate_normal(w_mean, Q_nom, size = dim_t).T
        w_noise_kf = np.zeros(dim_x)
        v_noise = np.random.multivariate_normal(kf.v_mean, kf.R, size = dim_t).T
        
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
        
        #time zero
        x_true[:, 0] = utils_wp.integrate_dae(F, x_true[:, 0], u[:, 0], par_true_fx, (t[0], t[1])) + w_plant[:, 0]*mult_w_real
        y[:, 0] = utils_wp.eval_hx(hx, x_true[:, 0], par_true_hx) + v_noise[:, 0]
        x_ol[:, 0] = utils_wp.integrate_dae(F, x_ol[:, 0], u[:, 0], par_kf_fx, (t[0], t[1]))
        
        #%% Simulate the plant and UKF
        for i in range(1,dim_t):
            t_span = (t[i-1], t[i])
            if noise_used == noise_case[2]: #random parameters at every time step
                #draw a new set of parameters
                par_true_fx_val = np.random.multivariate_normal(list(par_mean_fx.values()), P_par_fx)
                par_true_fx = {key:val for key, val in zip(par_mean_fx.keys(), par_true_fx_val)}
            
            u[-1, i] = utils_wp.semi_random_number(u[-1, i-1], chance_of_moving = .98, u_lb = .5, u_hb = .80)
            
            x_true[:, i] = utils_wp.integrate_dae(F, x_true[:, i-1], u[:, i], par_true_fx, t_span) + w_plant[:, i]*mult_w_real
                   
            #Make a new measurement and add measurement noise
            y[:, i] = utils_wp.eval_hx(hx, x_true[:, i], par_true_hx) + v_noise[:, i]
                
            
            if not calc_RMSE:
                # Solve the open loop model prediction, based on the same info as UKF has (no measurement)
                x_ol[:, i] = utils_wp.integrate_dae(F, x_ol[:, i-1], u[:, i], par_kf_fx, t_span) #add the interval to the full list
            else: #calculate it either way
                # pass
                x_ol[:, i] = utils_wp.integrate_dae(F, x_ol[:, i-1], u[:, i], par_kf_fx, t_span) #add the interval to the full list
                
            fx_ukf = lambda x: utils_wp.integrate_dae(F, x, u[:, i], par_kf_fx, t_span)
            
          
            #Prediction and correction step of UKF. Calculate condition numbers
            if run_ukf:
                if noise_used == noise_case[0]: #additive noise
                    Q_kf = Q_nom
                else:
                    raise ValueError("Not implemented yet")
                kf.predict(fx = fx_ukf, Q = Q_kf)
                kf.update(y[:, i])
                if calc_condition_number:
                    kappa[0, i] = np.linalg.cond(kf.P_post)
                    kappa[1, i] = np.linalg.cond(kf.P_prior)
                    kappa[2, i] = np.linalg.cond(kf.Py_pred)
           
            #Prediction and correction step of normalized UKF. Calculate condition numbers
            if run_ukf_norm:
                kf_norm.predict(fx = fx_ukf)
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
        cost_func[:, Ni] = utils_wp.compute_performance_index_valappil(x_post, x_ol, x_true, RMSE = calc_RMSE)
        cost_func_norm[:, Ni] = utils_wp.compute_performance_index_valappil(x_post_norm, x_ol, x_true, RMSE = calc_RMSE)
        
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
        if (x_post_norm_max > [1e7, 1e3, 1e3, 1e3, 1e3]).any():
            diverged_sim_norm.append(rand_seed)
        if (x_post_max > [1e7, 1e3, 1e3, 1e3, 1e3]).any():
            diverged_sim.append(rand_seed)
    
        Ni += 1
        rand_seed += 1
        if (Ni%5 == 0): #print every 5th iteration                                                               
            print(f"End of iteration {Ni}/{N}")
    except BaseException as e:
        # print(e)
        
        raise e
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
    ylabels = [r"$p_1$ [bar]", r"$p_2$ [bar]", r"$p_3$ [bar]", r"$Q [m^3/h]$", r"$\rho [kg/m^3]$"]#
    kwargs_fill = dict(alpha = .2)    
    fig1, ax1 = plt.subplots(dim_x + 1, 1, sharex = True, layout = "constrained")
    for i in range(dim_x): #plot true states and ukf's estimates
        ax1[i].plot(t, x_true[i, :], color = color_true, label = "True")
        
        if i <=2:
            ax1[i].plot(t, y[i,:], marker = "x", markersize = 3, linewidth = 0, color = color_true, label = "y")
        elif i == 3: #Q aka dp_venturi measurement
            ax1[-1].plot(t, y[i,:], marker = "x", markersize = 3, linewidth = 0, color = color_true, label = "y")
            # ax1[-1].set_ylabel(r"$dP_{venturi}$ [mbar]")
            ax1[-1].set_ylabel(r"$dP_{Q}$ [mbar]")
            
            
        
        # ax1[i].plot([np.nan, np.nan], [np.nan, np.nan], color='w', alpha=0, label=' ')
        if run_ukf:
            l_post = ax1[i].plot(t, x_post[i, :], color = color_std, label = r"UKF")[0]
        if run_ukf_norm:
            l_post_norm = ax1[i].plot(t, x_post_norm[i, :], color = color_norm, label = "NUKF")[0]
            # l_post_norm = ax1[i].plot(t, x_post_norm[i, :], label = r"$UKF_{norm}$")[0]
        if not calc_RMSE:
            ax1[i].plot(t, x_ol[i, :], color = color_ol, label = "OL")
        else: #plot it either way
            # pass
            ax1[i].plot(t, x_ol[i, :], color = color_ol, label = "OL")
        
        
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
    ax1[-1].set_xlabel("Time [years]")
    # #Plot measurements
    # ax1[-1].plot(t, y[1,:], marker = "x", markersize = 3, linewidth = 0, label = ylabels[-1])
    # ax1[-2].set_ylabel(ylabels[-2])
    # ax1[-1].set_ylabel(ylabels[-1])
    # ax1[0].legend()        
    ax1[0].legend(ncol = 3,
                  frameon = True)      
    
    
    fig_u, ax_u = plt.subplots(dim_u, sharex = True, layout = "constrained")
    ylabels_u = [r"$\omega$ [rpm]", r"$Z$ [-]"]#
    for i in range(dim_u): #plot true states and ukf's estimates
        ax_u[i].plot(t, u[i, :])
        ax_u[i].set_ylabel(ylabels_u[i])
    ax_u[-1].set_xlabel("Time [years]")
#%% RIO conference - Plot single trajectory
plot_it = False
plot_it = True
if plot_it:
    
    #Q-measurement from Venturi, fixed density
    par_venturi = par_true_hx.copy()
    par_venturi["rho"] = 700#par_true_fx["theta_rho_0"]
    Q_fixed_rho = utils_wp.q_from_dp_venturi(y[-1,:]*100, par_venturi)*3600
    Q_ol_rho = np.zeros(dim_t)
    for k in range(dim_t):
        par_venturi["rho"] = x_ol[-1, k]
        Q_ol_rho[k] = utils_wp.q_from_dp_venturi(y[-1,k]*100, par_venturi)*3600
    par_venturi["rho"] = 700#par_true_fx["theta_rho_0"]
    
    y_size = 5
    # ylabels = [r"$x_1 [ft]$", r"$x_2 [ft/s]$", r"$x_3 [ft^3$/(lb-$s^2)]$", "$y [ft]$"]#
    # ylabels = [r"$x_1$ [ft]", r"$x_2$ [ft/s]", r"$x_3$ [*]", "$y$ [ft]"]#
    ylabels = [r"$p_1$ [bar]", r"$p_2$ [bar]", r"$p_3$ [bar]", r"$Q [m^3/h]$", r"$\rho [kg/m^3]$"]#
    kwargs_fill = dict(alpha = .2)    
    fig1, ax1 = plt.subplots(3, 1, sharex = True, layout = "constrained")
    j=0
    for i in [2,3,4]: #plot true states and ukf's estimates
        ax1[j].plot(t, x_true[i, :], color = color_true, label = "True")
        
        if i <=2:
            ax1[j].plot(t, y[i,:], marker = "x", markersize = y_size, linewidth = 0, color = color_true, label = "y")
        elif i == 3: #Q
            ax1[j].plot(t, Q_fixed_rho, marker = "x", markersize = y_size, linewidth = 0, color = color_true, label = fr"y ($\rho$ = {par_venturi['rho']} $kg/m^3$)")
            ax1[j].plot(t, Q_ol_rho, marker = "x", markersize = y_size, linewidth = 0, color = color_ol, label = fr"y ($\rho$ = OL)")
        
     
            
        
        # ax1[j].plot([np.nan, np.nan], [np.nan, np.nan], color='w', alpha=0, label=' ')
        if run_ukf:
            l_post = ax1[j].plot(t, x_post[i, :], color = color_std, label = r"UKF")[0]
        if run_ukf_norm:
            l_post_norm = ax1[j].plot(t, x_post_norm[i, :], color = color_norm, label = "NUKF")[0]
            # l_post_norm = ax1[j].plot(t, x_post_norm[i, :], label = r"$UKF_{norm}$")[0]
        if not calc_RMSE:
            ax1[j].plot(t, x_ol[i, :], color = color_ol, label = "OL")
        else: #plot it either way
            # pass
            ax1[j].plot(t, x_ol[i, :], color = color_ol, label = "OL")
        
        
        if True:
            #Standard UKF
            if run_ukf:
                ax1[j].fill_between(t, 
                                    x_post[i, :] + 2*np.sqrt(P_post[i,:]),
                                    x_post[i, :] - 2*np.sqrt(P_post[i,:]),
                                    **kwargs_fill,
                                    color = l_post.get_color())
                ax1[j].fill_between(t, 
                                    x_post[i, :] + 1*np.sqrt(P_post[i,:]),
                                    x_post[i, :] - 1*np.sqrt(P_post[i,:]),
                                    **kwargs_fill,
                                    color = l_post.get_color())
            
            #Normalized UKF
            if run_ukf_norm:
                ax1[j].fill_between(t, 
                                    x_post_norm[i, :] + 2*std_dev_x_post[i, :],
                                    x_post_norm[i, :] - 2*std_dev_x_post[i, :],
                                    **kwargs_fill,
                                    color = l_post_norm.get_color())
                ax1[j].fill_between(t, 
                                    x_post_norm[i, :] + 1*std_dev_x_post[i, :],
                                    x_post_norm[i, :] - 1*std_dev_x_post[i, :],
                                    **kwargs_fill,
                                    color = l_post_norm.get_color())
            
            
        
        ax1[j].set_ylabel(ylabels[i])
        j += 1
    ax1[-1].set_xlabel("Time [years]")
    # #Plot measurements
    # ax1[-1].plot(t, y[1,:], marker = "x", markersize = 3, linewidth = 0, label = ylabels[-1])
    # ax1[-2].set_ylabel(ylabels[-2])
    # ax1[-1].set_ylabel(ylabels[-1])
    # ax1[0].legend()        
    ax1[0].legend(ncol = 2,
                  frameon = True)      
    ax1[1].legend(ncol = 2,
                  frameon = True, loc = "lower left")      
    
    fig_q, ax_q = plt.subplots(1,1)
    ax_q.plot(t, Q_fixed_rho - x_true[3,:], marker = "x", markersize = y_size, linewidth = 0, color = color_true, label = fr"y ($\rho$ = {par_venturi['rho']} $kg/m^3$)")
    ax_q.plot(t, Q_ol_rho - x_true[3,:], marker = "x", markersize = y_size, linewidth = 0, color = color_ol, label = r"y ($\rho$ = OL)")
    ax_q.plot(t, x_post_norm[3,:] - x_true[3,:], color = color_norm, label = "NUKF")
    ax_q.set_ylabel(r"$Q_{est}-Q_{true} [m^3/h]$")
    ax_q.set_xlabel("Time [years]")
    ax_q.legend()

#%% std_dev_y
plt_kwargs = dict(linewidth = .7)
plt_kwargs = dict(linewidth = 1.2)
plot_it_sy = False
if plot_it_sy:
    fig_sy, ax_sy = plt.subplots(dim_y, 1, sharex = True, layout = "constrained")
    
    for Ni in range(N):
        for i in range(dim_y):
            ax_sy[i].plot(t, std_dev_y_trajectories[Ni][i,:], color = color_std, **plt_kwargs)
    ax_sy[0].set_ylabel(r"$\sigma_{y_1} [m]$")
    ax_sy[1].set_ylabel(r"$\sigma_{y_2} [Pa]$")
    ax_sy[1].set_xlabel(r"$t$ [s]")
            
#%% std_dev_x_post/prior
plot_it_sx = False
if plot_it_sx:
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
    cols_x = ["x1", "x2", "x3", "x4", "x5"]
    
    cost_diff = cost_func_norm - cost_func
    df_j_diff = pd.DataFrame(columns = [cols_x], data = cost_diff.T)
    
    df_cost = pd.DataFrame(columns = [cols_x], data = cost_func_norm.T)
    # df_cost["Filter"] = "sigmaRho"
    
    df_cost2 = pd.DataFrame(columns = [cols_x], data = cost_func.T)
    df_cost2["Filter"] = "Standard"
    
    fig_j, ax_j = plt.subplots(dim_x,1)
    y_label = [r"$x_1$", r"$x_2$", r"$x_3$", r"$x_4$", r"$x_5$"]
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
    
    # fig_kappa_hist, ax_kappa_hist = plt.subplots(1,1, layout = "constrained")
    # ax_kappa_hist = sns.stripplot(data = df_kappa, x = "Matrix", y = ylabel_kappa, ax = ax_kappa_hist, hue = "Method")
    # ax_kappa_hist.set_yscale("log")
    
#%% condition number trajectories - Monte Carlo
fig_kt, ax_kt = plt.subplots(3, 1, sharex = True, layout = "constrained")
ylabels = [r"$\kappa(P^+,\rho^+)$", r"$\kappa(P^-,\rho^-)$", r"$\kappa(P_y, \rho_y)$"]
# plt_kwargs = dict()

for i in range(3):
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
ax_kt[-1].set_xlabel("Time [years]")

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
            
            
