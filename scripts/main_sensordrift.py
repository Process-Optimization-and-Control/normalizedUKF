# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 11:55:08 2022

@author: halvorak
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pathlib
import scipy.stats
import utils_wp
import casadi as cd
import time
import pickle

from myFilter import sigma_points as ukf_sp
from myFilter import UKF
from myFilter import myExceptions

# import sd_model
# import sd_mhe
# import sd_simulator
# from template_mpc import template_mpc
# from template_simulator import template_simulator
# from template_mhe import template_mhe

# model = sd_model.sd_model()
# mpc = template_mpc(model)
# simulator = template_simulator(model)
# mhe = template_mhe(model)

dir_project = pathlib.Path(__file__).parent.parent 
dir_data = os.path.join(dir_project, "data")

#enumeration of variables
P1 = 0
P2 = 1
P3 = 2
Q = 3
BHP = 4
ETA = 5
YD = 6
x_var = ["P1", "P2", "P3", "Q", "BHP", "eta", "yd"] #labelling in plots etc
# x_var = ["P1", "P2", "P3", "Q", "BHP", r"$\eta$", r"$y^d$"] #labelling in plots etc

#%% Pump chart
df_ref = pd.read_csv(os.path.join(dir_data, "pump_chart_data_ref.txt"), delimiter = "\t") #Q [m3/h], H [m], eta [%]
df_ref["eta"] = df_ref["eta"] # [%]
g = 9.81 # [m/s2]
df_ref["rho"] = 997 # [kg/m3]
df_ref["bhp"] = (utils_wp.useful_power(df_ref["H"], df_ref["Q"]/3600, df_ref["rho"])/(df_ref["eta"]/100))/1000 #[kW]
df_ref["omega"] = 2900 # [rpm]
rho = df_ref["rho"][0]


#%% Param and initial values
x0 = utils_wp.get_x0()
u0 = utils_wp.get_u0()

dim_x = x0.shape[0]

(par_mean_fx, par_cov_fx, par_det_fx, 
 par_det_hx) = utils_wp.get_literature_values(df_ref)
par_fx = {**par_mean_fx, **par_det_fx}

#randomly draw values for par_fx_true from an assumed normal distribution and insert them into the dict
par_fx_true_val = np.random.multivariate_normal(
    mean = list(par_mean_fx.values()), 
    cov = par_cov_fx)
par_true_fx = {list(par_fx.keys())[i]: par_fx_true_val[i] for i in range(len(par_fx_true_val))}
par_true_fx = {**par_true_fx, **par_det_fx} #append deterministic par at the end
# par_true_fx2 = par_fx.copy()

par_true_hx = {**par_det_hx, 
               "rho": par_true_fx["rho"]}

std_dev_repeatability, accuracy, drift_rate_lim = utils_wp.get_sensor_data() #sensor data

par_se_fx = par_fx.copy()
par_se_hx = par_true_hx.copy() #no uncertainty in hx variables

unit_op = {
    "pa2mbar": 1e-2,
    "mbar2pa": 1e2}


dim_par_fx = len(par_true_fx)
dim_par_hx = len(par_true_hx)
dim_u = u0.shape[0]

#%% Def casadi functions
F,jac_p_func,_,_,_,_,_,_,_,_,_,_,_,res, p_aug_var, Qk_lin= utils_wp.ode_model_plant(par_cov_fx)
hx = utils_wp.hx_cd()
res_jac = res.jacobian()

#%%Set time

# t_end = 15 # [h]
h_p_y = 24*365 # [h/year]
min_p_h = 60 # [min/h]
# t_end = 1000/h_p_y # [year]. 1/h_p_y is 1 h
t_end = 10*h_p_y/h_p_y # [year]. 1/h_p_y is 1 h
dt_y = 60*20/min_p_h/h_p_y # [year] Measurement frequency (write in terms of minutes)
# dt_y = 1/min_p_h/h_p_y # [year] Measurement frequency (write in terms of minutes)
t = np.linspace(0, t_end, int(t_end/dt_y))
dim_t = t.shape[0]

y0 = utils_wp.eval_hx(hx, x0, par_true_hx)
dim_y = y0.shape[0]
y = np.zeros((dim_y, dim_t))
v = np.zeros((dim_y, dim_t))
y[:, 0] = y0*np.nan

x_true = np.zeros((dim_x, dim_t)) 
x_ol = np.zeros((dim_x, dim_t)) #open loop simulation
x_post_norm = np.zeros((dim_x, dim_t)) 

P_diag_post_norm = np.zeros((dim_x, dim_t)) #diagonal elements
v[P1, :] = np.random.normal(loc = 0, 
                            scale = std_dev_repeatability["PT"], size = dim_t)
v[P2, :] = np.random.normal(loc = 0, 
                            scale = std_dev_repeatability["PT"], size = dim_t)
v[P3, :] = np.random.normal(loc = 0, 
                            scale = std_dev_repeatability["PT"], size = dim_t)
v[Q, :] = np.random.normal(loc = 0, 
                            scale = std_dev_repeatability["dP"], size = dim_t)
v[BHP, :] = np.random.normal(loc = 0, 
                            scale = std_dev_repeatability["BHP"], size = dim_t)
acc_sens = np.array([np.random.uniform(-accuracy["PT"], accuracy["PT"]),#P1
                     np.random.uniform(-accuracy["PT"], accuracy["PT"]),#P2
                     np.random.uniform(-accuracy["PT"], accuracy["PT"]), #P3
                     np.random.uniform(-accuracy["dP"], accuracy["dP"]), #Q
                     np.random.uniform(-accuracy["BHP"], accuracy["BHP"]) #BHP
                     ]) #initial offset. Constant through sim
x_true[YD, :] = 0 #acc_sens[P1].copy()
acc_sens[P1] = 0

#history of estimated noise
Q_lin_hist = np.zeros((dim_x, dim_t)) #diagonal elements

#Make control law
u = np.tile(u0.reshape(-1,1), dim_t)
# u[0, :] = np.linspace(3100, 2700, dim_t) #not implemented
# u[1, :] = np.linspace(.3,.95, dim_t)


#%% Def drift
drift_type = "rw"
n_steps = 7
drift_impulse ={
    "t_step": [int(dim_t*j/n_steps) for j in range(1,n_steps)]}
drift_linear = {
    "rate": drift_rate_lim["PT"]*.9}
drift_random_walk = {
    "step_size": np.sqrt(1e-2)*5}


#%%State estimator
var_uniform = lambda a, b: 1/12*(b-a)**2 #variance of X~U(a,b)
var_uniform_sens = lambda a: var_uniform(-a, a) #variance of X~U(a,b)
R_acc = np.diag([var_uniform_sens(accuracy["PT"]), #P1
                 var_uniform_sens(accuracy["PT"]), #P2
                 var_uniform_sens(accuracy["PT"]), #P3
                 var_uniform_sens(accuracy["dP"]), #Q
                 var_uniform_sens(accuracy["BHP"]) #BHP
                 ])
R_rep = np.square(np.diag([std_dev_repeatability["PT"],
                           std_dev_repeatability["PT"],
                           std_dev_repeatability["PT"],
                           std_dev_repeatability["dP"],
                           std_dev_repeatability["BHP"]]))
# R = R_rep + R_acc*0 #don't include accuracy? Should be included for all sensors where I don't estimate the drift?
R_acc_mod = R_acc.copy()
R_acc_mod[0, 0] = 0
R = R_rep + R_acc_mod #don't include accuracy? Should be included for all sensors where I don't estimate the drift?
Qk = np.eye(dim_x)*1e-2
Qk[:dim_y, :dim_y] = R.copy()*.5 #same noise as measurement
Qk[Q, Q] = 5e-1
# Qk[BHP, BHP] = 1e-2
Qk[ETA, ETA] = 5e-2
# Qk[-1, -1] = 1e-4


# # alpha = 1
# # beta = 0
# # kappa = 3-dim_x
# alpha = 1e-2
# beta = 2.
# kappa = 0.#3-dim_x
# sigma_points_scaled = ukf_sp.MerweScaledSigmaPoints(dim_x,
#                                         alpha,
#                                         beta,
#                                         kappa)
# fx_ukf_lin = None
# nlp_update = utils_wp.create_NLP_update(R, hx, dim_x, dim_y, par_se_fx, par_se_hx)
# ukf_nlp = UKF_constrained.NLP_UKF_Kolaas(dim_x = dim_x, 
#                                      dim_z = dim_y, 
#                                      dt = 100, 
#                                      nlp_update = nlp_update, 
#                                      fx = fx_ukf_lin,
#                                      name = "lin",
#                                      points = sigma_points_scaled
#                                     )

# points_norm = spc.JulierSigmaPoints(dim_x,
#                                   kappa = 3-dim_x,
#                                   sqrt_method = sqrt_fn)

# corr_post_lim = 0.95

P0 = np.diag(x_true[:, 0]*.05)
P0[YD, YD] = R_acc[0, 0]

import copy
import sigma_points_classes as spc
sqrt_fn = lambda P: scipy.linalg.cholesky(P, lower = True)
corr_post_lim = np.inf
corr_prior_lim = copy.copy(corr_post_lim)
corr_y_lim = np.inf#.97
corr_xy_lim = np.inf
# corr_xy_lim = copy.copy(corr_y_lim)

points_norm = spc.ScaledSigmaPoints(dim_x,sqrt_method = sqrt_fn)

#kf is where Q adapts based on UT of parametric uncertainty
kf_norm = UKF.Normalized_UKF_additive_noise_v2(x0 = x_post_norm[:, 0], P0 = P0, fx = fx_ukf, hx = hx_ukf,
                                points_x = points_norm,
                                Q = Q_nom, R = R_nom,
                                corr_post_lim = corr_post_lim,
                                corr_prior_lim = corr_prior_lim,
                                corr_y_lim = corr_y_lim,
                                corr_xy_lim = corr_xy_lim
                                )



# ukf_nlp.x = x_post_lin[:, 0]
# ukf_nlp.P = copy.deepcopy(P0)
ukf_nlp.Q = Qk #to be updated in a loop
ukf_nlp.R = R #to be updated in a loop
# Q_lin = Q_nom.copy()

#define limits, drift limit changes over time
yd_lim = t*drift_rate_lim["PT"] + accuracy["PT"]
lbx = np.zeros(dim_x) #zero for lower bounds on the states, except yd. Limit for yd is changed within the loop (don't need to store huge matrices then)
# lbx[YD] = -yd_lim[i]
ubx = np.ones(dim_x)*np.inf #no upper bounds
# ubx[YD] = yd_lim[i]
lbg = -15 #max efficiency loss i 15%
ubg = 2 #margin on estimated efficiency compared to supplier's curve

#constant limits, don't need to update at every iteration
ukf_nlp.lbg = lbg
ukf_nlp.ubg = ubg

#%%Simulate the plant
# x_true[:, 0] = utils_wp.integrate_ode(F, x0, u0, par_true_fx)
dim_yd = (dim_x - 
          F.size_in(0)[0])#physical states used in rootfinder
x_true[:-1, 0] = x0[:-1]
x_true[:, 0] = utils_wp.integrate_ode_parametric_uncertainty(F, 
                                                             x_true[:, 0], 
                                                             u0, 
                                                             par_true_fx,
                                                             dim_yd
                                                             )
y[:, 0] = utils_wp.eval_hx(hx, x_true[:, 0], par_true_hx)
# x0_se = x_true[:,0].copy()
x0_se = np.random.multivariate_normal(mean = x_true[:, 0], cov = P0)
x0_se[YD] = 0. #guessing the mean value

#insert initial values
x_ol[:, 0] = x0_se.copy()
x_post_norm[:, 0] = x0_se.copy()
P_diag_post_norm[:, 0] = np.diag(P0)

ukf_nlp.x = x_post_norm[:, 0]
ukf_nlp.P = P0

Qk_min = np.eye(dim_x-dim_yd)*1e-8 #diagonal terms always added to Qk_est

t_s = time.time() #time in simulation
for i in range(1, dim_t):
    # t_span = (t[i-1], t[i])
    
    #Specify the sensor drift
    if drift_type=="impulse": #random step within drift limits at some specified time steps
        if i in drift_impulse["t_step"]: #should do step now
            print(f"i={i} doing step")
            # if i==drift_impulse["t_step"][-1]:
            #     yd_new = yd_lim[i]*.96 #see effect of large drift in final step
            # else: #random drift
            #     yd_new = np.random.uniform(low = -yd_lim[i],
            #                                high = yd_lim[i])
            yd_new = np.random.uniform(low = -yd_lim[i],
                                       high = yd_lim[i])
            x_true[YD, (i-1):] = yd_new
    elif drift_type == "linear":
        yd_new = t[i-1]*drift_linear["rate"]
        x_true[YD, (i-1):] = yd_new
    elif drift_type == "rw": #random walk
        yd_new = utils_wp.random_walk_drift(x_true[YD, (i-2)], -yd_lim[i-1], yd_lim[i-1], rw_mean = 5e-4, rw_step = drift_random_walk["step_size"])
        x_true[YD, (i-1):] = yd_new
    
        
    #generate new control input
    u[-1, i] = utils_wp.semi_random_number(u[-1, i-1], chance_of_moving = .98, u_lb = .35, u_hb = .95, rw_step = .05)
    
    #Simulate the true plant
    x_true[:, i] = utils_wp.integrate_ode_parametric_uncertainty(F, x_true[:, i-1], u[:, i], par_true_fx, dim_yd)
    
    #get measurement
    y[:, i] = (utils_wp.eval_hx(hx, x_true[:, i], par_true_hx) #"true" y
               + acc_sens #accuracy
               + v[:, i] #repeatability
               )
    
    #Simulate the open loop
    x_ol[:, i] = utils_wp.integrate_ode_parametric_uncertainty(F, x_ol[:, i-1], u[:, i], par_se_fx, dim_yd)
    
    #update drift limits based on current time step
    lbx[YD] = -yd_lim[i]
    ubx[YD] = yd_lim[i]
    
    # Estimate Qk by parametric uncertainty
    par_aug = np.hstack((u[:, i-1], np.array(list(par_se_fx.values()))))
    Qk_est_lin = np.array(Qk_lin(*[x_post_norm[:-1, i-1], par_aug]))
    Qk[:-dim_yd, :-dim_yd] = Qk_est_lin + Qk_min#random walk for drift is hand-tuned
    
    ukf_nlp.Q = Qk
    
    #Update prediction equation for UKF with w_mean (applicable only if the mean of the noise is estimated)
    fx_ukf = lambda x, dt_kf: (utils_wp.integrate_ode_parametric_uncertainty(F, x, u[:, i], par_se_fx, dim_yd)
                                    # + w_mean_gutw
                                    ) #prediction function
    
    #Do UKF steps
    ukf_nlp.predict(fx = fx_ukf, recompute_sigmas_f = True)
    ukf_nlp.update(y[:, i], lbx = lbx, ubx = ubx)
    x_post_norm[:,i] = ukf_nlp.x
    P_diag_post_norm[:, i] = np.diag(ukf_nlp.P)
    
    if (i%100)==0:
        print(f"Iter {i}/{dim_t} done. t_tot = {(time.time()-t_s)/60: .1f} min")

head = utils_wp.calc_pump_head(par_true_fx["p0"], x_true[0])
dp_venturi = utils_wp.dp_from_q_venturi(x_true[Q,:]/3600, df_ref["rho"][0], par_true_hx)*unit_op["pa2mbar"]
#%% Plot
plot_it = True
plt_std_dev = True
if plot_it:
    alpha_fill = .2
    kwargs_pred = {"linestyle": "dashed"}
    kwargs_y = dict(label = "y", s = 5, alpha = .5, marker = "o")
    kwargs_norm = {"alpha": alpha_fill}
    kwargs_ol = {"linestyle": "dotted", "label": "OL"}
    
    fig_x, ax_x = plt.subplots(5, 1, sharex = True)
    for i in range(P3+1):
        l = ax_x[0].plot(t, x_true[i, :], label = x_var[i])
        ax_x[0].scatter(t, y[i, :], **kwargs_y, color = l[0].get_color())
        l_ukf = ax_x[0].plot(t, x_post_norm[i, :], 
                            # label = r"$\hat{x}^+_{GenUT}$",
                            color = l[0].get_color(),
                            **kwargs_pred)
        ax_x[0].plot(t, x_ol[i, :], **kwargs_ol, color = l[0].get_color())
        
        if plt_std_dev:
            kwargs_norm.update({"color": l_ukf[0].get_color()})
            ax_x[0].fill_between(t, 
                                x_post_norm[i, :] + 2*np.sqrt(P_diag_post_norm[i,:]),
                                x_post_norm[i, :] - 2*np.sqrt(P_diag_post_norm[i,:]),
                                **kwargs_norm)
            ax_x[0].fill_between(t, 
                                x_post_norm[i, :] + 1*np.sqrt(P_diag_post_norm[i,:]),
                                x_post_norm[i, :] - 1*np.sqrt(P_diag_post_norm[i,:]),
                                **kwargs_norm)
    
    
    l = ax_x[1].plot(t, x_true[Q, :], label = x_var[Q])
    ax_x[1].scatter(t, utils_wp.q_from_dp_venturi(y[Q, :]*unit_op["mbar2pa"], 
                                                  par_true_hx)*3600, **kwargs_y, color = l[0].get_color())
    l_ukf = ax_x[1].plot(t, x_post_norm[Q, :], 
                        label = r"$\hat{x}^+$",
                        **kwargs_pred)
    if plt_std_dev:
        kwargs_norm.update({"color": l_ukf[0].get_color()})
        ax_x[1].fill_between(t, 
                            x_post_norm[Q, :] + 2*np.sqrt(P_diag_post_norm[Q,:]),
                            x_post_norm[Q, :] - 2*np.sqrt(P_diag_post_norm[Q,:]),
                            **kwargs_norm)
        ax_x[1].fill_between(t, 
                            x_post_norm[Q, :] + 1*np.sqrt(P_diag_post_norm[Q, :]),
                            x_post_norm[Q, :] - 1*np.sqrt(P_diag_post_norm[Q, :]),
                            **kwargs_norm)
    ax_x[1].plot(t, x_ol[Q, :], **kwargs_ol)
    
    l = ax_x[2].plot(t, x_true[BHP, :], label = x_var[BHP])
    ax_x[2].scatter(t, y[BHP, :], **kwargs_y, color = l[0].get_color())
    l_ukf = ax_x[2].plot(t, x_post_norm[BHP, :], 
                        label = r"$\hat{x}^+$",
                        **kwargs_pred)
    if plt_std_dev:
        kwargs_norm.update({"color": l_ukf[0].get_color()})
        ax_x[2].fill_between(t, 
                            x_post_norm[BHP, :] + 2*np.sqrt(P_diag_post_norm[BHP,:]),
                            x_post_norm[BHP, :] - 2*np.sqrt(P_diag_post_norm[BHP,:]),
                            **kwargs_norm)
        ax_x[2].fill_between(t, 
                            x_post_norm[BHP, :] + 1*np.sqrt(P_diag_post_norm[BHP, :]),
                            x_post_norm[BHP, :] - 1*np.sqrt(P_diag_post_norm[BHP, :]),
                            **kwargs_norm)
    ax_x[2].plot(t, x_ol[BHP, :], **kwargs_ol)
    
    ax_x[3].plot(t, x_true[ETA, :], label = x_var[ETA])
    l_ukf = ax_x[3].plot(t, x_post_norm[ETA, :], 
                        label = r"$\hat{x}^+$",
                        **kwargs_pred)
    if plt_std_dev:
        kwargs_norm.update({"color": l_ukf[0].get_color()})
        ax_x[3].fill_between(t, 
                            x_post_norm[ETA, :] + 2*np.sqrt(P_diag_post_norm[ETA,:]),
                            x_post_norm[ETA, :] - 2*np.sqrt(P_diag_post_norm[ETA,:]),
                            **kwargs_norm)
        ax_x[3].fill_between(t, 
                            x_post_norm[ETA, :] + 1*np.sqrt(P_diag_post_norm[ETA, :]),
                            x_post_norm[ETA, :] - 1*np.sqrt(P_diag_post_norm[ETA, :]),
                            **kwargs_norm)
    ax_x[3].plot(t, x_ol[ETA, :], **kwargs_ol)
    
    #plot sensor drift
    ax_x[4].plot(t, x_true[YD, :], label = r"$x_{true}$")
    ax_x[4].plot([t[0], t[-1]], [yd_lim[0], yd_lim[-1]],
                 color = "r", linestyle = "dashed", label = r"$y^d_{lim}$")
    ax_x[4].plot([t[0], t[-1]], [-yd_lim[0], -yd_lim[-1]],
                 color = "r", linestyle = "dashed")
    l_ukf = ax_x[4].plot(t, x_post_norm[YD, :], 
                        label = r"$\hat{x}^+$",
                        **kwargs_pred)
    if plt_std_dev:
        kwargs_norm.update({"color": l_ukf[0].get_color()})
        ax_x[4].fill_between(t, 
                            x_post_norm[YD, :] + 2*np.sqrt(P_diag_post_norm[YD,:]),
                            x_post_norm[YD, :] - 2*np.sqrt(P_diag_post_norm[YD,:]),
                            **kwargs_norm)
        ax_x[4].fill_between(t, 
                            x_post_norm[YD, :] + 1*np.sqrt(P_diag_post_norm[YD, :]),
                            x_post_norm[YD, :] - 1*np.sqrt(P_diag_post_norm[YD, :]),
                            **kwargs_norm)
    ax_x[4].plot(t, x_ol[YD, :], **kwargs_ol)
    
    ax_x[0].legend()
    ax_x[-1].legend()
    
    ax_x[0].set_ylabel(r"P [kPa]")
    ax_x[1].set_ylabel(x_var[Q] + r" [$m^3/h$]")
    ax_x[2].set_ylabel(x_var[BHP] + " [kW]")
    ax_x[3].set_ylabel(x_var[ETA] + " [-]")
    ax_x[4].set_ylabel(x_var[YD] + r" [kPa]")
    ax_x[-1].set_xlabel(r"t [year]")
    
    #%% Pump curve plot
    fig_ref, (ax_h, ax_bhp, ax_eta) = plt.subplots(3,1, sharex = True)
    
    kwargs_fat = dict(s = 15, label = "FAT", marker = "x")
    # kwargs_sc = dict(label = "Supplier's curve")
    kwargs_true = dict(label = r"$x_{true}$")
    kwargs_se = dict(linestyle = "dashed", label = r"$\hat{x}^+$")
    kwargs_constraint = dict(color = "r")
    kwargs_y_est = kwargs_y.copy()
    kwargs_y_est["label"] = r"$y_{calc}$"
    
    df_xtrue = pd.DataFrame(data = x_true.T, columns = x_var)
    df_xtrue = df_xtrue.sort_values(by = ["Q"])
    # df_xtrue.sort_values(by = ["Q"])
    
    
    df_xukf = pd.DataFrame(data = x_post_norm.T, columns = x_var)
    # n_skip_ukf = 5 #skip first 5 estimates (bad first estimate)
    # df_xukf = df_xukf.iloc[n_skip_ukf:,:]
    df_xukf = df_xukf.sort_values(by = ["Q"])
    
    y_q = utils_wp.q_from_dp_venturi(y[Q,:]*unit_op["mbar2pa"], par_true_hx)*3600
    
    #Plot Q vs H
    l=ax_h.plot(df_xtrue["Q"], utils_wp.calc_pump_head(par_true_fx["p0"], df_xtrue["P1"]), **kwargs_true)
    ax_h.scatter(y_q, utils_wp.calc_pump_head(par_true_fx["p0"], y[P1, :]), **kwargs_y_est, color = l[0].get_color())
    
    ax_h.scatter(df_ref["Q"], df_ref["H"], **kwargs_fat)
    
    l=ax_h.plot(df_xukf["Q"], utils_wp.calc_pump_head(par_se_fx["p0"], df_xukf["P1"]),**kwargs_se)
    
    ax_h.set_ylabel("H [m]")
    ax_h.legend()
    
    #Plot Q vs BHP
    l=ax_bhp.plot(df_xtrue["Q"], df_xtrue["BHP"], **kwargs_true)
    ax_bhp.scatter(y_q, y[BHP,:], **kwargs_y, color = l[0].get_color())
    ax_bhp.scatter(df_ref["Q"], df_ref["bhp"], **kwargs_fat)
    ax_bhp.plot(df_xukf["Q"], df_xukf["BHP"],**kwargs_se)
    ax_bhp.set_ylabel(x_var[BHP] + " [kW]")
    ax_bhp.legend()
    
    #Plot Q vs BHP
    l=ax_eta.plot(df_xtrue["Q"], df_xtrue["eta"], **kwargs_true)
    ax_eta.scatter(y_q, 
                   utils_wp.calc_pump_eta(par_true_fx["p0"], y[P1, :], y_q, y[BHP, :]),
                   **kwargs_y_est, color = l[0].get_color())
    ax_eta.scatter(df_ref["Q"], df_ref["eta"], **kwargs_fat)
    ax_eta.plot(df_xukf["Q"], df_xukf["eta"],**kwargs_se)
    ax_eta.set_ylabel(x_var[ETA] + " [%]")
    
    func_eta = lambda q, par: par["eta_a0"] + par["eta_a1"]*q + par["eta_a2"]*q**2
    ub_eta = func_eta(df_xukf["Q"], par_se_fx) + ubg
    lb_eta = (ub_eta - ubg) +lbg
    ax_eta.plot(df_xukf["Q"], ub_eta, **kwargs_constraint)
    ax_eta.plot(df_xukf["Q"], lb_eta, **kwargs_constraint, label = "constraint")
    ax_eta.legend()
    
    
    #%% u - input usage
    fig_u, (ax_u) = plt.subplots(1,1)
    ax_u.plot(t, u[-1,:]*100)
    ax_u.set_xlabel("t [year]")
    ax_u.set_ylabel("Valve opening [%]")
    
    #%% Save variables
    dir_project = pathlib.Path(__file__).parent.parent 
    dir_data = os.path.join(dir_project, "data")
    
    np.save(os.path.join(dir_data, "t.npy"), t)
    np.save(os.path.join(dir_data, "u.npy"), u)
    np.save(os.path.join(dir_data, "y.npy"), y)
    np.save(os.path.join(dir_data, "yd_lim.npy"), yd_lim)
    np.save(os.path.join(dir_data, "x_true.npy"), x_true)
    np.save(os.path.join(dir_data, "x_ol.npy"), x_ol)
    np.save(os.path.join(dir_data, "x_post_norm.npy"), x_post_norm)
    np.save(os.path.join(dir_data, "P_diag_post_norm.npy"), P_diag_post_norm)
    df_ref.to_csv(os.path.join(dir_data, "df_ref.csv"))
    lb_eta.to_csv(os.path.join(dir_data, "lb_eta.csv"))
    ub_eta.to_csv(os.path.join(dir_data, "ub_eta.csv"))
    
    a_file = open(os.path.join(dir_data, "par_true_hx.pkl"), "wb")
    pickle.dump(par_true_hx, a_file)
    a_file. close()
    
    a_file = open(os.path.join(dir_data, "par_true_fx.pkl"), "wb")
    pickle.dump(par_true_fx, a_file)
    a_file. close()
    
    a_file = open(os.path.join(dir_data, "par_se_fx.pkl"), "wb")
    pickle.dump(par_se_fx, a_file)
    a_file. close()
    
    a_file = open(os.path.join(dir_data, "par_se_hx.pkl"), "wb")
    pickle.dump(par_se_hx, a_file)
    a_file. close()
    
    

    # df_conc.to_csv(os.path.join(dir_data, "consistency.csv"))
    # if "df_cost_rmse_all" in locals(): #check if variable exists
    #     df_cost_rmse_all.to_csv(os.path.join(dir_data, "df_cost_rmse_all.csv"))
    #     df_cost_mean_all.to_csv(os.path.join(dir_data, "df_cost_mean_all.csv"))
