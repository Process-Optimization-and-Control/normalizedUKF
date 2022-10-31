# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 11:26:43 2022

@author: halvorak
"""

import numpy as np
import scipy.stats
import casadi as cd
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import casadi as cd
import seaborn as sns
import copy
# import sklearn.preprocessing


# from 
#Self-written modules
# import sigma_points_classes as spc
# import unscented_transformation as ut

# from time_out_manager import time_limit

font = {'size': 14}

matplotlib.rc('font', **font)
#enumeration of variables
P1 = 0
P2 = 1
P3 = 2
Q = 3
RHO = 4

def ode_model_plant(par_cov):
    
    #Make parameters
    
    #pump system parameters
    h_a0 = cd.SX.sym("h_a0", 1) # [?] #flow to head polynomial par (h=a0+a1*q+a2*q**2)
    h_a1 = cd.SX.sym("h_a1", 1) # [?]
    h_a2 = cd.SX.sym("h_a2", 1) # [?]

    cv = cd.SX.sym("cv", 1) # [?] #cv, control valve
    
    #reservoir parameters
    theta_p1_0 = cd.SX.sym("theta_p1_0", 1) #p1 = p1_0 + theta_p1_1*(t) + theta_p1_2*t**2
    theta_p1_1 = cd.SX.sym("theta_p1_1", 1) #p1 = p1_0 + theta_p1_1*t + theta_p1_2*t**2
    theta_p1_2 = cd.SX.sym("theta_p1_2", 1) #p1 = p1_0 + theta_p1_1*t + theta_p1_2*t**2
    theta_p3_0 = cd.SX.sym("theta_p3_0", 1) #p3 = p3_0 + theta_p3_1*t + theta_p3_2*t**2
    theta_p3_1 = cd.SX.sym("theta_p3_1", 1) #p3 = p3_0 + theta_p3_1*t + theta_p3_2*t**2
    theta_p3_2 = cd.SX.sym("theta_p3_2", 1) #p3 = p3_0 + theta_p3_1*t + theta_p3_2*t**2
    theta_rho_0 = cd.SX.sym("theta_rho_0", 1) #rho = rho_0 + theta_rho_1*t 
    theta_rho_1 = cd.SX.sym("theta_rho_1", 1) #rho = rho_0 + theta_rho_1*t
    
    #fixed parameters
    g = 9.81 #[m/s2]
    omega_ref = 3000 # [rpm] reference speed for the pump curve
    
    
    #States
    p1 = cd.SX.sym("p1", 1)
    p2 = cd.SX.sym("p2", 1)
    p3 = cd.SX.sym("p3", 1)
    Q = cd.SX.sym("Q", 1)
    rho = cd.SX.sym("rho", 1) #drift term
    
    
    #Inputs
    omega = cd.SX.sym("omega", 1) # [rpm] pump speed
    u_cv = cd.SX.sym("u_cv", 1) # [%] opening of choke valve
    
    #Time
    t = cd.SX.sym("t", 1) # [s,min,h,d,y], time
    
    #Concatenate equation, states, inputs and parameters
    x_var = cd.vertcat(p1, p2, p3, Q, rho)
    u_var = cd.vertcat(omega, u_cv)
    p_var = cd.vertcat(h_a0, h_a1, h_a2,
                       theta_p1_0, theta_p1_1, theta_p1_2,
                       theta_p3_0, theta_p3_1, theta_p3_2,
                       theta_rho_0, theta_rho_1,
                       cv
                       )
    p_aug_var = cd.vertcat(u_var, p_var, t) #input order to the function
    
    #scaling factors
    u_ref = 1
    pa2kpa = 1/1e3
    pa2bar = 1/1e5
    rho_water = 997 #kg/m3
    
    #root finding system (residuals). Define supplementary equations
    H_ref = h_a0 + h_a1*Q + h_a2*(Q**2) # [m]
    H = H_ref*(omega/omega_ref)**2 #affinity law correction
    SG = rho/rho_water #specific gravity
    
    #residuals to be solved
    res_p1 = theta_p1_0 + theta_p1_1*t + theta_p1_2*(t**2)  - p1 
    res_p2 = p1 + (rho * g * H)*pa2bar - p2
    res_p3 = theta_p3_0 + theta_p3_1*t + theta_p3_2*(t**2)  - p3 
    res_q = Q - cv*u_cv*cd.sqrt((p2 - p3)/SG)
    res_rho = theta_rho_0 + theta_rho_1*t - rho 
    
    res_all = cd.vertcat(res_p1, res_p2, res_p3, res_q, res_rho)
    
    res = cd.Function("res", 
                      [x_var, p_aug_var], 
                      [cd.vertcat(res_p1, res_p2, res_p3, res_q, res_rho)] #equations/residuals
                      )
    # jac_p_func = res.jacobian(p_aug_var)
    
    #Form ode dict and integrator
    # opts = {"abstol": 1e-14,
    #         "linear_solver": "csparse"
    #         }
    opts = {}
    F = cd.rootfinder("F", "newton", res, opts)
    # jac_p_func = cd.jacobian(res_all, p_aug_var)
    jac_p_func = cd.jacobian(res_all, p_var)
    Qk_func = jac_p_func @ par_cov @ jac_p_func.T
    Qk_lin = cd.Function("Qk_lin", [x_var, p_aug_var], [Qk_func])
    # jac_p_func = res.jacobian(1)
    jac_p = cd.Function("jac_p", [x_var, p_aug_var], [jac_p_func])
    # jac_p = None
    S_zz = None
    S_xz = None
    S_xp = None
    S_zp = None
    z_var = None
    diff_eq = None
    alg = None
    obj_fun = None
    return F,jac_p,S_zz,S_xz,S_xp,S_zp,x_var,z_var,u_var,p_var,diff_eq,alg,obj_fun, res, p_aug_var, Qk_lin

#test

# F,jac_p,S_zz,S_xz,S_xp,S_zp,x_var,z_var,u_var,p_var,diff_eq,alg,obj_fun, res, p_aug_var, Qk_lin = ode_model_plant()

def integrate_ode(F, x0, uk, par_fx, tk):
    x0 = cd.vertcat(x0)
    pk = list(uk)
    pk.extend(list(par_fx.values())) #combines two lists
    pk.append(tk) #this is a float
    xk = F(x0, cd.vcat(pk))
    # xf = Fend["xf"]
    # xf_np = np.array(xf).flatten()
    # return xf_np
    return np.array(xk).flatten()

def integrate_ode_parametric_uncertainty(F, x0, uk, par_fx, dim_par):
    xf_ode = integrate_ode(F, x0[:-dim_par], uk, par_fx)
    xf = np.hstack((xf_ode, x0[-dim_par:]))
    return xf


# def get_measurement_matrix(dim_x, par):
#     H = np.eye(dim_x)
#     H[ETA, ETA] = 0
#     return H

# def hx(x, par):#, v):
#     #change this later? For volume, we assume dP measurement. From the venturi flowmeter for the subsea pump, we have an accuracy of 0,065%*span_DP*multiplier_accuracy/sigma_level =0,065/100*320mbar*0,5/2 = 0,052 mbar. 
#     H = get_measurement_matrix(x.shape[0], par)
#     y = np.dot(H, x)# + v
#     # y[-1] = dp_measurement(y[-1], par) #need to add Venturi measurement
#     return y

def hx_cd():
    
    #parameters for the venturi
    d = cd.SX.sym("d", 1)
    D = cd.SX.sym("D", 1)
    eps = cd.SX.sym("eps", 1)
    C = cd.SX.sym("C", 1)
    rho = cd.SX.sym("rho", 1)
    beta = d/D
    
    #states
    p1 = cd.SX.sym("p1", 1)
    p2 = cd.SX.sym("p2", 1)
    p3 = cd.SX.sym("p3", 1)
    Q = cd.SX.sym("Q", 1)
    rho = cd.SX.sym("rho", 1)
    
    Q_si = Q/3600 #[m3/s], SI units
    
    x_var = cd.vertcat(p1, p2, p3, Q, rho)
    p_var = cd.vertcat(C, eps, d, D)
    
    
    dP = (rho*(1-(beta**4))/2)*(Q_si/(
        (cd.pi/4)*(d**2)*C*eps))**2
    
    dP_mbar = dP*1e-2 #[mbar] from Pa
    
    hx = cd.Function("hx", 
                      [x_var, p_var], 
                      [cd.vertcat(p1, #p1 has sensor drift
                                  p2, 
                                  p3, 
                                  dP_mbar)] 
                      )
    return hx

def eval_hx(hx, xk, par_hx):
    xk = cd.vertcat(xk)
    pk = list(par_hx.values())
    yk = hx(xk, cd.vcat(pk))
    return np.array(yk).flatten()

# def create_NLP_update(R, hx, dim_x, dim_y, par_fx, par_hx):
    
#     assert isinstance(hx, cd.Function), f"hx is not a casadi Function, hx is {type(hx)}"
#     assert hx.size_in(0) == (dim_x, 1), f"Check input size of hx. Have hx.size_in(0) = {hx.size_in(0)} and dim_x = {dim_x}"
#     assert hx.size_out(0) == (dim_y, 1), f"Check output size of hx. Have hx.size_out(0) = {hx.size_out(0)} and dim_y = {dim_y}"
    
#     R_inv = cd.inv(R)
    
#     #"parameters"
#     y = cd.SX.sym("y", dim_y, 1)
#     sig_f = cd.SX.sym("sig_f", dim_x, 1) #propagated sigma points
#     P_prior_array = cd.SX.sym("P_prior", dim_x**2)
#     P_prior = P_prior_array.reshape((dim_x, dim_x))
#     # P_prior = cd.reshape("P_prior", dim_x, dim_x)
#     P_prior_inv = cd.inv(P_prior)
    
#     par_hx_val = cd.vcat(list(par_hx.values()))
    
#     p = cd.vertcat(y, sig_f, P_prior_array)
    
#     # Decision variable
#     sig_post = cd.SX.sym("sig_post", dim_x, 1) #estimate of measurements
    
#     y_pred = hx(sig_post, par_hx_val)
#     cost = (cd.transpose((y - y_pred)) @ R_inv @ (y - y_pred) 
#             + cd.transpose((sig_post - sig_f)) @ P_prior_inv @ (sig_post - sig_f))
    
#     #define bounds
#     eta_max = (par_fx["eta_a0"] + par_fx["eta_a1"]*sig_post[Q]
#                + par_fx["eta_a2"]*sig_post[Q]**2)
    
#     g_eta = sig_post[ETA] - eta_max
    
#     nlp = {"x": sig_post, "f": cost, "g": g_eta, "p": p}
#     opts_solver = {"print_time": 0, "ipopt": {"print_level": 0}}
#     nlp_update = cd.nlpsol("nlp_update", "ipopt", nlp, opts_solver)
#     return nlp_update
    
    

def useful_power(h, q, rho, g = 9.81):
    """
    Useful hydraulic power

    Parameters
    ----------
    h : TYPE np.array((n,))
        DESCRIPTION. Head of pump [m]
    q : TYPE np.array((n,))
        DESCRIPTION. Flowrate through pump [m3/s]
    rho : TYPE np.array((n,))
        DESCRIPTION. Density of liquid [kg/m3]
    g : TYPE, optional float
        DESCRIPTION. The default is 9.81. Gravitational acceleration [m/s2]

    Returns
    -------
    bhp_u : TYPE np.array((n,))
        DESCRIPTION. Useful power [W]

    """
    bhp_u = rho*g*q*h
    return bhp_u


def poly_2nd_order(q, a0, a1, a2):
    # Used for fitting a 2nd order polynomial
    y = a0 + a1*q + a2*q**2
    return y

def lin_reg(q, a0, a1):
    # Used for fitting a 1st order polynomial
    y = a0 + a1*q 
    return y

def get_literature_values(df_reservoir, df_pump_chart):
    
    par_opt_h, par_cov_h = scipy.optimize.curve_fit(poly_2nd_order, df_pump_chart["Q"], df_pump_chart["H"]) # Q [m3/h] ==> H [m]
    
    #reservoir parameters
    par_opt_p1, par_cov_p1 = scipy.optimize.curve_fit(poly_2nd_order, df_reservoir["t"], df_reservoir["p1"]) # t [-] ==> p1 [bar] 
    par_opt_p3, par_cov_p3 = scipy.optimize.curve_fit(poly_2nd_order, df_reservoir["t"], df_reservoir["p3"]) # t [-] ==> p3 [bar] 
    par_opt_rho, par_cov_rho = scipy.optimize.curve_fit(lin_reg, df_reservoir["t"], df_reservoir["rho"]) # t [-] ==> rho [kg/m3] 
    
    par_mean_fx = {
        "h_a0": par_opt_h[0], # [?]
        "h_a1": par_opt_h[1], # [?]
        "h_a2": par_opt_h[2], # [?]
        "theta_p1_0": par_opt_p1[0], # [?]
        "theta_p1_1": par_opt_p1[1], # [?]
        "theta_p1_2": par_opt_p1[2], # [?]
        "theta_p3_0": par_opt_p3[0], # [?]
        "theta_p3_1": par_opt_p3[1], # [?]
        "theta_p3_2": par_opt_p3[2], # [?]
        "theta_rho_0": par_opt_rho[0], # [?]
        "theta_rho_1": par_opt_rho[1], # [?]
        "cv": 47.505 #[(m3/h)/sqrt(bar / kg/m3)]
        }
    
    
    par_sigma_fx_univar = {
        "cv": par_mean_fx["cv"]*0.03, #[(kg/h)/sqrt(kPa kg/m3)]
        }
    
    cov_univar = np.diag(np.square(np.array(list(par_sigma_fx_univar.values()))))
    par_cov_fx = scipy.linalg.block_diag(par_cov_h, par_cov_p1, par_cov_p3, par_cov_rho, cov_univar)
    
    
    par_det_hx = {#Parameters for the Venturi flowmeter calculations. Doc reference:  D110-AKSEH-I-CA-0003
    "C": 1.01,#[-], flow coefficient
    "epsilon": 1., #[-], expansibility factor
    # "d": 44.565/1e3, #[m], Venturi throat diameter
    # "D": 102.3/1e3 #[m], upstream internal pipe diameter
    "d": 52.5/1e3, #[m], Venturi throat diameter
    "D": 154.051/1e3 #[m], upstream internal pipe diameter
    }
    
    x0 = get_x0()
    P0 = np.diag(0.05*x0)
    Q_nom = np.eye(x0.shape[0])*1e-6
    std_dev_repeatability, accuracy, drift_rate = get_sensor_data()
    R_nom = np.diag(np.array([std_dev_repeatability["PT"]**2,
                              std_dev_repeatability["PT"]**2,
                              std_dev_repeatability["PT"]**2,
                              std_dev_repeatability["dP"]**2]))
    
    return x0, P0, par_mean_fx, par_det_hx, par_cov_fx, Q_nom, R_nom

def get_sensor_data():
    # Describe sensors and upload the data. Data from the plant is stored in the sensors
    # From datasheets of sensors (inherent properties of the physically installed sensors).
    sensor_specs = {"FS": np.array([0, 345]),# [bara]. "Full scale" equivalent to calibrated range, pressure sensor
            "CR": np.array([0, 120]) + 273.15, # [degK]. Calibrated range, temperature sensor
            "acc_PT": 0.03/100, # +/-0,03% * FS, accuracy of PT sensor
            "acc_TT": 0.25/100, # +/-0,25% * CR, accuracy of TT sensor
            "stab_PT": 0.02/100, # <+/-0,02% FS/Yr, stability (drift) of PT sensor
            "stab_TT": 0.2/100, # <+/-0,2% CR/Yr, stability (drift) of TT sensor
            "acc_DP": .065/100,# 0,65% of Span of DP sensor
            "span_DP": 320, #[mbar], equivalent to calibrated range of DP sensor
            "stab_DP": .1/100/2, #[mbar/year], stability/drift as a function of URL for DP sensor
            "url_DP": 1300, #[mbar], upper range limit of DP sensor
            "line_pressure_effect_zero_DP": .2/100, #[mbar] line pressure effect for DP sensor NB: says "per 100 bar", this is not accounted for
            "line_pressure_effect_span_DP": .2/100, #[mbar ] line pressure effect for DP sensor NB: says "per 100 bar", this is not accounted for
            "CR_VSD": np.array([0., 1600.]), #GUESSED VALUE - NOT FROM DS
            "acc_VSD": .01/100, #GUESSED VALUE
            "stab_VSD": .01/100 #GUESSED VALUE
            }
    bar2kpa = 100
    accuracy = {
        "PT": np.max(sensor_specs["FS"]) * sensor_specs["acc_PT"]*bar2kpa, #[kPa]
        "dP": sensor_specs["acc_DP"]*sensor_specs["span_DP"], #[mbar]
        "BHP": .01 #[kW] - guessed value
        }
    
    #assume repeatability is 50% of accuracy for all the sensors
    std_dev_repeatability = {key: value/2 for (key, value) in accuracy.items()}
    
    drift_rate = {
        "PT": sensor_specs["stab_PT"]*np.max(sensor_specs["FS"])*bar2kpa, #[kPa/year]
        "dP": sensor_specs["stab_DP"]*sensor_specs["url_DP"], #[mbar/year]
        "BHP": accuracy["BHP"]*.8 #[kW/year] - guessed value (80% of accuracy)
        }
    
    return std_dev_repeatability, accuracy, drift_rate

def get_x0():
    x0 = np.array([50, #p1
                   80, #p2
                   50, #p3
                   100, #Q
                   600 # rho
                   ])
    return x0

def get_u0():
    u0 = np.array([3000, #[rpm], omega
                   .60 #[%], choke valve opening
                   ])
    return u0

def compute_performance_index_valappil(x_kf, x_ol, x_true, RMSE = True):
    J_RMSE = np.linalg.norm(x_kf - x_true, axis = 1, ord = 2)
    if RMSE:
        return J_RMSE
    else:
        J_valappil = np.divide(J_RMSE,
                               np.linalg.norm(x_ol - x_true, axis = 1, ord = 2))
    return J_valappil



def calc_pump_head(p_in, p_out, rho = 997):
    g = 9.81
    kpa2pa = 1000
    h = (p_out-p_in)*kpa2pa/(g*rho)
    return h #[m]

def calc_pump_eta(p_in, p_out, Q, bhp, rho = 997):
    g = 9.81
    kpa2pa = 1000
    H = calc_pump_head(p_in, p_out, rho = rho)
    bhp_u = (rho*g*(Q/3600))*H/1000 # [kW] - useful power, for calc eta
    eta = (bhp_u/bhp)*100 # [%]
    return eta

def dp_from_q_venturi(q, rho, par):
    """
    Venturi equation. Back-calculates the pressure drop over a Venturi for a given flowrate and density

    Parameters
    ----------
    q : TYPE float or np.array() [m3/s]
        DESCRIPTION. Calculated volumetric flowrate through the Venturi.
    rho : TYPE float or np.array() [kg/m3]
        DESCRIPTION. Density of the fluid
    par : TYPE dict
        DESCRIPTION. Containing the design details of the venturi flowmeter. Has the following keys:
            "epsilon": Expansibility factor [-]
            "d": Venturi throat diameter [m]
            "D": Upstream pipe diameter [m]
            "beta": d/D [-]
            "C": Flow coefficient [-]

    Returns
    -------
    dP : TYPE float or np.array() [Pa]
        DESCRIPTION. The measured pressure drop over the Venturi

    """
    beta = par["d"]/par["D"]
    dP = (rho*(1-(beta**4))/2)*np.square(q/(
        (np.pi/4)*(par["d"]**2)*par["C"]*par["epsilon"]))
    return dP

def q_from_dp_venturi(dp, par):
    """
    Venturi equation. Calculates volumetric flowrate from dp over a venturi orifice

    Parameters
    ----------
    dp : TYPE float or np.array() [Pa]
        DESCRIPTION. The measured pressure drop over the Venturi
    rho : TYPE float or np.array() [kg/m3]
        DESCRIPTION. Density of the fluid
    par : TYPE dict
        DESCRIPTION. Containing the design details of the venturi flowmeter. Has the following keys:
            "epsilon": Expansibility factor [-]
            "d": Venturi throat diameter [m]
            "D": Upstream pipe diameter [m]
            "beta": d/D [-]
            "C": Flow coefficient [-]
            "rho": Density of liquid [kg/m3]

    Returns
    -------
    q : TYPE float or np.array() [m3/s]
        DESCRIPTION. Calculated volumetric flowrate through the Venturi.

    """
    beta = par["d"]/par["D"]
    q = ((np.pi/4)*(par["d"]**2)*par["C"]*par["epsilon"])*np.sqrt(
        2*np.maximum(0, dp)#element wise maximum between two arrays
        /(par["rho"]*(1-(beta**4))))
    return q 

def evaluate_jac_p(jac_p_fun, x, u, par_nom):
    """
    Calculate df/dp|x, u, par_nom

    Parameters
    ----------
    jac_p_fun : TYPE casadi.Function
        DESCRIPTION. Takes as input [x, p_aug]
    x : TYPE np.array((dim_x,))
        DESCRIPTION. Values of x. dim_x must correspond to casadi variable x in ode_model_plant
    t_span : TYPE tuple
        DESCRIPTION. Integration time. dt=t_span[1]-t_span[0]
    u : TYPE np.array
        DESCRIPTION. Input
    par_nom : TYPE dict
        DESCRIPTION. Nominal parameter values. p_aug = [u, par_nom.values(), dt]

    Returns
    -------
    TYPE np.array((dim_x, dim_u + dim_par + 1))
        DESCRIPTION.

    """
    par_aug = np.hstack((u, np.array(list(par_nom.values()))))
    jac_p_args = [x, par_aug]
    jac_p_aug_val = jac_p_fun(*jac_p_args) #cd.DM type. Shape: ((dim_f=dim_x, dim_p_aug))
    print(jac_p_aug_val)
    jac_p_aug_val = np.array(jac_p_aug_val) #cast to numpy
    
    #Extract the correct jacobian. Have df/dp_aug, want only df_dp
    dim_u = u.shape[0]
    dim_x = x.shape[0]
    dim_par = len(par_nom)
    jac_p_val = jac_p_aug_val[:, dim_u:-1]
    dim_par = jac_p_val.shape[1]
    if not (dim_par == jac_p_val.shape[1]) and (jac_p_val.shape[0] == dim_x):
        raise ValueError(f"Dimension mismatch. Par: {jac_p_val.shape[1]} and {dim_par}. States: {jac_p_val.shape[0]} and {dim_x}")
        
    return jac_p_val


def get_Q_from_linearization(jac_p_fun, x, u, par_nom, par_cov):
    jac_p = evaluate_jac_p(jac_p_fun, x, u, par_nom)
    Q = jac_p @ par_cov @ jac_p.T
    return Q


def semi_random_number(u, chance_of_moving = .98, u_lb = .35, u_hb = .95, rw_step = .01):
    random_number = np.random.uniform(low = 0., high = 1.)
    if random_number >= (1-chance_of_moving):
        return u
    else:
        new_u = np.clip(u + np.random.normal(loc = 0, scale = rw_step), u_lb, u_hb)
        # new_u = u + np.random.normal(loc = 0, scale = rw_step)
        # if new_u < u_lb:
        #     new_u = u_lb
        # if new_u > u_hb:
        #     new_u = u_hb
        # # new_u = np.random.uniform(low = u_lb, high = u_hb)
        return new_u
def random_walk_drift(yd, yd_lb, yd_hb, rw_mean = 1e-4, rw_step = 1e-3):
    new_yd = yd + np.random.normal(loc = rw_mean, scale = rw_step)
    if new_yd < yd_lb:
        new_yd = yd_lb
    if new_yd > yd_hb:
        new_yd = yd_hb
    return new_yd

def correlation_from_covariance(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation, v