# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 13:47:07 2021

@author: halvorak
"""
import numpy as np
import scipy.stats
import sklearn.datasets
#Self-written modules
import sigma_points_classes as spc
import unscented_transformation as ut
import matplotlib.pyplot as plt
import matplotlib

# font = {'size': 14}
font = {'size': 18}

matplotlib.rc('font', **font)


def ode_model_plant(t, x, w, par):
    
    #Unpack states and parameters
    rho_0 = par["rho_0"]
    g = par["g"]
    k = par["k"]
    
    #Allocate space and write the model
    x_dot = np.zeros(x.shape)
    x_dot[0] = x[1] + w[0]
    x_dot[1] = rho_0 * np.exp(-x[0] / k) * np.square(x[1]) * x[2]/ 2 - g + w[1]
    x_dot[2] = w[2] #it is equal to w[2]
    return x_dot

def hx(x, par):
    
    # y2_1 = (par["Tb"] + (x[0] - par["hb"])*par["Lb"])/par["Tb"]
    # y2_1 = np.max([y2_1, 1e-10])
    # y2_2 = y2_1**par["exp"]
    # y2_3 = par["Pb"]*y2_2
    
    # y = np.array([np.sqrt(np.square(par["M"]) + np.square(x[0] - par["a"])),
    #               y2_3
    #               ])
    y = np.array([np.sqrt(np.square(par["M"]) + np.square(x[0] - par["a"])),
                  par["Pb"]*(((par["Tb"] + (x[0] - par["hb"])*par["Lb"])/par["Tb"])**par["exp"])
                  ])
    # print((par["Tb"] + (x[0] - par["hb"])*par["Lb"])/par["Tb"])
    return y



def fx_ukf_ode(ode_model, t_span, x0, args_ode = None, args_solver = {}):
    res = scipy.integrate.solve_ivp(ode_model,
                                    t_span,
                                    x0,
                                    args = args_ode,
                                    **args_solver)
    x_all = res.y
    x_final = x_all[:, -1]
    return x_final

def get_literature_values():
    #starting point
    x0 = np.array([300e3, #[ft]
                   -20e3, #[ft/s]
                   1e-3  # [ft3/(lb-s2)] (should be, from UoM-check)
                   ])
    
    #Initial covariance matrix for the UKF
    P0 = np.diag([3e8,#[ft^2], altitute, initial covariance matrix for UKF
                4e6, # [ft^2], horizontal range
                1e-6 # [?] ballistic coefficient
                ])
    #example 14.2 in "Optimal state estimation" by Dan Simon
    P0 = np.diag([1e6,#[ft^2], altitute, initial covariance matrix for UKF
                4e6, # [ft^2], horizontal range
                200 # [?] ballistic coefficient
                ])
    
    P0 = np.diag([1e6,#[ft^2], altitute, initial covariance matrix for UKF
                4e6, # [ft^2], horizontal range
                1e-3 # [?] ballistic coefficient
                ])
    P0 = np.diag([1e4,#[ft^2], altitute, initial covariance matrix for UKF
                4e4, # [ft^2], horizontal range
                1e-3 # [?] ballistic coefficient
                ])
    
    # P0=np.array([[ 9.29030400e+08,  5.39003846e+05, -8.07304597e-01],
    #               [ 5.39003846e+05,  4.64515200e+02, -8.92873834e-04],
    #               [-8.07304597e-01, -8.92873834e-04,  3.89725026e-09]])
    # matrixSize = 3 
    # A = np.random.rand(matrixSize, matrixSize)
    # cov0 = np.dot(A, A.transpose())
    
    
    # P0 = sklearn.datasets.make_spd_matrix(3)
    std_dev0 = np.sqrt(np.diag(P0))
    std_dev0 = np.sqrt(np.diag(P0))
    s0_inv = np.diag([1/si for si in std_dev0])
    corr_0 = s0_inv @ P0 @ s0_inv
    # s0 = np.sqrt(np.diag(cov0))
    # del cov0, s0, s0_inv
    
    
    # corr_0 = np.array([[1., .999, .3],
    #                     [.999, 1., .2],
    #                     [.3, .2, 1.]])
    # std_dev0 = np.sqrt(np.array([5e6, 5e-3, 1e-10]))
    
    
    #Nominal parameter values
    par_mean_fx = {"rho_0": 2., # [lb-sec^2/ft4]
                "k": 20e3, # [ft]
                "g": 32.2 # [ft/s2]
                }
    
    
    #Note: for barometric parameters in the measurement equation, see https://en.wikipedia.org/wiki/Barometric_formula for the barometric tables. Chosen 71 0000 m as the reference level (everything with "b" afterwards)
    par_mean_hx = {
        "M": 100e3, # [ft]
        "a": 100e3, # [ft]
        "g0": 9.80665, #[ m/s2] updated measurent model. Additional parameters
        "Mw": 0.0289644, # [kg/mol] - molar mass
        "R": 8.3144598,  # [J/(mol·K)] - gas constant
        "hb": 70e3,  # [m] - height of reference level (b is reference value)
        "Tb": 214.65,  # [K] - reference temperature
        "Pb": 3.96,  # [Pa] - reference pressure - can be changed to bar
        "Lb": -0.002,  # [K/m] - temperature lapse rate
        }
    par_mean_hx["exp"] = -par_mean_hx["g0"]*par_mean_hx["Mw"]/(par_mean_hx["R"]*par_mean_hx["Lb"]) # [-] exponent in the barometric equation
    
    

    #Kalman filter values in the description
    # Q = np.diag([0., 0., 0.]) #as in Dan Simon's exercise text
    #Can try different Q-values
    # Q = np.diag([1e3, 1e3, 1e-8]) 
    # Q = np.diag([1e3, 1e3, 1e-8]) #used 
    # Q = np.diag([1e-8, 1e-8, 1e-8]) 
    # Q = np.eye(3)*1e-6
    Q = np.diag([1e-4, 1e-3, 0])
    Q = np.diag([1e3, 1e3, 1e-6])
    # Q = np.diag([1e2, 1e2, 1e-7])
    # Q = np.eye(3)*0.
    
    #Measurement noise
    R = np.diag([10e3, #[ft^2]
                 5e1]) #[Pa**2]
    
    #convert to SI units
    kg_per_lbs = 0.45359237
    m_per_ft = 0.3048
    
    #and make scaling worse
    # bar_per_pa = 1e-5
    
    x0[0] *= m_per_ft
    x0[1] *= m_per_ft
    x0[2] *= (m_per_ft**3)/kg_per_lbs
    
    std_dev0[0] *= (m_per_ft)
    std_dev0[1] *= (m_per_ft)
    std_dev0[2] *= ((m_per_ft**3)/kg_per_lbs)
    
    par_mean_fx["rho_0"] *=kg_per_lbs/(m_per_ft**4)
    par_mean_fx["k"] *= m_per_ft
    par_mean_fx["g"] *= m_per_ft
    par_mean_hx["M"] *= m_per_ft
    par_mean_hx["a"] *= m_per_ft
    
    # par_mean_hx["Pb"] *= bar_per_pa
    
    R[0,0] *= m_per_ft**2
    # R[1,1] *= bar_per_pa**2
    Q[0,0] *= m_per_ft**2 
    Q[1,1] *= m_per_ft**2 
    Q[2,2] *= ((m_per_ft**3)/kg_per_lbs)**2 
    
    x0 = np.array([91e3, -6.1e3, 6.24e-5])
    std_dev0 = np.diag(std_dev0)
    std_dev0 = np.diag([1e4, 1e3, 1e-5])
    P0 = std_dev0 @ corr_0 @ std_dev0
    P0 = .5*(P0 + P0.T)
    
    #values which look nicer in SI-units
    Q = np.diag([1e2, 1e2, 1e-8])
    R[0,0]=1e3
    
    return x0, P0, par_mean_fx, par_mean_hx, Q, R

def compute_performance_index_valappil(x_kf, x_ol, x_true, RMSE = True):
    J_RMSE = np.linalg.norm(x_kf - x_true, axis = 1, ord = 2)
    if RMSE:
        return J_RMSE
    else:
        J_valappil = np.divide(J_RMSE,
                               np.linalg.norm(x_ol - x_true, axis = 1, ord = 2))
    return J_valappil


    
    
    
    
    
    