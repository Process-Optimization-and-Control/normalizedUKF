# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 13:47:07 2021

@author: halvorak
"""
import numpy as np
import scipy.stats


def ode_model_plant(t, x, w, par):
    """
    Model of the system. Gets current state values, returns dx/dt

    Parameters
    ----------
    t : TYPE float
        DESCRIPTION. Time
    x : TYPE np.array(dim_x,)
        DESCRIPTION. Current state value
    w : TYPE np.array(dim_w,)
        DESCRIPTION. Noise realization
    par : TYPE dict
        DESCRIPTION. Parameters

    Returns
    -------
    x_dot : TYPE np.array(dim_x,)
        DESCRIPTION. Derivative of the states

    """
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
    """
    Measurement model

    Parameters
    ----------
    x : TYPE np.array(dim_x,)
        DESCRIPTION. Current state value
    par : TYPE dict
        DESCRIPTION. Parameters

    Returns
    -------
    y : TYPE np.array(dim_y,)
        DESCRIPTION. Measurement (without measurement noise)

    """
   
    y = np.array([np.sqrt(np.square(par["M"]) + np.square(x[0] - par["a"])),
                  par["Pb"]*(((par["Tb"] + (x[0] - par["hb"])*par["Lb"])/par["Tb"])**par["exp"])
                  ])
    return y



def fx_ukf_ode(ode_model, t_span, x0, args_ode = None, args_solver = {}):
    """
    Solve x_{k+1}=f(x_{k}) for the provided model

    Parameters
    ----------
    ode_model : TYPE function
        DESCRIPTION. Derivative, dx/dt
    t_span : TYPE tuple
        DESCRIPTION. Integration time, (t_start, t_end)
    x0 : TYPE np.array(dim_x,)
        DESCRIPTION. Initial value for integrator at t_start.
    args_ode : TYPE, optional dict
        DESCRIPTION. The default is None. Optional parameters for the ode_model
    args_solver : TYPE, optional dict
        DESCRIPTION. The default is {}. Optional parameters for the ode-solver scipy.integrate.solve_ivp

    Returns
    -------
    x_final : TYPE np.array(dim_x,)
        DESCRIPTION. Integrated state values at t_end

    """
    res = scipy.integrate.solve_ivp(ode_model,
                                    t_span,
                                    x0,
                                    args = args_ode,
                                    **args_solver)
    x_all = res.y
    x_final = x_all[:, -1]
    return x_final

def get_literature_values():
    """
    Initial values, parameters etc. Made here for making main script cleaner.

    Returns
    -------
    x0 : TYPE np.array(dim_x,)
        DESCRIPTION. Starting point for UKF (mean value of initial guess)
    P0 : TYPE np.array((dim_x, dim_x))
        DESCRIPTION. Initial covariance matrix. Gives uncertainty of starting point. The starting point for the true system is drawn as x_true ~ N(x0,P0)
    par_mean_fx : TYPE dict
        DESCRIPTION. Parameters for the process model.
    par_mean_hx : TYPE dixt
        DESCRIPTION. Parameters for the measurement model.
    Q : TYPE np.array((dim_w, dim_w))
        DESCRIPTION. Process noise covariance matrix
    R : TYPE np.array((dim_v, dim_v))
        DESCRIPTION. Measurement noise covariance matrix

    """
    
    #Nominal parameter values (values are somewhat similar to example 13.2 in Dan Simon's book Optimal State Estimation)
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
    
    #convert to SI units
    kg_per_lbs = 0.45359237
    m_per_ft = 0.3048
    
    par_mean_fx["rho_0"] *=kg_per_lbs/(m_per_ft**4)
    par_mean_fx["k"] *= m_per_ft
    par_mean_fx["g"] *= m_per_ft
    par_mean_hx["M"] *= m_per_ft
    par_mean_hx["a"] *= m_per_ft
    
    #Initial state and uncertainty
    x0 = np.array([91e3, -6.1e3, 6.24e-5])
    std_dev0 = np.diag([1e4, 1e3, 1e-5])
    corr_0 = np.eye(x0.shape[0])
    P0 = std_dev0 @ corr_0 @ std_dev0
    P0 = .5*(P0 + P0.T)
    
    #Process and measurement noise
    Q = np.diag([1e2, 1e2, 1e-8])
    R = np.diag([1e3, 50])
    
    return x0, P0, par_mean_fx, par_mean_hx, Q, R

def compute_performance_index_valappil(x_kf, x_ol, x_true, RMSE = True):
    """
    Compute error in state estimates. Either root mean square (RMSE) or the index by Valappil, which is RMSE(state estimate)/RMSE(open loop simulation). Valappil's cost function says how much better the estimator is compared to a simple model prediction.

    Parameters
    ----------
    x_kf : TYPE np.array((dim_x, dim_t))
        DESCRIPTION. State estimates
    x_ol : TYPE np.array((dim_x, dim_t))
        DESCRIPTION. Open loop/model prediction
    x_true : TYPE np.array((dim_x, dim_t))
        DESCRIPTION. True state values
    RMSE : TYPE, optional bool
        DESCRIPTION. The default is True. RMSE if True, if False Valappil's cost function is returned

    Returns
    -------
    TYPE np.array(dim_x,)
        DESCRIPTION. Value of cost function for each state

    """
    J_RMSE = np.linalg.norm(x_kf - x_true, axis = 1, ord = 2)
    if RMSE:
        return J_RMSE
    else:
        J_valappil = np.divide(J_RMSE,
                               np.linalg.norm(x_ol - x_true, axis = 1, ord = 2))
    return J_valappil


    
    