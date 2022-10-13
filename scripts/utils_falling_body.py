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
    # M = par["M"]
    g = par["g"]
    # a = par["a"]
    k = par["k"]
    

    #Allocate space and write the model
    x_dot = np.zeros(x.shape)
    x_dot[0] = x[1] + w[0]
    x_dot[1] = rho_0 * np.exp(-x[0] / k) * np.square(x[1]) * x[2]/ 2 - g + w[1]
    x_dot[2] = w[2] #it is equal to w[2]
    return x_dot

def hx(x, par):
    
    y2_1 = (par["Tb"] + (x[0] - par["hb"])*par["Lb"])/par["Tb"]
    y2_1 = np.max([y2_1, 1e-10])
    y2_2 = y2_1**par["exp"]
    y2_3 = par["Pb"]*y2_2
    
    y = np.array([np.sqrt(np.square(par["M"]) + np.square(x[0] - par["a"])),
                  y2_3
                  ])
    # y = np.array([np.sqrt(np.square(par["M"]) + np.square(x[0] - par["a"])),
    #               par["Pb"]*(((par["Tb"] + (x[0] - par["hb"])*par["Lb"])/par["Tb"])**par["exp"])
    #               ])
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
    
    std_dev0 = np.diag(std_dev0)
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

def get_param_ukf_case1(par_fx = True, std_dev_prct = 0.05, plot_dist = True):
    """
    Generates gamma distributions for the parameters. Mean of gamma dist = par_mean from get_literature_values, and standard deviation of gamma dist = mean_literature*std_dev_prct

    Parameters
    ----------
    std_dev_prct : TYPE, optional float
        DESCRIPTION. The default is 0.05. Percentage of standard deviation compared to mean value of parameter
    plot_dist : TYPE, optional bool
        DESCRIPTION. The default is True. Whether or not to plot the distributions

    Returns
    -------
    par_dist : TYPE, dict
        DESCRIPTION. Each key cntains a gamma distribution, with mean = literature mean and std_dev = literature standard dev
    par_det : TYPE, dict
        DESCRIPTION. Key contains deterministic parameters.

    """
    par_dist = {}
    par_det = {}
    
    if par_fx: #want parameters for state equations
        par_mean, par_hx, Q, R = get_literature_values()
    else: #want parameters for measurement equation
        par_func, par_mean, Q, R = get_literature_values()
    
    
    for (key, val) in par_mean.items():
        if key == "g":
            continue
        alpha, loc, beta = get_param_gamma_dist(val, val*std_dev_prct, num_std = 2)
        # print(f"{key}: {alpha}, {loc}, {beta}")
        par_dist[key] = scipy.stats.gamma(alpha, loc = loc, scale = 1/beta)
        
    if "g" in par_mean.keys():   
        par_det["g"] = par_mean["g"]
    
    if plot_dist:
        # par_dist.pop("k")
        dim_dist = len(par_dist)
        fig, ax = plt.subplots(dim_dist, 1)
        if dim_dist == 1:
            ax = [ax]
        i = 0
        
        for key, dist in par_dist.items():
            #compute the mode numerically, as dist.mode() is not existing
            mode = scipy.optimize.minimize(lambda x: -dist.pdf(x),
                                           dist.mean(),
                                           tol = 1e-10)
            mode = mode.x
            # print(key, mode, dist.mean())
            x = np.linspace(dist.ppf(1e-5), dist.ppf(.999), 100)
            ax[i].plot(x, dist.pdf(x), label = "pdf")
            # ax[i].set_xlabel("Air density at sea level, " + r"$\rho_0$" + r" ($lb-s^2/ft^4$)")
            ax[i].set_xlabel(key)
            ax[i].set_ylabel("pdf")
            ylim = ax[i].get_ylim()
            
            # ax[i].plot([par_mean[key], par_mean[key]], [ylim[0], ylim[1]], 
            #         label = "nom")
            # ax[i].plot([dist.mean(), dist.mean()], [ylim[0], ylim[1]], 
            #         linestyle = "dashed", label = "Mean")
            # ax[i].plot([dist.mean()*(1-std_dev_prct), dist.mean()*(1-std_dev_prct)], [ylim[0], ylim[1]], 
            #         label = "Mean-std_lit")
            # ax[i].plot([dist.mean() - dist.std(), dist.mean() - dist.std()], [ylim[0], ylim[1]], 
            #         linestyle = "dashed", label = "Mean-std_gamma")
            # ax[i].plot([mode, mode], [ylim[0], ylim[1]], label = "Mode")
            
            
            ax[i].scatter(par_mean[key], dist.pdf(par_mean[key]), label = r"$\mu_\Gamma = \mu_{lit} = \theta_{UKF}$")
            ax[i].scatter([par_mean[key] - dist.std(), par_mean[key] + dist.std()], 
                              [dist.pdf(par_mean[key] - dist.std()), dist.pdf(par_mean[key] + dist.std())], label = r"$\mu_\Gamma \pm \sigma_\Gamma = \mu_{lit} \pm \sigma_{lit}$")
            ax[i].scatter(mode, dist.pdf(mode), label = r"$\theta_{true}$")
            ndist = 2
            ax[i].scatter(dist.mean() - ndist*dist.std(), 
                          dist.pdf(dist.mean() - ndist*dist.std()), 
                          label = r"$\mu_{lit} - 2\sigma_{lit}$")
            
            # ax[i].plot([dist.mode(), dist.mode()], [ylim[0], ylim[1]], 
            #         linestyle = "dashed", label = "Mean")
            ax[i].set_ylim(ylim)
            ax[i].legend(frameon = False)
            i += 1
            
        return par_dist, par_det, fig, ax
    else:
        fig, ax = None, None
        return par_dist, par_det, fig, ax
  

def get_param_gamma_dist(mean, std_dev, num_std = 3):
    
    ##For making (mean_gamma = mean) AND (var_gamma = std_dev**2)
    loc = mean - std_dev*num_std
    alpha = num_std**2
    beta = num_std/std_dev
    
    #For putting (mode= mean-std_dev) AND 
    # loc = mean - std_dev*num_std
    # beta = 1/std_dev
    # # alpha = num_std#*std_dev**2
    # alpha = beta**2*std_dev**2#*std_dev**2
    return alpha, loc, beta

def get_sigmapoints_and_weights(par_dist):
    """
    Returns sigma points and weights for the distributions in the container par_dist.

    Parameters
    ----------
    par_dist : TYPE list, dict, a container which is iterable by for loop. len(par_dist) = n
        DESCRIPTION. each element contains a scipy.dist

    Returns
    -------
    sigmas : TYPE np.array((n, (2n+1)))
        DESCRIPTION. (2n+1) sigma points
    w : TYPE np.array(2n+1,)
        DESCRIPTION. Weight for every sigma point

    """
    
    n = len(par_dist) #dimension of parameters
    
    # Compute the required statistics
    mean = np.array([dist.mean() for k, dist in par_dist.items()])
    var = np.diag([dist.var() for k, dist in par_dist.items()])
    cm3 = np.array([scipy.stats.moment(dist.rvs(size = int(1e6)), moment = 3) 
                    for k, dist in par_dist.items()]) #3rd central moment
    cm4 = np.array([scipy.stats.moment(dist.rvs(size = int(1e6)), moment = 4) 
                    for k, dist in par_dist.items()]) #4th central moment
    
    # print(f"mean: {mean}\n",
    #       f"var: {var}\n",
    #       f"cm3: {cm3}\n",
    #       f"cm4: {cm4}")
    
    #Generate sigma points
    sigma_points = spc.GenUTSigmaPoints(n) #initialize the class
    s, w = sigma_points.compute_scaling_and_weights(var,  #generate scaling and weights
                                                    cm3, 
                                                    cm4)
    sigmas, P_sqrt = sigma_points.compute_sigma_points(mean, #sigma points and P_sqrt
                                                        var, 
                                                        s)
    return sigmas, w

def get_P0_Tuveri():
    P0 = np.array([[.1054, -.11, -.0846, .057, -.0898],
                  [-.11, .1537, .0508, -.070, .0489],
                  [-.0846, .0508, .1282, -.0267, .0813],
                  [.057, -.070, -.0267, .0491, -.0657],
                  [-.0898, .0489, .0813, -.0657, .202]])
    P0 *=1e-10
    return P0
    
    
    
    
    
    
    
    
    
    
    