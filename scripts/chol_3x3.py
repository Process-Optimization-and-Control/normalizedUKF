# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 14:39:43 2022

@author: halvorak
"""

import scipy.linalg
import numpy as np



sig1 = 1e2
sig2 = 1e-2
sig3 = 1e-1

a = 1e-1
b = 0#1e-1
c = 1e-1

# [a, b, c] = [0, 0.9, 0]

dim_x = 3

std_dev = np.diag([sig1, sig2, sig3])
# std_dev_inv = np.diag([1/si for si in np.diag(std_dev)])
corr = np.array([[1, a, c],
                 [a, 1, b],
                 [c, b, 1]])

P0 = std_dev @ corr @ std_dev

#standard
L_P0=scipy.linalg.cholesky(P0, lower = True)
# L_P0=np.linalg.cholesky(P0)

#sigma-rho
L_corr=scipy.linalg.cholesky(corr, lower = True)
# L_corr=np.linalg.cholesky(corr)
L_P02 = std_dev @ L_corr

# L_own = cholesky_2x2(P0)
# L_corr_own = cholesky_2x2(corr)
# L_own2 = std_dev @ L_corr_own


print(f"L_33, sigma_rho: {L_P02[-1,-1]}\n",
      f"L_33, std: {L_P0[-1,-1]}\n"
      )
print(L_P0 == L_P02)



# L_corr2 = std_dev_inv@L_P0



