# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 14:39:43 2022

@author: halvorak
"""

import scipy.linalg
import numpy as np

if True: #well-conditioned (change to False if you want to see the other case)
    sig1 = 1e2
    sig2 = 1e-2
    sig3 = 1e-1
else:
    #ill-conditioned
    sig1 = 1e7
    sig2 = 1e-7
    sig3 = 1e-1

std_dev = np.diag([sig1, sig2, sig3])

# make correlation matrix
a = 1e-1
b = 0
c = 1e-1
corr = np.array([[1, a, c],
                 [a, 1, b],
                 [c, b, 1]])

P0 = std_dev @ corr @ std_dev #covariance matrix

#standard approach
L_P0=scipy.linalg.cholesky(P0, lower = True)

#normalized version
L_corr=scipy.linalg.cholesky(corr, lower = True)
L_P02 = std_dev @ L_corr

print(f"L_33, sigma_rho: {L_P02[-1,-1]}\n",
      f"L_33, std: {L_P0[-1,-1]}\n"
      )
print(L_P0 == L_P02)


