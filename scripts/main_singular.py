# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 14:39:43 2022

@author: halvorak
"""

import scipy.linalg
import numpy as np


def cholesky_2x2(P):
    assert (2,2) == P.shape
    L_00 = np.sqrt(P[0,0])
    L_10 = P[1,0]/L_00
    L_11 = np.sqrt(P[1,1] - L_10**2)
    L = np.zeros(P.shape)
    L[0,0] = L_00
    L[1,0] = L_10
    L[1,1] = L_11
    return L

sig1 = 1e-2
sig2 = 1e-10
delta = 1e-7

sig1 = 1e-2
sig2 = 1e-10
delta = 1e-4

dim_x = 2

std_dev = np.diag([sig1, sig2])
# std_dev_inv = np.diag([1/si for si in np.diag(std_dev)])
corr = np.array([[1, delta],
                 [delta, 1]])

P0 = std_dev @ corr @ std_dev

#standard
L_P0=scipy.linalg.cholesky(P0, lower = True)
# L_P0=np.linalg.cholesky(P0)

#sigma-rho
L_corr=scipy.linalg.cholesky(corr, lower = True)
# L_corr=np.linalg.cholesky(corr)
L_P02 = std_dev @ L_corr

L_own = cholesky_2x2(P0)
L_corr_own = cholesky_2x2(corr)
L_own2 = std_dev @ L_corr_own


print(f"L_22, sigma_rho: {L_P02[-1,-1]}\n",
      f"L_22, std: {L_P0[-1,-1]}\n",
      f"L_22, own: {L_own[-1,-1]}\n",
      f"L_22, sr-own: {L_own2[-1,-1]}\n",
      )
print(L_P0 == L_P02)
print(L_P0 == L_own)
print(L_own2 == L_own)
print(L_own2 == L_P02)



# L_corr2 = std_dev_inv@L_P0



