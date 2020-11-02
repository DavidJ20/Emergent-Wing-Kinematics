# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 19:06:59 2020

@author: Dave
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct as dct
from scipy.fftpack import idct as idct
from numpy.polynomial.chebyshev import chebval as chebval
from scipy.special import eval_chebyt as chebyt
from fwing import fct as fct
from fwing import algorithm1 as algorithm1
from fwing import ifct as ifct
from fwing import cheb_diff as cheb_diff
from fwing import algorithm2 as algorithm2
from fwing import ifst as ifst
from fwing import fst as fst
from fwing import algorithm3 as algorithm3
import pdb

#%%
#Define the needed parameters and discrete domain (-1,1)
N = 101
n = np.arange(0,N,1)
theta = np.pi*(2*n+1)/((2*(N)))
x = np.cos(theta)	

eta_singular = (2+x)*np.sqrt(1-x**2)-(1+2*x)*np.arccos(x)

eta_trial = x #define a trial function for eta
b = np.transpose(np.array(fct(eta_trial),ndmin = 2)) #find the chebyshev coefficients of the trial function
alpha = 1
beta = 1
sigma = 1
S = 1
eta_le = 1
eta_le_p = 1
tol = 0.001
R = 1
#%%
#Now we want to use algorithm2 to evaluate the nonlocal operator script(L)

eta = algorithm3(S, R, sigma, eta_le, eta_le_p, tol, x, eta_singular, alpha, beta, eta_trial)