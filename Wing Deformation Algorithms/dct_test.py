#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 23:29:41 2020

@author: davidy
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
import pdb

N = 101
n = np.arange(0,N,1)
theta = np.pi*(2*n+1)/((2*(N)))
x = np.cos(theta)	

f = (2+x)*np.sqrt(1-x**2)-(1+2*x)*np.arccos(x)
#f = np.sqrt((1-x)/(1+x))
#f = np.sin(x)
b = fct(f)
eta_trial = np.transpose(np.array(x,ndmin = 2))
alpha = 1
beta = 1
##Apply algorithm 1 to find the second antiderivative of f (in spectral space)
B = algorithm1(b)

##Transform to physical space
pn = chebval(x,B)
#pn = ifct(B,N)

pn = np.transpose(pn)
bp = cheb_diff(b)


fp = chebval(x,bp)
#fp = ifct(bp)
fp = np.transpose(fp)
#plt.plot(x,fp,'.g')
#plt.plot(x,np.exp(x))
#plt.plot(x,np.subtract(pn,f)[99])


eta_singular  = algorithm1(pn)
plt.plot(x,pn)
psi = algorithm2(eta_trial,eta_singular,alpha,beta)
