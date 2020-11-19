# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 19:06:59 2020

@author: Dave
9"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct as dct
from scipy.fftpack import idct as idct
from fwing import fct as fct
from fwing import algorithm1, ifct, cheb_diff, algorithm2, ifst, fst, algorithm3, L_eta_func
from scipy.sparse.linalg import gmres,LinearOperator



#%%
#Define the needed parameters and discrete domain (-1,1)
N = 1001
n = np.arange(0,N,1)
theta = np.pi*(2*n+1)/((2*(N)))
x = np.cos(theta)
alpha = 200
DDeta_singular = (1/(2*alpha))*((2+x)*np.sqrt(1-x**2)-(1+2*x)*np.arccos(x))
eta_singular = algorithm1(fct(DDeta_singular),-1)
eta_trial = x #define a trial function for eta
b = fct(eta_trial) #find the chebyshev coefficients of the trial function
eta_le = 1
eta_le_p = 0
tol = np.ones((N,1))*1e-12

#%%
#Now we want to use algorithm3 to evaluate the nonlocal operator script(L) using GMRES to find eta

Lambda = (eta_le + eta_le_p*x + eta_le_p*x/x)
Lambda = fct(Lambda)
L = LinearOperator((N,N), matvec = L_eta_func)

eta = gmres(L,Lambda,tol)
#%%
plt.plot(x,ifct(eta[0].imag))
#plt.ylim(0.9,1.2)
#Lambda = ((eta_le + eta_le_p*(x+1)).reshape(len(x),))
#plt.plot(x,Lambda)

#%%

h = ifct(eta[0].real)*np.cos(2*np.pi*t) - (eta[0].imag)*np.sin(np.pi*t)

#%%
# i = 0
# for t in np.linspace(0,1,100):
#     plt.plot(x,ifct(eta[0].real)*np.cos(2*np.pi*t) - ifct(eta[0].imag)*np.sin(2*np.pi*t))
#     plt.xlim()
#     plt.ylim(-3,6)
    
#     plt.savefig(str(i) +'.png')
#     plt.close()
#     i = i + 1
#     #plt.plot(x,np.exp(2*np.pi*t*1j)*ifct(eta[0]))

#%%
for t in [0,1/8,2/8,3/8]:
    plt.plot(x,ifct(eta[0].real)*np.cos(2*np.pi*t) - ifct(eta[0].imag)*np.sin(2*np.pi*t))    
    