# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 19:06:59 2020

@author: Dave
9"""


import numpy as np
from fwing import fct, ifct, L_eta_func
from scipy.sparse.linalg import gmres,LinearOperator
import matplotlib.pyplot as plt
#%%
class gmres_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))

#%%
#Define the needed parameters and discrete domain (-1,1)
N = 1001
n = np.arange(0,N,1)
theta = np.pi*(2*n+1)/((2*(N)))
x = np.cos(theta)

eta_le = 1
eta_le_p = 0
tol = 1e-12

#%%
#Now we want to use algorithm3 to evaluate the nonlocal operator script(L) using GMRES to find eta

Lambda = (eta_le + eta_le_p*(x+1))
Lambda = fct(Lambda)
Lambda[Lambda < tol] = 0
L = LinearOperator((N,N), matvec = L_eta_func)
counter = gmres_counter()

eta = gmres(L,Lambda,tol = tol ,maxiter = 100,x0 = fct(x), callback = counter)
#%%
plt.plot(x,ifct(2*eta[0].real))
#plt.ylim(0.9,1.2)
#Lambda = ((eta_le + eta_le_p*(x+1)).reshape(len(x),))
#plt.plot(x,Lambda)

#%%

# i = 0
# for t in np.linspace(0,1,100):
#     plt.plot(x,ifct(eta[0].real)*np.cos(2*np.pi*t) - ifct(eta[0].imag)*np.sin(2*np.pi*t))
#     plt.xlim()
#     plt.ylim(-5,5)
    
#     plt.savefig(str(i) +'.png')
#     plt.close()
#     i = i + 1
#     #plt.plot(x,np.exp(2*np.pi*t*1j)*ifct(eta[0]))

#%%
fig, axs = plt.subplots(1,4,figsize = (15,6))
i = 0
plt.setp(axs,xlim = (-1.2,1.2), ylim = (-5,5))
for t in [0,1/8,1/4,3/8]:
    axs[i].plot(x,ifct(eta[0].real)*np.cos(2*np.pi*t) - ifct(eta[0].imag)*np.sin(2*np.pi*t)) 
    axs[i].plot(x,x*0,'--g')
    i += 1
#%%
# xx = np.linspace(-1,1,N-1)
# yy = np.linspace(-1,1,N-1)

# [x,y] = np.meshgrid(xx,yy)

# ksi = x + 1j*y

# z = 1/2 * (ksi + 1/ksi)