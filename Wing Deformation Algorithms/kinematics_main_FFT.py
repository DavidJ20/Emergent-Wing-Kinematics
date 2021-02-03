# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 23:04:26 2021

@author: Dave
"""

import numpy as np
from fwing_FFT import fct, ifct, L_eta_func
from scipy.sparse.linalg import gmres,LinearOperator
import matplotlib.pyplot as plt
from scipy.fft import dct, idct, dst, idst, fftn, ifftn,fftfreq,fftshift
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
t = np.linspace(0,10,N)
eta_le = 0.1
eta_le_p = 0
tol = 1e-12

#%%
#Now we want to use algorithm3 to evaluate the nonlocal operator script(L) using GMRES to find eta
##NEW STUFF
w_trial = np.zeros
w_hat_trial = fftn(w_trial,axis = 0) #expanding the fourier series in time along the rows of the matrix for each value x
w_hat_trial = w_hat_trial[:,x_val] #working only on one specific x-value along the chord. Then we iterate this function for each
Lambda = (eta_le + eta_le_p*(x+1))
Lambda = fct(Lambda)
Lambda[Lambda < tol] = 0
L = LinearOperator((N,N), matvec = L_eta_func)
counter = gmres_counter()

w_hat = gmres(L,Lambda,tol = tol ,maxiter = 100,x0 = fct(x), callback = counter)
#%%
plt.plot(x,ifct(eta[0].real))
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
plt.setp(axs,xlim = (-1.2,1.2), ylim = (-1,1))
for t in [0,1/8,1/4,3/8]:
    axs[i].plot(x,ifct(eta[0].real)*np.cos(2*np.pi*t) - ifct(eta[0].imag)*np.sin(2*np.pi*t)) 
    axs[i].plot(x,x*0,'--g')
    axs[i].set_title('t = ' + str(t) )
    i += 1
#%%
# xx = np.linspace(-1,1,N-1)
# yy = np.linspace(-1,1,N-1)

# [x,y] = np.meshgrid(xx,yy)

# ksi = x + 1j*y

# z = 1/2 * (ksi + 1/ksi)