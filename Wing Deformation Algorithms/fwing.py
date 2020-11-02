#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 02:14:26 2020
Library used to solve for the emergent wing kinematics in accordance with Moore 2017
@author: David Yudin
"""
#%%
import numpy as np
from scipy.fft import dct as dct
from scipy.fft import idct as idct
from scipy.fft import dst as dst
from scipy.fft import idst as idst
import scipy.special
from scipy.sparse.linalg import gmres, aslinearoperator
import pdb
#%%
def fct(f):
# =============================================================================
#     INPUT: Function evauluated at callocation points
#     OUTPUT: Chebyshev coefficients  
#==============================================================================
    ##Calculate chebyshev coefficients with DCT
    b = dct(f)/(len(f))#*np.sqrt(1/(2*len(f)))
    ##Correct to coincide with Moore 2017
    b[0] = b[0]/2#*np.sqrt(2)
    #b[1:-1] = b[1:-1]*np.sqrt(2/(N))

    
    return b

#%%
def ifct(f):
# =============================================================================
#     INPUT: Chebyshev coefficients
#     OUTPUT: Inverse discrete cosine tranform of f (f is in spectral space). This function takes f from spectral space to physical space
#==============================================================================
    ##Calculate chebyshev coefficients with DCT
    F = dct(f,type = 3)
    ##Correct to coincide with Moore 2017
    F[0] = F[0]#*2#*np.sqrt(2)
    #b[1:-1] = b[1:-1]*np.sqrt(2/(N))
    
    return F
#%%
def fst(f):
# =============================================================================
#     INPUT: Function evauluated at callocation points
#     OUTPUT: Chebyshev coefficients  
#==============================================================================
    ##Calculate chebyshev coefficients with DCT
    a = dst(f)/len(f)
    ##Correct to coincide with Moore 2017
    a[0] = a[0]/2
    #b[1:-1] = b[1:-1]*np.sqrt(2/(N))
    return a
    #%%
def ifst(a):
# =============================================================================
#     INPUT: Function evauluated at callocation points
#     OUTPUT: Chebyshev coefficients  
#==============================================================================
    ##Calculate chebyshev coefficients with DCT
    f= idst(a)/len(a)
    ##Correct to coincide with Moore 2017
    #b[0] = b[0]/2
    #b[1:-1] = b[1:-1]*np.sqrt(2/(N))
    return f
#%%
def Theodorsen(sigma):
# =============================================================================
#     INPUT: reduced frequency sigma
#     OUTPUT: Theodorsen function evaluated at sigma
#==============================================================================
    C_sigma = scipy.special.kv(1,1j*sigma)/(scipy.special.kv(0,1j*sigma)+(scipy.special.kv(1,1j*sigma)))
    return C_sigma
#%%
def cheb_diff(b):
# =============================================================================
#     INPUT: N chebyshev coefficients of b
#     OUTPUT: The derivative of b (bp=b') in spectral space
#==============================================================================
	bp = np.zeros((len(b)+1,1))
	#bp[N] = 0; bp[N-3] = 0
	for k in np.arange(len(b)-2,-1,-1):
		bp[k] = bp[k+2] + 2*(k+1)*b[k+1]
		
        
	bp[0] = bp[0]/2 
	bp = np.delete(bp,len(bp)-1,0)
	return bp

#%%
def algorithm1(b):
# =============================================================================
#     INPUT: N+1 chebyshev coefficients of g
#     OUTPUT: The second antiderivative of f (in spectral space)
#==============================================================================
    N = len(b)
    #Calculate the first antiderivative
    B = np.zeros((N,1))
    B[0] = b[0]/4
    B[1] = b[0] - b[2]/2
    
    for k in np.arange(2,N-1,1):
        B[k] = (1/(2*k))*(b[k-1]-b[k+1])
        
    ##Enforce boundary conditions
        
    ##x = 1
    D_1g = 0
    for k in np.arange(1,N,1):
        D_1g = D_1g + (-1)**k * B[k]
        
    D_1g = D_1g +B[0]/2
    

    B[0] = B[0] - D_1g #- D_m1g
        
    f = np.zeros((N,1))

    f[0] = B[0]/4
    f[1] = B[0] - B[2]/2
    
    ##Calculate the second antiderivative
    for k in np.arange(2,N-2,1):
        f[k] = (1/(2*k))*(B[k-1]-B[k+1])
        #f[k] = (1/(2*k))*((1/(2*(k+1))*(b[k]-b[k+2])) - (1/(2*(k-1)))*(b[k-2]-b[k]))
        #f[k] = (b[k]-b[k+2])/(4*k*(k+1)) - (b[k-2]-b[k])/(4*k*(k-1))
        
    ##Enforce boundary conditions
        
    ##x = -1
    D_m2g = 0
    for k in np.arange(1,N,1):
        D_m2g = D_m2g + (-1)**k * f[k]
    D_m2g =  D_m2g + f[0]/2
    
    f[0] = f[0] - D_m2g
     
    print('D_1g:',D_1g)
    print('D_m2g:',D_m2g)
    #pdb.set_trace()

    return f
#%%
def algorithm2(eta_trial,eta_singular,alpha,beta):
# =============================================================================
#     INPUT: N+1 chebyshev coefficients of the trial kinematics (eta_trial) and eta_singular, and alpha and beta
#     OUTPUT: The operator L applied to the trial kinematics (eta_trial)
#==============================================================================
    U = 1 # Nondimensional freestream velocity
    sigma = 1
    eta_trial = np.transpose(np.array(fct(eta_trial),ndmin = 2))
    aa = 4*np.pi**2 * eta_trial
    bb = - 4*np.pi*1j*U*cheb_diff(eta_trial)
    cc = - U**2 * cheb_diff(cheb_diff(eta_trial))
    DPsi = aa + bb + cc
    Psi = np.zeros((len(DPsi),1)) + 1j*np.zeros((len(DPsi),1)) 
    
    Psi[0] = DPsi[0]/4
    Psi[1] = DPsi[0] - DPsi[2]/2
    
    #Integrate DPsi once to solve for a_k, k = 1,...,inf. We can't find a_0 yet (constant of integration)
    for k in np.arange(2,len(DPsi)-1,1):
        Psi[k] = (1/(2*k))*(DPsi[k-1]-DPsi[k+1])
        
    a_k = Psi #get the coefficients a_k
    
    Q_r = ifst(a_k) #Compute the regular part of the hydrodynamic load with the ifst from a_k's
    eta_trial = ifct(eta_trial) #transform back to physical space from spectral space
    
    beta_eta_Q_r = beta*eta_trial + Q_r
    
    
    P_m1_beta_eta_Q_r = algorithm1(fct(beta_eta_Q_r)) #Transform back to spectral space and apply algorithm1
    
    eta_trial = fct(eta_trial) #transform back to spectral space
    V = 2*np.pi*1j * eta_trial + U * cheb_diff(eta_trial) #compute V(x) from eq(29)
    V_hat = fct(V) #transform to spectral space
    
    
    a_0 = -U*Theodorsen(sigma)*(V_hat[0]+V_hat[1]) + U * V_hat[1]
    eta_s = np.transpose(np.array(fct(eta_singular),ndmin = 2))
    L = eta_trial - a_0*eta_s - P_m1_beta_eta_Q_r #build the final operator L
    
    return L
#%%
def algorithm3(S,R,sigma,eta_le,eta_le_p,tol,x,eta_singular,alpha,beta,eta_trial):
    
    Lambda = eta_le + eta_le_p*(x+1)
    
    L = aslinearoperator(algorithm2(eta_trial,eta_singular,alpha,beta))
    eta = gmres(aslinearoperator(algorithm2(eta_trial,eta_singular,alpha,beta)),Lambda,tol)
    
    return eta