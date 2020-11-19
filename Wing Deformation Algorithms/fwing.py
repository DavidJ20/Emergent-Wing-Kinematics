#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 02:14:26 2020
Library used to solve for the emergent wing kinematics in accordance with Moore 2017
@author: David Yudin
"""
#%% Libraries
import numpy as np
from scipy.fft import dct as dct
from scipy.fft import idct as idct
from scipy.fft import dst as dst
from scipy.fft import idst as idst
import scipy.special
from scipy.sparse.linalg import gmres,LinearOperator
from numpy.polynomial.chebyshev import chebval as chebval
from scipy.integrate import trapz

#%% fct
def fct(f):
# =============================================================================
#     INPUT: Function evauluated at callocation points
#     OUTPUT: Chebyshev coefficients  
#==============================================================================
    ##Calculate chebyshev coefficients with DCT
    b = dct(f)/(len(f))
    ##Correct to coincide with Moore 2017
    b[0] = b[0]/2
    return b


#%% ifct
def ifct(f):
# =============================================================================
#     INPUT: Chebyshev coefficients
#     OUTPUT: Inverse discrete cosine tranform of f (f is in spectral space). This function takes f from spectral space to physical space
#==============================================================================
    # ##Calculate chebyshev coefficients with DCT
    N = 1001
    n = np.arange(0,N,1)
    theta = np.pi*(2*n+1)/((2*(N)))
    x = np.cos(theta)
    #Correct to coincide with Moore 2017
    
    #f[0] = f[0]*2
    #F = idct(f)*len(f)
    F = chebval(x,f)
    print('i')
    return F

#%% fst
def fst(f):
# =============================================================================
#     INPUT: Function evauluated at callocation points
#     OUTPUT: Chebyshev coefficients  
#==============================================================================
    ##Calculate chebyshev coefficients with DCT
    a = dst(f)/len(f)
    ##Correct to coincide with Moore 2017
    a[0] = a[0]/2
    
    return a
#%% ifst
def ifst(a):
# =============================================================================
#     INPUT: Function evauluated at callocation points
#     OUTPUT: Chebyshev coefficients  
#==============================================================================
    ##Calculate chebyshev coefficients with DCT
    ##Correct to coincide with Moore 2017
    a[0] = a[0]*2
    f= idst(a)*len(a)

    return f

#%% Theodorsen
def Theodorsen(sigma):
# =============================================================================
#     INPUT: reduced frequency sigma
#     OUTPUT: Theodorsen function evaluated at sigma
#==============================================================================
    C_sigma = scipy.special.kv(1,1j*sigma)/(scipy.special.kv(0,1j*sigma)+(scipy.special.kv(1,1j*sigma)))
    return C_sigma

#%% cheb_diff
def cheb_diff(b):
# =============================================================================
#     INPUT: N chebyshev coefficients of b
#     OUTPUT: The derivative of b (bp=b') in spectral space
#==============================================================================
    bp = np.zeros((len(b)+1,1)) + 1j*np.zeros((len(b)+1,1))
	
    for k in np.arange(len(b)-2,-1,-1):
        bp[k] = bp[k+2] + 2*(k+1)*b[k+1]
		
    bp[0] = bp[0]/2
    bp = np.delete(bp,len(bp)-1,0)
    bp = bp.reshape(len(b),)

    
    return bp

#%% algorithm1
def algorithm1(b,x_end):
# =============================================================================
#     INPUT: N+1 chebyshev coefficients of b
#     OUTPUT: The second antiderivative of f (in spectral space)
#==============================================================================
    N = len(b)
    #Calculate the first antiderivative
    B = np.zeros((N,)) + 1j*np.zeros((N,))
    #B[0] = b[0]/2
    B[1] = b[0] - b[2]/2
    
    for k in np.arange(2,N-1,1):
        B[k] = (1/(2*k))*(b[k-1]-b[k+1])
    
    #B[0] = b[0]/2
    ##Enforce boundary conditions f(x_end) = 0
    
    D_1g = 0 +0*1j
    for k in np.arange(1,N,1):
        D_1g = D_1g + (x_end)**k * B[k]
        
    D_1g = D_1g + B[0]/2 #sum prime notation
    

    B[0] = B[0] - D_1g 
        
    f = np.zeros((N,)) + 1j*np.zeros((N,))

    #f[0] = B[0]/2
    f[1] = B[0] - B[2]/2
    
    ##Calculate the second antiderivative
    for k in np.arange(2,N-1,1):
        f[k] = (1/(2*k))*(B[k-1]-B[k+1])
        
    #f[0] = B[0]/2
    ##Enforce boundary condition f(x_end) = 0
    
    D_m2g = 0 + 0*1j
    for k in np.arange(1,N,1):
        D_m2g = D_m2g + (x_end)**k * f[k]
    
    D_m2g =  D_m2g + f[0]/2 #sum prime notation
    
    f[0] = f[0] - D_m2g
     
    #f = f.reshape(N,)

    return f

#%% algorithm2
def algorithm2(eta_trial,eta_singular,alpha,beta,U,sigma):
# =============================================================================
#     INPUT: N+1 chebyshev coefficients of the trial kinematics (eta_trial) and eta_singular, and alpha and beta
#     OUTPUT: The operator L applied to the trial kinematics (eta_trial)
#==============================================================================
    #1 Evaluate eq(26) for d/dx (Psi)
    aa = 4*np.pi**2 * eta_trial
    bb = (- 4*np.pi*1j*U*cheb_diff(eta_trial)).reshape(len(eta_trial),)
    cc = (- U**2 * cheb_diff(cheb_diff(eta_trial))).reshape(len(eta_trial),)
    DPsi = aa + bb + cc
    Psi = np.zeros((len(DPsi),)) + 1j*np.zeros((len(DPsi),))
    
    Psi[1] = DPsi[0] - DPsi[2]/2
    #Integrate DPsi once to solve for a_k, k = 1,...,N. We can't find a_0 yet (constant of integration)
    for k in np.arange(2,len(DPsi)-1,1):
        Psi[k] = (1/(2*k))*(DPsi[k-1]-DPsi[k+1])
    
        
    
    
    a_k = Psi #get the coefficients a_k
     
    Q_r = fst(a_k).reshape(len(eta_trial),) #Compute the regular part of the hydrodynamic load with the ifst from a_k's
    eta_trial = ifct(eta_trial) #4: transform back to physical space from spectral space
    
    beta_eta_Q_r = beta*eta_trial + Q_r
    
    P_m1_beta_eta_Q_r = fct(beta_eta_Q_r)
    P_m1_beta_eta_Q_r = algorithm1(P_m1_beta_eta_Q_r,1)
    P_m1_beta_eta_Q_r = ifct(P_m1_beta_eta_Q_r)/alpha
    P_m1_beta_eta_Q_r = fct(P_m1_beta_eta_Q_r)
    P_m1_beta_eta_Q_r = algorithm1(P_m1_beta_eta_Q_r,-1)
   # P_m1_beta_eta_Q_r = algorithm1(fct(ifct(algorithm1(fct(beta_eta_Q_r),1))/alpha),-1) #Transform back to spectral space and apply algorithm1 with x_end = 1, then transform to physical space and divide by alpha and then apply algorithm1 again, but with x_end = -1
    
    eta_trial = fct(eta_trial) #transform back to spectral space
    
    V_hat = (2*np.pi*1j * eta_trial + U * cheb_diff(eta_trial)) #compute V(x) from eq(29)
    a_0 = -U*Theodorsen(sigma)*(V_hat[0]+V_hat[1]) + U * V_hat[1]
    
    

    #eta_s = np.transpose(np.array(fct(eta_singular),ndmin = 2))
    l_eta = eta_trial.reshape(len(eta_trial),) - a_0*eta_singular.reshape(len(eta_singular),) - P_m1_beta_eta_Q_r.reshape(len(eta_trial),) #build the final vector L[eta_trial]
    
    return l_eta

#%% L_eta_func
def L_eta_func(eta_trial):
    
    N = 1001
    n = np.arange(0,N,1)
    theta = np.pi*(2*n+1)/((2*(N)))
    x = np.cos(theta)	
    b = 1
    c = 1
    rho = 1
    U_inf = 1#4*np.pi**2
    
    R = 1
    f = np.pi*2
    E = 20
    beta = 8*np.pi**2*R
    S =15#E*b**3/(rho*U_inf**2*c**3)
    sigma =1.5 # np.pi*c*f/U_inf
    alpha = S*8*np.pi**2/(3*sigma**2)
    U = 2*np.pi/sigma

    DDeta_singular = (1/(2*alpha))*((2+x)*np.sqrt(1-x**2)-(1+2*x)*np.arccos(x))
    eta_singular = algorithm1(fct(DDeta_singular),-1)
    L_eta = algorithm2(eta_trial,eta_singular,alpha,beta,U,sigma)
    
    return L_eta.reshape(N,)
#%% algorithm3
def algorithm3(eta_le,eta_le_p,tol,x,eta_trial):
    
    Lambda = ((eta_le + eta_le_p*x + eta_le_p*x/x + 1j*0*x).reshape(len(x),))
    Lambda = fct(Lambda)
    #capL = aslinearoperator(algorithm2(eta_trial,eta_singular,alpha,beta))
    L = LinearOperator((1001,1001), matvec = L_eta_func)

    eta = gmres(L,Lambda,tol)
    
    return eta