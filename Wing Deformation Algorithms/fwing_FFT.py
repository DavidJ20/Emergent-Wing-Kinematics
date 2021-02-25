# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 21:47:35 2021

@author: David Yudin
"""

#%% Libraries
import numpy as np
from scipy.fft import dct, idct, dst, idst, fftn, ifftn,fftfreq,fftshift,fft,ifft
import scipy.special
from scipy.sparse.linalg import gmres,LinearOperator
from numpy.polynomial.chebyshev import chebval as chebval


#%% fct
def fct(f):
# =============================================================================
#     INPUT: Function evauluated at callocation points
#     OUTPUT: Chebyshev coefficients  
#==============================================================================
    ##Calculate chebyshev coefficients with DCT4
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
    C_sigma = scipy.special.kv(1,1j*sigma)/(scipy.special.kv(0,1j*sigma)+scipy.special.kv(1,1j*sigma))
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
def algorithm2(eta_trial,eta_singular,alpha,beta,U,sigma,n):
# =============================================================================
#     INPUT: N+1 chebyshev coefficients of the trial kinematics (eta_trial) and eta_singular, and alpha and beta
#     OUTPUT: The operator L applied to the trial kinematics (eta_trial)
#==============================================================================
    D_m1_eta_trial =np.zeros((len(eta_trial),)) + 1j*np.zeros((len(eta_trial),))
    D_m1_eta_trial[1] = eta_trial[0] - eta_trial[2]/2

    for k in np.arange(2,len(eta_trial)-1,1):
        D_m1_eta_trial[k] = (1/(2*k))*(eta_trial[k-1]-eta_trial[k+1])
    #1 Evaluate eq(26) for d/dx (Psi)
    aa = n**2 * D_m1_eta_trial
    bb = (- 2*n*1j*U*(eta_trial)).reshape(len(eta_trial),)
    cc = (- U**2 * (cheb_diff(eta_trial))).reshape(len(eta_trial),)
    Psi = aa + bb + cc
    # Psi = np.zeros((len(DPsi),)) + 1j*np.zeros((len(DPsi),))
    
    # Psi[1] = DPsi[0] - DPsi[2]/2
    # #Integrate DPsi once to solve for a_k, k = 1,...,N. We can't find a_0 yet (constant of integration)
    # for k in np.arange(2,len(DPsi)-1,1):
    #     Psi[k] = (1/(2*k))*(DPsi[k-1]-DPsi[k+1])
    
    a_k = Psi #get the coefficients a_k
     
    Q_r = ifst(a_k).reshape(len(eta_trial),) #Compute the regular part of the hydrodynamic load with the ifst from a_k's
    eta_trial = ifct(eta_trial) #4: transform back to physical space from spectral space

        
    beta_eta_Q_r = n**2*beta*eta_trial + Q_r
    
    P_m1_beta_eta_Q_r_0 = fct(beta_eta_Q_r)
    P_m1_beta_eta_Q_r_1 = algorithm1(P_m1_beta_eta_Q_r_0,1) #apply algorithm 1
    P_m1_beta_eta_Q_r_2 = ifct(P_m1_beta_eta_Q_r_1) #transform to physical space and divide by alpha (diving by alpha  is removed due to the convolution introduced by the Fourier spectral method) ####(01/21/21)
    P_m1_beta_eta_Q_r_3 = fct(P_m1_beta_eta_Q_r_2) #transform back to spectral space
    P_m1_beta_eta_Q_r_4 = algorithm1(P_m1_beta_eta_Q_r_3,-1) #apply algorithm 1 once more
   # P_m1_beta_eta_Q_r = algorithm1(fct(ifct(algorithm1(fct(beta_eta_Q_r),1))/alpha),-1) #Transform back to spectral space and apply algorithm1 with x_end = 1, then transform to physical space and divide by alpha and then apply algorithm1 again, but with x_end = -1
    
    eta_trial = fct(eta_trial) #transform back to spectral space
    
    V_hat = (2*np.pi*1j * eta_trial + U * cheb_diff(eta_trial)) #compute V(x) from eq(29)
    a_0 = -U*Theodorsen(n)*(V_hat[0]+V_hat[1]) + U * V_hat[1]
    
    #eta_s = np.transpose(np.array(fct(eta_singular),ndmin = 2))
    l_eta = eta_trial.reshape(len(eta_trial),) - a_0*eta_singular.reshape(len(eta_singular),) - P_m1_beta_eta_Q_r_4.reshape(len(eta_trial),) #build the final vector L[eta_trial]
    
    return l_eta

#%% L_eta_func
def L_eta_func(eta_trial):

    N = 1001
    n = np.arange(0,N,1)
    theta = np.pi*(2*n+1)/((2*(N)))
    x = np.cos(theta)
    t = np.linspace(0,1,N)
    omega = 1

    R = 1
    beta = 8*np.pi**2*R
    S = 15
    sigma = 1.5 # np.pi*c*f/U_inf
    alpha = S*8*np.pi**2/(3*sigma**2)*t
    alpha_hat = fft(alpha)
    U = 2*np.pi/sigma

    DDeta_singular = (1/(2))*((2+x)*np.sqrt(1-x**2)-(1+2*x)*np.arccos(x)) #Division by alpha removed ####(01/21/21)
    eta_singular = algorithm1(fct(DDeta_singular),-1)
    
    L_eta = algorithm2(eta_trial,eta_singular,alpha,beta,U,sigma)
    return L_eta.reshape(N,)
#%% FFT_w
def algorithm4(w_trial):
    n = len(w_trial)
    L_w_hat = np.zeros(len(w_trial))
    w_hat = fft(w_trial) #find the Fourier coefficients of our trial kinematics
    w_hat = fct(w_hat) #convert to spectral space
    for n in np.arange(-n,n+1):
        L_w_hat[0] = L_eta_func(w_hat[n])
    return 0
#%% convolve alpha and w_hat's
def cv(alpha_hat,w_hat):
    N = len(w_hat)
    w_hat_convolved = []
    for i in range(N):
        c = np.convolve(alpha_hat,w_hat[i,:],'same')
        w_hat_convolved = np.append(w_hat_convolved,c)
    return w_hat_convolved
#%% Solve equation (23) for each fixed frequency n (in spectral space)
def eq23(w_hat_trial,U,N): #Expect input to be in matrix form initially (N by K)
    w_hat_trial = w_hat_trial.reshape([10,10])
    Psi = [] #in spectral space
    for n in np.arange(0,N):
        D_m1_w_hat_trial = np.zeros((len(w_hat_trial[n,:]),)) + 1j*np.zeros((len(w_hat_trial[n,:]),)) #define an empty array to hold the first antiderivative of w_hat_n
        D_m1_w_hat_trial[1] = w_hat_trial[0,n] - w_hat_trial[2,n]/2 #define the first term in the sum prime array

        for k in np.arange(2,len(w_hat_trial[n,:])-1,1):
            D_m1_w_hat_trial[k] = (1/(2*k))*(w_hat_trial[k-1,n]-w_hat_trial[k+1,n])
        aa = n**2 * D_m1_w_hat_trial
        bb = (- 2*n*1j*U*(w_hat_trial[n,:]))#.reshape(len(w_hat_trial[n,:]),)
        cc = (- U**2 * (cheb_diff(w_hat_trial[n,:])))#.reshape(len(w_hat_trial[n,:]),)
        Psi_n = aa + bb + cc
        Psi = np.append(Psi,Psi_n)
    return Psi.reshape([N,K])
#%% Perform the IDST on a_k,n to compute the regular part of the hydrodynamic load Q_r,n (in spectral space)
def compute_Q_r(a_k_n):
    N  = len(a_k_n[1,:])
    q_n_hat_r = []
    #First we must take the Fourier coefficients of the a_k's (as in eq (13)) and convert them back to an array of a_k using the inverse fast Fourier transform
    for n in np.arange(0,N):
        q_n_hat = ifst(a_k_n[n,:]) #Take the nth column of a_k_n, and perform the ifst to find the nth Fourier coefficient, q_n_hat, of the hydrodynamic load
        q_n_hat_r = np.append(q_n_hat_r,q_n_hat) #flattened matrix of the regular part of the load

    return q_n_hat_r.reshape([N,N]) #in spectral space. Each row is the Fourier coefficent q_n_hat_r
#%% Next we apply the preconditioner to the leftmost two terms of eq (30)
def precondition(q_n_hat_r,w_hat_trial,mu): #q in spectral, w_hat in spectral
    N = len(w_hat_trial[1,:])
    mu_n_2_w_hat_trial = []
    preconditioned = []
    for n in np.arange(0,N):
        mu_n_2_w_hat_trial_n = mu*n**2*w_hat_trial[n,:] #multiply in spectral space
        mu_n_2_w_hat_trial = np.append(mu_n_2_w_hat_trial,mu_n_2_w_hat_trial_n) #append
    q_plus_w = q_n_hat_r + mu_n_2_w_hat_trial.reshape([N,N]) #add the two terms in the preconditioner's argument
    for n in np.arange(0,N):
        p_m1 = algorithm1(q_plus_w[n,:],1)
        p_m2 = algorithm1(p_m1,-1)
        preconditioned = np.append(preconditioned, p_m2)
    return preconditioned.reshape([N,N])
#%% operator construction
def linoperator(w_hat_trial):

    N = 100
    n = np.arange(0,N,1)
    theta = np.pi*(2*n+1)/((2*(N)))
    x = np.cos(theta)
    U = 1
    mu = 1
    sigma = 1.5
    t = np.linspace(0,1,N)
    alpha = np.exp(2*np.pi*t)
    alpha_hat = fftshift(fft(alpha)/N)
    term1 = eq23(w_hat_trial,U,N)
    Q_r = compute_Q_r(term1) #computed the regular part of the hydrodynamic load in spectral space
    D2_w_s = 0.5*((2+x)*np.sqrt(1-x**2) - (1+2*x)*np.arccos(x)) #define the semianalytical, de-singularized load
    D2_w_s = fct(D2_w_s) #convert to spectral space
    D_w_s = algorithm1(D2_w_s,-1)
    w_s = algorithm1(D_w_s,-1) #precompute singular part of hydrodynamic load
    
    #Now we compute the a_0,n values
    
    q_s = [] #Define an empty array to hold values
    
    for n in np.arange(-N/2,N/2):
        vhat_n = 1j*n * w_hat_trial[n,:] + U*cheb_diff(w_hat_trial)
        a_hat_0_n = -U*Theodorsen(n*sigma)*(vhat_n[0] +vhat_n[1]) + U*vhat_n[1]
        q_s_n = a_hat_0_n*w_s*1/2 #compute the first term in eq(18)
        q_s = np.append(q_s,q_s_n)
        
    tp = precondition(Q_r, w_hat_trial, mu)
    alpha_w_hat_trial = cv(alpha_hat,w_hat_trial)
    
    L = alpha_w_hat_trial - q_s - tp #construct the final operator
    
    return L