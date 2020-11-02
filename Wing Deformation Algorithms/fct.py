#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 02:14:26 2020

@author: davidy
"""

import numpy as np
from scipy.fftpack import dct as dct


def fct(f,N):
    
# =============================================================================
#     INPUT: Function evauluated at callocation points
#     OUTPUT: Chebyshev coefficients 
# =============================================================================
    b = dct(f,norm = 'ortho')
    b[0] = b[0]*np.sqrt(1/(N))
    b[1:-1] = b[1:-1]*np.sqrt(2/(N))

    
    return b