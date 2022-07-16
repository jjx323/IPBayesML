#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 16:58:21 2022

@author: Jiaming Sui and Junxiong Jia
"""

import numpy as np
# import scipy.sparse as sps
# import scipy.sparse.linalg as spsl


def pre_chol_QR(Y_, prior_ope):
    Z, Ry = np.linalg.qr(Y_)
    Zbar = prior_ope(Z)
    Rz = np.linalg.cholesky(Z.T @ Zbar)
    Q = Z @ np.linalg.inv(Rz)
    R = Rz @ Ry
    return Q, R


def double_pass(H, M, Minv, omega, r, Mg, l=20, cutval = 1):

    '''
    Solve eigensystem: H v = \lambda M v 
    References: hIPPYlib.pdf and 
        Randomized algorithms for generalized Hermitian eigenvalue 
        problems with application to computing Karhunen–Loeve expansion.pdf
    input
    H: scipy operator or a function allow vectorized inputs
    M: scipy operator or a function allow vectorized inputs
    Minv: calculate M^{-1}, scipy operator or a function allow vectorized inputs
    omega : a random matrix with the size of dim-of-eigenvector * (r + l)
    r : the number of eigenpairs we wish to compute
    l : an oversampling factor
    cutval : truncated value of eigenvalues
 
    output
    
    d : eigenvalues in descending order
    U : eigenvector
    '''
    
    np.random.seed(seed=1)
    # Ybar = H * omega
    Ybar = H(omega)
    # Y = Minv * Ybar
    Y = Minv(Ybar)
    Q, R = pre_chol_QR(Y, M)
    # AQ = H * Q
    AQ = H(Q)
    T = Q.T @ AQ
    d, V = np.linalg.eigh(T)
    sort_perm = d.argsort()
    sort_perm = sort_perm[::-1]
    d = d[sort_perm[0:r]]
    V = V[:, sort_perm[0:r]] 
    
    index = d > cutval
    
    d = d[index]
    V = V[:, index]
    
    U = Q @ V
    return d, U










