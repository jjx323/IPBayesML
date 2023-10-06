#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 16:58:21 2022

@author: Jiaming Sui and Junxiong Jia and Haoyu Lu
"""

import numpy as np
import scipy.sparse as sps
# import scipy.sparse.linalg as spsl


def pre_chol_QR(Y_, M, eva_Qbar=True):
    Z, Ry = np.linalg.qr(Y_)
    Zbar = M(Z)
    Rz = np.linalg.cholesky(Z.T@Zbar)
    ## The output of np.linalg.cholesky is different from that discribled in 
    ## "U. Villa, N. Petra, O, Ghattas, hIPPYlib" collected in the file references
    ## /core/References
    ## So we need to use Rz.T in the following code
    tmp = np.linalg.inv(Rz.T)
    Q = Z@tmp 
    R = Rz.T@Ry
    if eva_Qbar == True:
        Qbar = Zbar@tmp 
        return Q, Qbar, R
    else:
        return Q, R

def double_pass(H, M, Minv, n, r, omega=None, l=20, cutval=None, random_seed=None):

    '''
    Solve eigensystem: H v = \lambda M v;  H and Mshould be symmetric
    References: hIPPYlib.pdf and 
        Randomized algorithms for generalized Hermitian eigenvalue 
        problems with application to computing Karhunen–Loeve expansion.pdf
    input
    H: scipy operator or a function allow vectorized inputs or a sparse matrix
    M: scipy operator or a function allow vectorized inputs or a sparse matrix
    Minv: calculate M^{-1}, scipy operator or a function allow vectorized inputs
          or a sparse matrix
    omega : a random matrix with the size of dim-of-eigenvector * (r + l)
    n : length of the parameter v
    r : the number of eigenpairs we wish to compute
    l : an oversampling factor
    cutval : truncated value of eigenvalues
 
    output
    
    d : eigenvalues in descending order
    U : eigenvector
    '''
    if random_seed is not None:
        np.random.seed(seed=random_seed)
        
    if omega is None:
        omega = np.random.randn(n, r+l)
    
    if sps.isspmatrix(H): 
        def H(x):
            return np.array(H@x)
    
    if sps.isspmatrix(M):
        def M(x):
            return np.array(M@x)
    
    if sps.isspmatrix(Minv):
        def Minv(x):
            return np.array(Minv@x)

    Ybar = H(omega)
    Y = Minv(Ybar)
    Q, R = pre_chol_QR(Y, M, eva_Qbar=False)
    AQ = H(Q)
    T = Q.T@AQ
    d, V = np.linalg.eigh(T)
    
    sort_perm = d.argsort()
    sort_perm = sort_perm[::-1]
    d = d[sort_perm[0:r]]
    V = np.array(V[:, sort_perm[0:r]])
    
    if cutval is not None:
        index = d > cutval    
        d = d[index]
        V = V[:, index]

    U = Q@V
    return np.array(d), np.array(U)


def single_pass(H, M, Minv, n, r, omega=None, l=20, cutval=None, random_seed=None):

    '''
    Needs further refinement!!!!!
    
    Solve eigensystem: H v = \lambda M v;  H and Mshould be symmetric 
    References: hIPPYlib.pdf and 
        Randomized algorithms for generalized Hermitian eigenvalue 
        problems with application to computing Karhunen–Loeve expansion.pdf
    input
    H: scipy operator or a function allow vectorized inputs
    M: scipy operator or a function allow vectorized inputs
    Minv: calculate M^{-1}, scipy operator or a function allow vectorized inputs
    omega : a random matrix with the size of dim-of-eigenvector * (r + l)
    n : length of the parameter v
    r : the number of eigenpairs we wish to compute
    l : an oversampling factor
    cutval : truncated value of eigenvalues
 
    output
    
    d : eigenvalues in descending order
    U : eigenvector
    '''
    
    if random_seed is not None:
        np.random.seed(seed=random_seed)
    
    if omega is None:
        omega = np.random.randn(n, r+l)

    Ybar = H(omega)
    Y = Minv(Ybar)
    Q, Qbar, R = pre_chol_QR(Y, M)

    ## solve a least-squares problem formula (29) of hIPPYlib.pdf in the references
    tmp1 = Qbar.T@omega
    tmp2 = Qbar.T@Y
    T = tmp2@tmp1@np.linalg.inv(tmp1@tmp1.T)
    # tmp1inv = np.linalg.inv(tmp1) 
    # T = tmp1inv@tmp2@tmp1inv

    d, V = np.linalg.eigh(T)
    
    sort_perm = d.argsort()
    sort_perm = sort_perm[::-1]
    d = d[sort_perm[0:r]]
    V = V[:, sort_perm[0:r]]
    
    if cutval is not None:
        index = d > cutval    
        d = d[index]
        V = V[:, index]
    
    U = Q@V
    return np.array(d), np.array(U)










