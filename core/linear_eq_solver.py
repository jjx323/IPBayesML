#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 21:03:17 2022

@author: jjx323
"""

import numpy as np
import scipy.sparse.linalg as spsl
import scipy.sparse as sps


def trans_matrix_to_operator(A):
    leng = A.shape[0]
    def AA(vec):
        return np.array(A@vec)
    Aop = spsl.LinearOperator((leng, leng), matvec=AA)
    return Aop


def cg_my(A, b, M=None, Minv=None, x0=None, tol=0.5, atol=0.1, maxiter=1000, 
          curvature_detector=False):
    '''
    Solving Ax = b by (preconditioned) conjugate gradient algorithm
    The following algorithm is implemented according to the following tutorial:
        http://math.stmarys-ca.edu/wp-content/uploads/2017/07/Mike-Rambo.pdf
    The references can be found in the folder "/core/Reference/"
    The terminate conditions are implemented according to the following article (Pages: 465-466):
        O. Ghattas, K. Willcox, Learning physics-based models from data: 
        perspectives from inverse problems and model reduction, Acta Numerica, 2021
    
    cg iteration will terminate when 
    1. norm(residual) <= min(atol, tol*|b|)
    2. curvature = di A di <= 0 (di is pk in the program)
    3. reach the maxiter
    
    input:
    A: could be a symmetric positive definite matrix (np.ndarray) or operator with shape (n, n)
    M: could be a matrix (np.ndarray) or operator with shape (n, n)
    Minv: could be a matrix (np.ndarray) or operator with shape (n, n)
        Here, M and Minv are set differet as in scipy. Minv is the matrix similar to A^{-1}
        If Minv cannot be calculate explicitly, we still can specify M if M^{-1}x can be 
        calculated easier than A^{-1}x
    b: a vector (np.ndarray), nx1
    
    output:
    1. xk: np.ndarray with shape (n,)
    2. info: converged, reach_maxiter, curvature_limit
    '''
    
    if type(A) == np.ndarray:
        A = trans_matrix_to_operator(A)
    leng = A.shape[0]
    
    if type(M) == np.ndarray:
        M = trans_matrix_to_operator(M)
        assert M.shape == A.shape, "Aop and Mop should has same shape"
        
    if type(Minv) == np.ndarray:
        Minv = trans_matrix_to_operator(Minv)
        assert Minv.shape == A.shape, "Aop and Mop should has same shape"
        
    info = "infomation"
    
    ## init
    if x0 is None:
        x0 = np.zeros(leng)
    assert x0.shape[0] == leng, "Incompactable init value"
    
    if M is None and Minv is None:
        rk_ = np.squeeze(b - A*x0)
        pk = rk_.copy()
        w = np.squeeze(A*pk)
        alphak = (rk_@rk_)/(pk@w)
    elif Minv is not None:
        rk_ = np.squeeze(b - A*x0)
        zk_ = np.squeeze(Minv*rk_)
        pk = zk_.copy()
        w = np.squeeze(A*pk)
        alphak = (rk_@zk_)/(pk@w)
    elif M is not None:
        rk_ = np.squeeze(b - A*x0)
        zk_ = np.squeeze(spsl.bicgstab(M, rk_)[0])
        pk = zk_.copy()
        w = np.squeeze(A*pk)
        alphak = (rk_@zk_)/(pk@w)
        
    xk = x0 + alphak*pk
    rk = rk_ - alphak*w
        
    k = 1
    while k <= maxiter:
        if M is None and Minv is None:
            t1 = rk@rk
            betak = t1/(rk_@rk_)
            _pk = rk + betak*pk
        elif Minv is not None:
            zk = np.squeeze(Minv*rk)
            rkzk = rk@zk
            betak = rkzk/(rk_@zk_)
            _pk = zk + betak*pk
        elif M is not None:
            zk = np.squeeze(spsl.bicgstab(M, rk)[0])
            rkzk = rk@zk
            betak = rkzk/(rk_@zk_)
            _pk = zk + betak*pk
            
        w = np.squeeze(A*_pk)
        t2 = _pk@w 
        if curvature_detector == True:
            if t2 <= 0:
                info = "curvature_limit"
                if k == 1: xk = b.copy()
                break
        
        if M is None and Minv is None:
            # _alphak = (rk@rk)/(_pk@w)
            _alphak = t1/t2
        else:
            _alphak = rkzk/t2        
            
        _xk = xk + _alphak*_pk
        _rk = rk - _alphak*w
        if np.sqrt(_rk@_rk) <= min(atol, tol*np.sqrt(b@b)):
            info = "converged"
            break
        
        k = k + 1
        
        xk = _xk.copy()
        rk = _rk.copy()
        pk = _pk.copy()
    
    if info != "converged" and info != "curvature_limit":
        info = "reach_maxiter"
        
    return xk, info, k
        

























 