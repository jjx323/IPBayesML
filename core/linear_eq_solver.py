#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 21:03:17 2022

@author: Junxiong Jia
"""

import numpy as np
import scipy.sparse.linalg as spsl
import scipy.sparse as sps
import cupy as cp


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
        

def spsolve_lu(L, U, b, perm_c=None, perm_r=None):
    """ an attempt to use SuperLU data to efficiently solve
        Ax = Pr.T L U Pc.T x = b
         - note that L from SuperLU is in CSC format solving for c
           results in an efficiency warning
        Pr . A . Pc = L . U
        Lc = b      - forward solve for c
         c = Ux     - then back solve for x
         
        (spsolve_triangular and spsolve seem all much less efficient than the 
        lu.solve() method in scipy, so the overall efficiency approximate to the spsolve if 
        we include the computational time of splu. 
        
        When we only use splu once and use spsolve_lu many times, 
        this implementation may be useful. However, we may use lu.solve() (scipy function)
        since it is much more efficient. 
        
        When implement autograd by pytorch (lu.solve in scipy can hardly be employed), 
        we may use splu once and spsolve_lu twice. 
        In this case, there seems no advantage compared with using spsolve directly.
        How to implement spsolve_lu much more efficient still needs to be explored!!)
    """
    if perm_r is not None:
        perm_r_rev = np.argsort(perm_r)
        b = b[perm_r_rev]

    try:    # unit_diagonal is a new kw
        ## spsolve_triangular seems not efficient as that of spsolve with permc_spec="NATURAL"
        # c = spsl.spsolve_triangular(L, b, lower=True, unit_diagonal=True)
        c = spsl.spsolve(L, b, permc_spec="NATURAL")
    except TypeError:
        c = spsl.spsolve_triangular(L, b, lower=True)
    # px = spsl.spsolve_triangular(U, c, lower=False)
    px = spsl.spsolve(U, c, permc_spec="NATURAL")
    if perm_c is None:
        return px
    return px[perm_c]


class SuperLU_GPU():

    def __init__(self, shape, L, U, perm_r, perm_c, nnz):
        """LU factorization of a sparse matrix.
           This function is modified from the SuperLU implementations in CuPy.
        Args:
            shape, L, U, perm_r, perm_c, nnz are typically variables in an scipy 
            object (scipy.sparse.linalg.SuperLU: LU factorization of a sparse
            matrix, computed by `scipy.sparse.linalg.splu`, etc).
        """

        self.shape = shape
        self.nnz = nnz
        self.perm_r = cp.array(perm_r)
        self.perm_c = cp.array(perm_c)
        self.L = cp.sparse.csr_matrix(L.tocsr())
        self.U = cp.sparse.csr_matrix(U.tocsr())

        self._perm_c_rev = cp.argsort(self.perm_c)
        self._perm_r_rev = cp.argsort(self.perm_r)
        

    def solve(self, rhs, trans='N'):
        """Solves linear system of equations with one or several right-hand sides.
        Args:
            rhs (cupy.ndarray): Right-hand side(s) of equation with dimension
                ``(M)`` or ``(M, K)``.
            trans (str): 'N', 'T' or 'H'.
                'N': Solves ``A * x = rhs``.
                'T': Solves ``A.T * x = rhs``.
                'H': Solves ``A.conj().T * x = rhs``.
        Returns:
            cupy.ndarray:
                Solution vector(s)
        """
        if not isinstance(rhs, cp.ndarray):
            raise TypeError('ojb must be cupy.ndarray')
        if rhs.ndim not in (1, 2):
            raise ValueError('rhs.ndim must be 1 or 2 (actual: {})'.
                             format(rhs.ndim))
        if rhs.shape[0] != self.shape[0]:
            raise ValueError('shape mismatch (self.shape: {}, rhs.shape: {})'
                             .format(self.shape, rhs.shape))
        if trans not in ('N', 'T', 'H'):
            raise ValueError('trans must be \'N\', \'T\', or \'H\'')
        if not cp.cusparse.check_availability('csrsm2'):
            raise NotImplementedError

        x = rhs.astype(self.L.dtype)
        if trans == 'N':
            if self.perm_r is not None:
                x = x[self._perm_r_rev]
            cp.cusparse.csrsm2(self.L, x, lower=True, transa=trans)
            cp.cusparse.csrsm2(self.U, x, lower=False, transa=trans)
            if self.perm_c is not None:
                x = x[self.perm_c]
        else:
            if self.perm_c is not None:
                x = x[self._perm_c_rev]
            cp.cusparse.csrsm2(self.U, x, lower=False, transa=trans)
            cp.cusparse.csrsm2(self.L, x, lower=True, transa=trans)
            if self.perm_r is not None:
                x = x[self.perm_r]

        if not x._f_contiguous:
            # For compatibility with SciPy
            x = x.copy(order='F')
        return x


















 
