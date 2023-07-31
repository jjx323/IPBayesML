#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 19:49:46 2022

@author: Junxiong Jia
"""

import numpy as np
import fenics as fe
import scipy.linalg as sl
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
# from scipy.sparse.linalg import LinearOperator
# from scipy.sparse.linalg import eigs, eigsh
# import dolfin as dl 

from core.misc import trans2spnumpy, construct_measurement_matrix


#############################################################################
class GaussianElliptic2(object):
    '''
    prior Gaussian probability measure N(m, C)
    C^{-1/2}: an elliptic operator -\alpha\nabla(\cdot\Theta\nabla \cdot) + a(x) Id
    
    Ref: A computational framework for infinite-dimensional Bayesian inverse problems
    part I: The linearized case, with application to global seismic inversion,
    SIAM J. Sci. Comput., 2013
    '''
    def __init__(self, domain, alpha=1.0, a_fun=fe.Constant(1.0), theta=1.0, 
                 mean_fun=None, tensor=False, boundary='Neumann',
                 use_LU=True):
        
        """
        boundary (string): 'Neumann' or 'Dirichlet'
        mean_fun (fenics.Function or None): None(set the mean function to zero)

        use_LU (True or False): 
                take LU decomposition of the sparse matrix K and M, then Mx=b and Kx=b 
                are all solved by lu.solve directly that may be faster than spsolve. 
                (splu in scipy may take longer time than spsolve, however, if we need 
                 to generate many samples, we only need to run splu once)
        """
        assert type(alpha) == type(1.0) or type(alpha) == type(np.array(1.0)) \
            or type(alpha) == type(1)
        assert boundary == 'Neumann' or boundary == 'Dirichlet', \
                "boundary must be 'Neumann' or 'Dirichlet'"
        
        self.domain = domain
        self.function_space_dim = self.domain.function_space.dim()
        self._alpha = alpha
        if type(a_fun) == type(1.0) or type(a_fun) == type(np.array(1.0)) or type(a_fun) == type(1):
            a_fun = fe.Constant(a_fun)
        self._a_fun = fe.interpolate(a_fun, domain.function_space)
        self._tensor = tensor

        if self._tensor == False:
            assert type(theta) == type(1.0) or type(theta) == type(np.array(1.0)) or type(theta) == type(1)
            self._theta = fe.interpolate(fe.Constant(theta), self.domain.function_space)
        elif self._tensor == True:
            self._theta = fe.as_matrix(((fe.interpolate(theta[0], self.domain.function_space), \
                                         fe.interpolate(theta[1], self.domain.function_space)), \
                                        (fe.interpolate(theta[2], self.domain.function_space), \
                                         fe.interpolate(theta[3], self.domain.function_space))))
        if mean_fun == None:
            self.mean_fun = fe.interpolate(fe.Expression("0.0", degree=2), self.domain.function_space)
        else:
            self.mean_fun = fe.interpolate(mean_fun, self.domain.function_space)
        self._mean_vec = self.mean_fun.vector()[:]

        u = fe.TrialFunction(self.domain.function_space)
        v = fe.TestFunction(self.domain.function_space)
        aa = fe.Constant(self._alpha)*fe.inner(self._theta*fe.grad(u), fe.grad(v))*fe.dx \
            + fe.inner(self._a_fun*u, v)*fe.dx
        self.K_ = fe.assemble(aa)
        bb = fe.inner(u, v)*fe.dx
        self.M_ = fe.assemble(bb)

        self.bc = boundary
        self.boundary(self.M_)
        self.boundary(self.K_)
        self.index_boundary = None
        if self.bc == "Dirichlet":
            self.boundary_index()

        # construct numpy matrix
        self.K = trans2spnumpy(self.K_)
        self.M = trans2spnumpy(self.M_)
        # self.Mid = sps.diags(1.0/self.M.diagonal())
        lamped_elements = np.array(np.sum(self.M, axis=1)).flatten()
        self.M_lamped_half = sps.diags(np.sqrt(lamped_elements))
        # self.M_half = sps.diags(np.sqrt(self.M.diagonal()))
        # construct FEniCS matrix
        self.M_lamped_half_ = fe.assemble(fe.inner(u, v)*fe.dx)
        # self.Mid_ = fe.assemble(fe.inner(u, v)*fe.dx)
        self.M_lamped_half_.zero()
        v = fe.Vector()
        self.M_.init_vector(v, 1)
        v[:] = np.sqrt(lamped_elements)
        self.M_lamped_half_.set_diagonal(v)
        # self.Mid_.zero()
        # vv = fe.Vector()
        # vv[:] = 1.0/self.M.diagonal()
        # self.Mid_.set_diagonal(vv)

        # auxillary functions
        self.temp0 = fe.Function(self.domain.function_space)
        self.temp1 = fe.Function(self.domain.function_space)
        self.temp2 = fe.Function(self.domain.function_space)
        
        ## using LU decomposition to accelerate computation
        self.use_LU = use_LU
        self.luM, self.luK = None, None
        if self.use_LU == True:
            self.luM = spsl.splu(self.M.tocsc()) 
            self.luK = spsl.splu(self.K.tocsc())
            
    @property
    def alpha(self):
        return self._alpha
    
    @alpha.setter
    def alpha(self, al):
        assert type(al) == type(1.0) or type(al) == type(np.array(1.0)) or type(al) == type(1)
        self._alpha = al
        self.generate_K()
        
    @property
    def a_fun(self):
        return self._a_fun
        
    @a_fun.setter
    def a_fun(self, al):
        self._a_fun = fe.interpolate(al, self.domain.function_space)
        self.generate_K()
        
    @property
    def mean_vec(self):
        self._mean_vec = np.array(self.mean_fun.vector()[:])
        return self._mean_vec
    
    @mean_vec.setter 
    def mean_vec(self, mean):
        self._mean_vec = np.array(mean)
        self.mean_fun.vector()[:] = self._mean_vec
    
    @property
    def theta(self):
        return self._theta
    
    @theta.setter
    def theta(self, th):
        if self._tensor == False:
            assert type(th) == type(1.0) or type(th) == type(np.array(1.0)) or type(th) == type(1)
            self._theta = th
        elif self._tensor == True:
            self._theta = fe.as_matrix(((fe.interpolate(th[0], self.domain.function_space), \
                                         fe.interpolate(th[1], self.domain.function_space)), \
                                        (fe.interpolate(th[2], self.domain.function_space), \
                                         fe.interpolate(th[3], self.domain.function_space))))
        self.generate_K()
        
    def update_mean_fun(self, mean_fun_vec):
        self.mean_fun.vector()[:] = mean_fun_vec
        
    def boundary(self, b):
        if self.bc == 'Dirichlet':
            def boundary(x, on_boundary):
                return on_boundary
            bc = fe.DirichletBC(self.domain.function_space, fe.Constant('0.0'), boundary)
            bc.apply(b)
    
    def boundary_index(self):
        a = fe.Function(self.domain.function_space)
        a.vector()[:] = 1.0
        v_ = fe.TestFunction(self.domain.function_space)
        aa = fe.assemble(a*v_*fe.dx)
        bb = fe.assemble(a*v_*fe.dx)
        
        def boundary(x, on_boundary):
            return on_boundary
        bc = fe.DirichletBC(self.domain.function_space, fe.Constant('0.0'), boundary)
        
        bc.apply(aa)
        self.index_boundary = (aa[:] != bb[:])
    
    def boundary_vec(self, b):
        if self.bc == 'Dirichlet':
            b[self.index_boundary] = 0.0
        return b
        
    def generate_K(self):
        u = fe.TrialFunction(self.domain.function_space)
        v = fe.TestFunction(self.domain.function_space)
        theta = self._theta
        a = fe.Constant(self._alpha)*fe.inner(theta*fe.grad(u), fe.grad(v))*fe.dx \
            + fe.Constant(self._alpha)*fe.inner(self._a_fun*u, v)*fe.dx
        self.K_ = fe.assemble(a)
        self.boundary(self.K_)
        self.K = trans2spnumpy(self.K_)
        if self.use_LU == True:
            self.luK = spsl.splu(self.K)
        return self.K
    
    def generate_M(self):
        u = fe.TrialFunction(self.domain.function_space)
        v = fe.TestFunction(self.domain.function_space)
        a = fe.inner(u, v)*fe.dx
        self.M_ = fe.assemble(a)
        self.boundary(self.M_)
        self.M = trans2spnumpy(self.M_)
        if self.use_LU == True:
            self.luM = spsl.splu(self.M)
        return self.M
    
    def generate_sample(self, method='numpy'):
        '''
        generate samples from the Gaussian probability measure
        the generated vector is in $\mathbb{R}_{M}^{n}$ by
        $m = m_{mean} + Ln$ with $L:\mathbb{R}^{n}\rightarrow\mathbb{R}_{M}^{n}$
        method == 'FEniCS' or 'numpy'
        '''
        
        sample = self.generate_sample_zero_mean(method=method)
        sample = self.mean_fun.vector()[:] + sample
        return np.array(sample)
            
    def generate_sample_zero_mean(self, method='numpy'):
        '''
        generate samples from the Gaussian probability measure
        the generated vector is in $\mathbb{R}_{M}^{n}$ by
        $m = 0.0 + Ln$ with $L:\mathbb{R}^{n}\rightarrow\mathbb{R}_{M}^{n}$
        method == 'FEniCS' or 'numpy'
        '''
        
        assert self.K is not None 
        assert self.M_lamped_half is not None
        
        fun = fe.Function(self.domain.function_space)
        
        if method == 'numpy':
            n = np.random.normal(0, 1, (self.function_space_dim,))
            b = self.M_lamped_half@n
            self.boundary_vec(b)
            if self.use_LU == False:
                fun_vec = spsl.spsolve(self.K, b)
            elif self.use_LU == True:
                fun_vec = self.luK.solve(b)
            else:
                raise NotImplementedError("use_LU must be True or False")
            return np.array(fun_vec)
        elif method == 'FEniCS':
            n_ = fe.Vector()
            self.M_lamped_half_.init_vector(n_, 1)
            n_.set_local(np.random.normal(0, 1, (self.function_space_dim,)))
            # fe.solve(self.K_, fun.vector(), self.M_half_*n_, 'cg', 'ilu')
            fe.solve(self.K_, fun.vector(), self.M_lamped_half_*n_)
            return np.array(fun.vector()[:])
        else:
            assert False, "method must be 'FEniCS' or 'numpy'"
            
    def evaluate_CM_inner(self, u_vec, v_vec, method='numpy'):
        """
        evaluate (C^{-1/2}u, C^{-1/2}v)
        """
        
        assert type(u_vec) == np.ndarray
        assert type(v_vec) == np.ndarray

        if method == 'numpy':
            temp1 = u_vec - self.mean_fun.vector()[:]
            temp2 = v_vec - self.mean_fun.vector()[:]
            if self.use_LU == False:
                return temp1@(self.K.T)@spsl.spsolve(self.M, self.K@temp2)
            elif self.use_LU == True:
                val = self.luM.solve(self.K@temp2)
                val = temp1@(self.K.T)@val
                return val
            else:
                raise NotImplementedError("use_LU must be True or False")
        elif method == 'FEniCS':
            self.temp0.vector()[:] = v_vec - self.mean_fun.vector()[:]
            fe.solve(self.M_, self.temp1.vector(), (self.K_*self.temp0.vector()))
            self.K_.transpmult(self.temp1.vector(), self.temp2.vector())
            self.temp0.vector()[:] = u_vec - self.mean_fun.vector()[:]
            return (self.temp0.vector()).inner(self.temp2.vector())
        else:
            assert False, "method must be 'FEniCS' or 'numpy'"

    
    def evaluate_grad(self, u_vec, method='numpy'):
        '''
        calculate the gradient vector at u_vec
        the input vector should be in $\mathbb{R}_{M}^{n}$
        the output vector is in $v1\in\mathbb{R}_{M}^{n}$
        '''
        assert type(u_vec) is np.ndarray
        
         if method == 'numpy':
            res = u_vec - self.mean_fun.vector()[:]
            if self.use_LU == False:
                grad_vec = (self.K.T)@spsl.spsolve(self.M, self.K@res)
                # grad_vec = spsl.spsolve(self.M, grad_vec)
            elif self.use_LU == True:
                grad_vec = self.luM.solve(self.K@res)
                grad_vec = (self.K.T)@grad_vec 
                # grad_vec = self.luM.solve(grad_vec)
            else:
                raise NotImplementedError("use_LU must be True or False")
            return grad_vec
        elif method == 'FEniCS':
            self.temp0.vector()[:] = u_vec - self.mean_fun.vector()[:]
            fe.solve(self.M_, self.temp1.vector(), (self.K_*self.temp0.vector()))
            self.K_.transpmult(self.temp1.vector(), self.temp2.vector())
            # fe.solve(self.M_, self.temp1.vector(), self.temp2.vector())
            return self.temp1.vector()[:]
        else:
            assert False, "method must be 'FEniCS' or 'numpy'"
    
    def evaluate_hessian(self, u_vec):
        '''
        evaluate HessianMatrix^{-1}*(gradient at u_vec)
        '''
        return -u_vec
        
    def evaluate_hessian_vec(self, u_vec, method='numpy'):
        '''
        evaluate HessianMatrix*u_vec
        the input vector should be in $\mathbb{R}_{M}^{n}$,
        the output vector is in $\mathbb{R}_{M}^{n}$
        '''
        assert type(u_vec) is np.ndarray
        
        if method == 'numpy':
            if self.use_LU == False:
                temp = (self.K.T)@spsl.spsolve(self.M, self.K@u_vec)
                # temp = spsl.spsolve(self.M, temp)
            elif self.use_LU == True:
                temp = self.luM.solve(self.K@u_vec)
                temp = (self.K.T)@temp
                # temp = self.luM.solve(temp)
            else:
                raise NotImplementedError("use_LU must be True or False")
            return np.array(temp)
        elif method == 'FEniCS':
            self.temp0.vector()[:] = self.K@u_vec
            fe.solve(self.M_, self.temp1.vector(), self.temp0.vector())
            self.K_.transpmult(self.temp1.vector(), self.temp2.vector())
            # fe.solve(self.M_, self.temp1.vector(), self.temp2.vector())
            return self.temp1.vector()[:]
        else:
            assert False, "method must be 'FEniCS' or 'numpy'"
        
    def precondition(self, m_vec):
        # temp = spsl.spsolve(self.K, self.M@spsl.spsolve((self.K).T, self.M@m_vec))
        ## Usually, algorithms need a symmetric matrix.
        ## Here, we drop the last M in prior, e.g., 
        ## For GaussianElliptic2, we calculate K^{-1}MK^{-1}m_vec instead of K^{-1}MK^{-1}M m_vec 
        # temp = spsl.spsolve(self.K, self.M@spsl.spsolve(self.K, self.M@m_vec))
        if self.use_LU == False:
            temp = spsl.spsolve(self.K, self.M@spsl.spsolve(self.K, m_vec))
        elif self.use_LU == True:
            temp = self.luK.solve(m_vec)
            temp = self.M@temp
            temp = self.luK.solve(temp)
        else:
            raise NotImplementedError("use_LU must be True or False")
        return np.array(temp)
    
    def precondition_inv(self, m_vec):
        if self.use_LU == False:
            temp = self.K@spsl.spsolve(self.M, self.K@m_vec)
        elif self.use_LU == True:
            temp = self.luM.solve(self.K@m_vec)
            temp = self.K@temp
        else:
            raise NotImplementedError("use_LU must be True or False")
        # temp = spsl.spsolve(self.M, temp)
        return np.array(temp)

    def pointwise_variance_field(self, xx, yy, method="numpy"):
        '''
        This function evaluate the pointwise variance field in a finite element discretization

        Parameters
        ----------
        xx : list
            [(x_1,y_1), \cdots, (x_N, y_N)]
        yy : list
            [(x_1,y_1), \cdots, (x_M, y_M)]

        Returns: variance field c(xx, yy), a matrix NxM
        -------
        None.

        '''
        
        SN = construct_measurement_matrix(np.array(xx), self.domain.function_space)
        SM = construct_measurement_matrix(np.array(xx), self.domain.function_space)
        SM = np.array((SM.T).todense())
        
        if method == "FEniCS":
            raise NotImplementedError
        elif method == "numpy":
            if self.use_LU == False:
                val = SN@spsl.spsolve(self.K, self.M@spsl.spsolve(self.K, SM))
            elif self.use_LU == True:
                val = self.luK.solve(SM)
                val = self.M@val
                val = self.luK.solve(val)
                val = SN@val 
            else:
                raise NotImplementedError("use_LU must be True or False")
                
            if type(val) == type(self.M):
                val = val.todense()
            return np.array(val)
        else:
            assert False, "method must be numpy or FEniCS (FEniCS has not yet been implemented)"


############################################################################################
class GaussianFiniteRank(object):
    '''
    [1] F. J. Pinski, G. Simpson, A. M. Stuart, H. Weber, 
    Algorithms for Kullback-Leibler approximation of probability measures in 
    infinite dimensions, SIAM J. Sci. Comput., 2015.
    
    [2] T. Bau-Thanh, Q. P. Nguyen, FEM-based discretization-invariant MCMC methods
    for PDE-constrained Bayesian inverse problems, Inverse Problems & Imaging, 2016
    
    Base Gaussian \mathcal{C}_0 = [\alpha (\beta I - \Delta)]^{-s}
    Typical example s = 2
    \mathcal{C}_0 v = \lambda^2 v
    
    Due to the calculations of eigendecompositions, this method may be not suitable
    for large-scale problems. A possible idea: projecting large-scale problems 
    to rough grid and calculate the eigendecompositions. 
    
    domain: the original fine grid
    domain_: the approximate rough grid used to evaluate the eigendecomposition
    !!! Only P1 elements can be employed!!!
    '''
    def __init__(self, domain, domain_=None, mean=None, num_KL=None, 
                 alpha=1.0, beta=1.0, s=2):
        if domain_ is None:
            domain_ = domain
        self.domain = domain
        self.domain_ = domain_
        u_, v_ = fe.TrialFunction(domain.function_space), fe.TestFunction(domain.function_space) 
        M_ = fe.assemble(fe.inner(u_, v_)*fe.dx)
        self.M = trans2spnumpy(M_)
        self.dim_full = self.M.shape[0]
        u_, v_ = fe.TrialFunction(domain_.function_space), fe.TestFunction(domain_.function_space)
        Ms_ = fe.assemble(fe.inner(u_, v_)*fe.dx)
        self.Ms = trans2spnumpy(Ms_)
        Delta_ = fe.assemble(fe.inner(fe.grad(u_), fe.grad(v_))*fe.dx)
        self.Delta = trans2spnumpy(Delta_) 
        self.dim = self.Ms.shape[0]
        if num_KL is None: num_KL = self.dim 
        self.num_KL = num_KL
        self.s = s
        self.K_org = alpha*(self.Delta + beta*self.Ms)
        if mean is None:
            self.mean_vec = np.zeros(self.dim_full)
        else:
            self.mean_vec = fe.interpolate(mean, self.domain.function_space).vector()[:]
        # self.Ms_half = np.eye(self.Ms.diagonal())
        
        ## help function
        self.fun = fe.Function(self.domain.function_space)
        self.fun_ = fe.Function(self.domain_.function_space)
        self.is_eig_available = False
        
        ## construct interpolation matrix
        coor = self.domain_.mesh.coordinates()
        # v2d = fe.vertex_to_dof_map(self.domain_.function_space)
        d2v = fe.dof_to_vertex_map(self.domain_.function_space)
        ## full to small matrix
        self.f2sM = construct_measurement_matrix(coor[d2v], self.domain.function_space)
        
        coor = self.domain.mesh.coordinates()
        # v2d = fe.vertex_to_dof_map(self.domain.function_space)
        d2v = fe.dof_to_vertex_map(self.domain.function_space)
        ## small to full matrix
        self.s2fM = construct_measurement_matrix(coor[d2v], self.domain_.function_space)

    def _K_org_inv_x(self, x):
        return np.array(spsl.spsolve(self.K_org, x)) 
    
    def _K_org_x(self, x):
        return np.array(self.K_org@x)
    
    def _M_x(self, x):
        return np.array(self.Ms@x)
    
    def _Minv_x(self, x):
        return np.array(spsl.spsolve(self.Ms, x))
    
    def _K_org_x_op(self):
        linear_op = spsl.LinearOperator((self.dim, self.dim), matvec=self._K_org_x)
        return linear_op
    
    def _K_org_inv_x_op(self):
        linear_op = spsl.LinearOperator((self.dim, self.dim), matvec=self._K_org_x) 
        return linear_op 
        
    def _M_x_op(self):
        linear_op = spsl.LinearOperator((self.dim, self.dim), matvec=self._M_x)
        return linear_op 
    
    def _Minv_x_op(self):
        linear_op = spsl.LinearOperator((self.dim, self.dim), matvec=self._Minv_x)
        return linear_op
    
    def calculate_eigensystem(self):
        ## calculat the eigensystem of I-\Delta, i.e., M^{-1}K_org V = sigma V
        ## Since the eigensystem calculator usually need the input to be symmetric
        ## matrix, we solve K_org V = sigma M V instead. 
        ## If self.num_gamma not reach self.dim, set l=20 to make sure an explicit
        ## calculations of the eigensystem. 
        
        assert self.num_KL <= self.dim

        self.sigma, self.eigvec_ = sl.eigh(self.K_org.todense(), self.Ms.todense())
        self.lam = np.power(self.sigma, -self.s/2)
        self.log_lam = np.log(self.lam)
        self.eigvec = self.s2fM@self.eigvec_
        
        self.num_KL = len(self.lam)
        self.is_eig_available = True
        
    def update_mean_fun(self, mean_vec):
        self.mean_vec = mean_vec
        
    def generate_sample_zero_mean(self, num_sample=1):
        assert self.is_eig_available == True
        # with torch.no_grad():
            # self.lam[:self.num_gamma] = self.gamma
        if num_sample == 1:
            n = np.random.normal(0, 1, (len(self.lam),))
            val = self.lam*n
        else:
            n = np.random.normal(0, 1, (len(self.lam), num_sample))
            val = np.diag(self.lam)@n
            
        val = self.eigvec@val
        
        return np.array(val)
    
    def generate_sample(self, num_sample=1):
        assert self.is_eig_available == True
        
        if num_sample == 1:
            val = self.mean_vec + self.generate_sample_zero_mean(num_sample=num_sample)
        else:
            val = self.mean_vec.reshape(-1,1) + self.generate_sample_zero_mean(num_sample=num_sample) 
            
        return np.array(val)
    
    def evaluate_CM_inner(self, u, v=None):
        if v is None:
            v = u
       
        mean_vec = self.f2sM@self.mean_vec 
        v = self.f2sM@v
        u = self.f2sM@u
        res = v - mean_vec
        val = self.Ms@res
        val = self.eigvec_.T@val
        lam_n2 = np.power(self.lam, -2)
        # self.Lam = Lam
        val = lam_n2*val
        val = self.eigvec_@val 
        val = self.Ms@val
        val = (u - mean_vec)@val
        
        return val
            
    def evaluate_grad(self, u_vec):
        assert type(u_vec) is np.ndarray
        
        u_vec = self.f2sM@u_vec
        val = self.eigvec_.T@self.Ms@u_vec
        lam_n2 = np.power(self.lam, -2)
        val = lam_n2*val
        val = self.eigvec_@val
        val = self.s2fM@val
        
        return np.array(val) 
    
    def evaluate_hessian_vec(self, u_vec):
        return self.evaluate_grad(u_vec)
    
    def precondition(self, m_vec):
        m_vec = self.f2sM@m_vec
        val = self.eigvec_.T@np.array(m_vec)
        lam_n2 = np.power(self.lam, 2) 
        val = lam_n2*val
        val = self.eigvec_@val
        val = self.s2fM@val
        
        return np.array(val)
    
    def precondition_inv(self, m_vec):
        m_vec = np.array(self.f2sM@m_vec)
        val = self.Ms@m_vec
        val = self.eigvec_.T@val
        lam_n2 = np.power(self.lam, -2)
        val = lam_n2*val
        val = self.eigvec_@val
        val = self.Ms@val
        val = self.s2fM@val
        
        return np.array(val)




