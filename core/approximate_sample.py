#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 16:48:26 2022

@author: jjx323
"""

import numpy as np
import fenics as fe
import scipy.sparse.linalg as spsl
import scipy.sparse as sps

from core.eigensystem import double_pass
from core.misc import construct_measurement_matrix, smoothing
from core.optimizer import NewtonCG


class LaplaceApproximate(object):
    '''
    Ref: T. Bui-Thanh, O. Ghattas, J. Martin, G. Stadler, 
    A computational framework for infinite-dimensional Bayesian inverse problems
    part I: The linearized case, with application to global seismic inversion,
    SIAM J. Sci. Comput., 2013
    '''
    def __init__(self, model):
        
        assert hasattr(model, "prior") and hasattr(model, "domain_equ")
        assert hasattr(model, "equ_solver") and hasattr(model, "noise")
        assert hasattr(model, "M") and hasattr(model, "S")
        
        self.fun_dim = model.domain_equ.function_space.dim()
        self.prior = model.prior
        self.equ_solver = model.equ_solver
        self.noise = model.noise 
        self.M = sps.csc_matrix(model.M)
        self.M_half = sps.csc_matrix(sps.diags(np.sqrt(self.M.diagonal())))
        self.Minv_half = sps.csc_matrix(sps.diags(np.sqrt(1/self.M.diagonal())))
        self.S = model.S
        
    def set_mean(self, vec):
        self.mean = np.array(vec)
    
    ## linearized_forward_solver is actually the incremental forward solver
    def _linearized_forward_solver(self, m_hat, **kwargs):
        val = self.equ_solver.incremental_forward_solver(m_hat, **kwargs)
        return np.array(val)
    
    ## linearized_adjoint_solver is actually the incremental adjoint solver
    def _linearized_adjoint_solver(self, vec, m_hat, **kwargs):
        val = self.equ_solver.incremental_adjoint_solver(vec, m_hat, **kwargs)
        return np.array(val)
    
    def _time_noise_precision(self, vec):
        if type(self.noise.precision) != type(None): 
            vec = self.noise.precision@(vec)
        else:
            vec = spsl.spsolve(self.noise.covariance, vec)
        return np.array(vec)
      
    ## Since the symmetric matrixes are need for computing eigensystem 
    ## F^{*}F = M^{-1} F^T F is not a symmetric matrix, so we multiply M.
    ## The following function actually evaluate M F^{*}F = F^T F
    def eva_Hessian_misfit_M(self):
        leng = self.M.shape[0]
        self.linear_ope = spsl.LinearOperator((leng, leng), matvec=self._eva_Hessian_misfit_M)
        return self.linear_ope
    
    def _eva_Hessian_misfit_M(self, vec):
        vec = np.squeeze(vec)
        val = self._linearized_forward_solver(vec)
        val = self._time_noise_precision(self.S@val)
        # print("-------------------------------------------")
        # print(val.shape, vec.shape)
        val = self._linearized_adjoint_solver(val, vec)
        return np.array(self.M@val)
    
    def _eva_prior_var_inv_M(self, vec):
        val = self.prior.K@spsl.spsolve(self.M, self.prior.K@vec)
        return np.array(val)
    
    ## K M^{-1} K as eva_Hessian_misfit_M, we also multiply M
    def eva_prior_var_inv_M(self):
        leng = self.M.shape[0]
        self.linear_ope = spsl.LinearOperator((leng, leng), matvec=self._eva_prior_var_inv_M)
        return self.linear_ope
    
    def _eva_prior_var_M(self, vec):
        val = spsl.spsolve(self.prior.K, vec)
        val = spsl.spsolve(self.prior.K, self.M@val)
        return np.array(val)
    
    ## K^{-1} M K^{-1}
    def eva_prior_var_M(self):
        leng = self.M.shape[0]
        self.linear_ope = spsl.LinearOperator((leng, leng), matvec=self._eva_prior_var_M)
        return self.linear_ope
        
    def calculate_eigensystem(self, num_eigval, method='double_pass', 
                                       oversampling_factor=20, **kwargs):
        '''
        Calculate the eigensystem of H_{misfit} v = \lambda \Gamma^{-1} v.
        (\Gamma is the prior covariance operator)
        The related matrixes (H_{misfit} and \Gamma) are not symmetric, 
        however, the standard eigen-system computing algorithm need these matrixes 
        to be symmetric. Hence, we actually compute the following problem:
                M H_{misfit} v = \lambda M \Gamma^{-1} v

        Parameters
        ----------
        num_eigval : int
            calucalte the first num_eigval number of large eigenvalues
        method : str, optional
            double_pass and scipy_eigsh can be choien. The default is 'double_pass'.
        oversampling_factor : int, optional
            To ensure an accurate calculation of the required eigenvalues. The default is 20.
        **kwargs : TYPE
            Depends on which method is employed.

        Returns
        -------
        None.
        
        The computated eignvalue will be in a descending order.
        '''
    
        if method == 'double_pass':
            Hessian_misfit = self._eva_Hessian_misfit_M
            prior_var_inv = self._eva_prior_var_inv_M
            prior_var = self._eva_prior_var_M
            rs = num_eigval + oversampling_factor
            omega = np.random.randn(self.M.shape[0], rs)
            self.eigval, self.eigvec = double_pass(
                Hessian_misfit, M=prior_var_inv, Minv=prior_var, Mg=self.M,
                omega=omega, r=num_eigval, l=oversampling_factor,
                )
        elif method == 'single_pass':
            raise NotImplementedError
        elif method == 'scipy_eigsh':
            ## The eigsh function in scipy seems much slower than the "double_pass" and 
            ## "single_pass" algorithms implemented in the package. The eigsh function 
            ## is kept as a baseline for testing our implementations. 
            Hessian_misfit = self.eva_Hessian_misfit_M()
            prior_var_inv = self.eva_prior_var_inv_M()
            prior_var = self.eva_prior_var_M()
            self.eigval, self.eigvec = spsl.eigsh(
                Hessian_misfit, M=prior_var_inv, k=num_eigval+oversampling_factor, which='LM', 
                Minv=prior_var, **kwargs
                )  #  optional parameters: maxiter=maxiter, v0=v0 (initial)
            index = self.eigval > 0.9
            if np.sum(index) == num_eigval:
                print("Warring! The eigensystem may be inaccurate!")
            self.eigval = np.flip(self.eigval[index])
            self.eigvec = np.flip(self.eigvec[:, index], axis=1)
        else:
            assert False, "method should be double_pass, scipy_eigsh"
        
        ## In the above, we actually solve the H v1 = \lambda \Gamma^{-1} v1
        ## However, we actually need to solve \Gamma^{1/2} H \Gamma^{1/2} v = \lambda v
        ## Notice that v1 \neq v, we have v = \Gamma^{-1/2} v
        self.eigvec = spsl.spsolve(self.M, self.prior.K@self.eigvec)
        
    def posterior_var_times_vec(self, vec):
        dr = self.eigval/(self.eigval + 1.0)
        Dr = sps.csc_matrix(sps.diags(dr))
        val1 = spsl.spsolve(self.prior.K, self.M@vec)
        val2 = self.eigvec@Dr@self.eigvec.T@self.M@val1
        val = self.M@(val1 - val2)
        val = spsl.spsolve(self.prior.K, val)
        return np.array(val)        
        
    def generate_sample(self):
        assert hasattr(self, "mean") and hasattr(self, "eigval") and hasattr(self, "eigvec")
        n = np.random.normal(0, 1, (self.fun_dim,))
        val1 = self.Minv_half@n
        pr = 1.0/np.sqrt(self.eigval+1.0) - 1.0
        Pr = sps.csc_matrix(sps.diags(pr))
        val2 = self.eigvec@Pr@self.eigvec.T@self.M@val1
        val = self.M@(val1 + val2)
        val = spsl.spsolve(self.prior.K, val)
        val = self.mean + val
        return np.array(val)
        
    def pointwise_variance_field(self, xx, yy):
        '''
        Calculate the pointwise variance field of the posterior measure. 
        Through a simple calculations, we can find the following formula:
            c_h(xx, yy) = \Phi(xx)^T[K^{-1}MK^{-1]} - K^{-1}MV_r D_r V_r^T M K^{-1}]\Phi(yy),
        which is actually the same as the formula (5.7) in the following paper: 
            A computational framework for infinite-dimensional Bayesian inverse problems
            part I: The linearized case, with application to global seismic inversion,
            SIAM J. Sci. Comput., 2013

        Parameters
        ----------
        xx : list
            [(x_1,y_1), \cdots, (x_N, y_N)]
        yy : list
            [(x_1,y_1), \cdots, (x_M, y_M)]

        Returns
        -------
        None.

        '''
        assert hasattr(self, "eigval") and hasattr(self, "eigvec")
        
        SN = construct_measurement_matrix(np.array(xx), self.prior.domain.function_space)
        SM = construct_measurement_matrix(np.array(xx), self.prior.domain.function_space)
        SM = SM.T
        
        val = spsl.spsolve(self.prior.K, SM)
        dr = self.eigval/(self.eigval + 1.0)
        Dr = sps.csc_matrix(sps.diags(dr))
        val1 = self.eigvec@Dr@self.eigvec.T@self.M@val
        val = self.M@(val - val1)
        val = spsl.spsolve(self.prior.K, val)
        val = SN@val        
        
        if type(val) == type(self.M):
            val = val.todense()
        
        return np.array(val)


#############################################################################

class rMAP(object):
    def __init__(self, model):
        '''
        Ref: K. Wang, T. Bui-Thanh, O. Ghattas, 
        A randomized maximum a posteriori method for posterior sampling of high
        dimensional nonlinear Bayesian inverse problems, SIAM J. Sci. Comput., 2018
        '''
        self.model = model
        self.prior = model.prior 
        self.noise = model.noise
        self.optim = NewtonCG(model=model)
        assert hasattr(self.prior, "generate_sample_zero_mean")
        assert hasattr(self.noise, "generate_sample_zero_mean")
        
        self.d = self.model.d.copy()
        self.prior_mean = self.model.prior.mean_vec
        self.K = self.model.prior.K
        self.M = self.model.prior.M
        if hasattr(self.noise, "precision"):
            self.Linv = self.noise.precision
            self.L = self.noise.covariance
        else:
            self.L = self.noise.covariance
            
        ## helping fun
        self.fun = fe.Function(self.model.domain_equ.function_space)
        
    def generate_random_element(self):
        epsilon = self.prior.generate_sample_zero_mean()
        theta  = self.noise.generate_sample_zero_mean()
        return epsilon, theta
    
    def optimize(self, m0=None, max_iter=100, cg_max=1000, method="cg_my", eta=1e-2, 
                 callback=None):
        self.optim.re_init(m0)
        pre_cost = self.optim.cost
        
        for itr in range(max_iter):
            self.optim.descent_direction(cg_max=cg_max, method=method)
            self.optim.step(method='armijo', show_step=False)
            if self.optim.converged == False:
                break
            if callback is not None:
                callback(itr, self.model)
            if np.abs(pre_cost - self.optim.cost) < eta*np.abs(pre_cost):
                break
            pre_cost = self.optim.cost.copy()
            
        return np.array(self.optim.mk), itr
    
    def calculate_MAP(self, **kwargs):
        self.model.d = self.d.copy()
        self.model.prior.mean_vec = self.prior_mean 
        self.map_point = self.optimize(**kwargs)[0]
        
    def calculate_Laplace_approximate(self, num_eigval=30):
        self.laplace_approximate = LaplaceApproximate(self.model)
        self.laplace_approximate.calculate_eigensystem(num_eigval=num_eigval, method="double_pass")
        assert hasattr(self, "map_point")
        self.laplace_approximate.set_mean(self.map_point)
        
    def prepare(self, num_eigval=None, **kwargs):
        self.calculate_MAP(**kwargs)
        if type(num_eigval) != type(None):
            self.calculate_Laplace_approximate(num_eigval=num_eigval)
    
    def random_optim(self, i=0, d_hat=None, u_hat=None, m0=None, **kwargs):
        np.random.seed(i)
        epsilon, theta = self.generate_random_element()
        self.model.d = self.d + theta
        self.model.prior.mean_vec = self.prior_mean + epsilon
        
        ## find the init value
        ## There are a lot of problems need to be clarified !!!
        if m0 is None:
            if d_hat is not None:
                assert False, "This choice is not correctly implemented"
                d_tilde = self.model.d - d_hat
                u_tilde = self.model.prior.mean_vec - u_hat
                ## calculate C^{-1} \tilde{u}, C^{-1} = M^{-1}KM^{-1}K
                val1 = spsl.spsolve(self.M, self.K@u_tilde)
                val1 = spsl.spsolve(self.M, self.K@val1)
                ## calculate \gabla G* (\hat{u}_i^{MAP}) L^{-1}\tilde{d}
                val2 = spsl.spsolve(self.L, d_tilde)
                self.model.update_m(u_hat, update_sol=True)
                val2 = self.model.equ_solver.adjoint_solver(val2)
                val = val1 + val2
                ## calculate T: \nabla^2 J(u^{MAP}, u0, d)T = val
                T = self.laplace_approximate.posterior_var_times_vec(val)
                m0 = u_hat + T
                # import matplotlib.pyplot as plt
                # plt.plot(m0)
            else:
                m0 = self.model.prior.mean_vec
        else:
            # self.fun.vector()[:] = self.model.prior.mean_vec
            # smooth_mean = smoothing(self.fun, alpha=0.1).vector()[:]
            m0 = m0 #+ smooth_mean
        
        param = self.optimize(m0=m0, **kwargs)
        mk, final_itr = param[0], param[1]
        
        return mk, self.model.d.copy(), self.model.prior.mean_vec.copy(), final_itr
        
    def generate_sample(self, num_samples=10, num_cores=1, m0=None, **kwargs):
        if num_cores == 1:
            m_samples = []
            for itr in range(num_samples):
                if itr == 0:
                    ## Initialization by previous samples is stated in the article, 
                    ## however, we cannot fully recover the methods correctly at present. 
                    ## For complex problems, we use the MAP estimate as the initial guess. 
                    ## See also "self.random_optim" function. 
                    dhat, uhat = None, None
                mk, _, _, final_itr = self.random_optim(itr, dhat, uhat, m0=m0, **kwargs)
                print("Sample number = %3d, optim itr = %3d" % (itr, final_itr))
                m_samples.append(mk)
            m_samples = np.array(m_samples)
        elif num_cores > 1:
            ## It seems the parallel computation cannot work when the code is 
            ## orgnized by functions in class for FEniCS.
            assert False, "Parallel computation is not implemented correctly"
            from multiprocessing import Pool
            with Pool(num_cores) as p:
                m_samples = p.map(self.random_optim, [i for i in range(num_samples)])
        
        return m_samples
                
    
        













