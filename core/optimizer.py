#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 16:37:54 2022

@author: Junxiong Jia
"""

import numpy as np
import scipy.sparse.linalg as spsl

from core.linear_eq_solver import cg_my

###########################################################################

class OptimBase(object):
    def __init__(self, model, c_armijo=1e-5, it_backtrack=20):
        
        assert hasattr(model, "M")
        assert hasattr(model, "gradient") 
        
        self.model = model
        self.c_armijo = c_armijo
        self.it_backtrack = it_backtrack
        self.M = self.model.M
        self.converged = True
    
    def set_init(self):
        raise NotImplementedError
        
    def armijo_line_search(self, mk, g, cost_pre, show_step=False):
        converged = True
        mk_pre = mk.copy()
        c_armijo = self.c_armijo
        step_length = 1
        backtrack_converged = False
        grad_norm2 = np.sqrt(g@self.M@g)
        for it_backtrack in range(self.it_backtrack):
            mk = mk + step_length*(g)/max(grad_norm2+1e-15, 1)
            self.model.update_m(mk, update_sol=True)
            cost_all = self.model.loss()
            cost_new = cost_all[0]
            if cost_new < cost_pre - step_length*c_armijo*grad_norm2:
                cost_pre = cost_new
                backtrack_converged = True
                break
            else:
                step_length*=0.5
                mk = mk_pre.copy()
            if show_step == True:
                print("search num is ", it_backtrack, " step_length is ", step_length)
        if backtrack_converged == False:
            print("Backtracking failed. A sufficient descent direction was not found")
            converged = False
        return mk, cost_all, converged
    
    def step(self):
        raise NotImplementedError
        
    def gradient(self):
        raise NotImplementedError
    
    
###########################################################################

class GradientDescent(OptimBase):
    def __init__(self, model, mk=None, lr=1e-5):
        super().__init__(model=model)
        
        assert hasattr(model, "prior") and hasattr(model.prior, "mean_vec")
        assert hasattr(model, "update_m") and hasattr(model, "loss")
        
        self.lr = lr
        if mk == None:
            mk = self.model.prior.mean_vec
        self.mk = mk
        self.model.update_m(mk, update_sol=True)
        cost_all = self.model.loss()
        self.cost, self.cost_res, self.cost_prior = cost_all[0], cost_all[1], cost_all[2]
        
    def re_init(self, mk=None):
        if mk is None:
            mk = self.model.prior.mean_vec
        self.mk = mk
        self.model.update_m(mk, update_sol=True)
        cost_all = self.model.loss()
        self.cost, self.cost_res, self.cost_prior = cost_all[0], cost_all[1], cost_all[2]

    def set_init(self, mk):
        self.mk = mk
        
    def descent_direction(self):
        self.model.update_m(self.mk, update_sol=False)
        gg = self.model.gradient(self.mk)
        self.grad, self.grad_res, self.grad_prior = gg[0], gg[1], gg[2]

    def step(self, method='armijo', show_step=False):
        if method == 'armijo':
            self.mk, cost_all, self.converged = self.armijo_line_search(self.mk, -self.grad, 
                                                       self.cost, show_step=show_step)
            self.cost, self.cost_res, self.cost_prior = cost_all[0], cost_all[1], cost_all[2]
        elif method == 'fixed':
            self.mk = self.mk - self.lr*self.grad
        else:
            assert False, "method should be fixed or armijo"
    

###########################################################################

class NewtonCG(OptimBase):
    def __init__(self, model, mk=None, lr=1.0):
        super().__init__(model=model)
        
        assert hasattr(model, "update_m")
        assert hasattr(model, "prior") and hasattr(model.prior, "mean_vec")
        assert hasattr(model, "hessian_linear_operator")
        assert hasattr(model, "precondition_linear_operator")
        assert hasattr(model, "loss") 
        
        self.lr = lr
        if mk is None:
            mk = self.model.prior.mean_vec
        self.mk = mk
        self.model.update_m(mk, update_sol=True)
        cost_all = self.model.loss()
        self.cost, self.cost_res, self.cost_prior = cost_all[0], cost_all[1], cost_all[2]
        
        ## Usually, algorithms need a symmetric matrix.
        ## Here, we calculate MxHessian to make a symmetric matrix. 
        self.hessian_operator = self.model.MxHessian_linear_operator()
        
    def re_init(self, mk=None):
        if mk is None:
            mk = self.model.prior.mean_vec
        self.mk = mk
        self.model.update_m(mk, update_sol=True)
        cost_all = self.model.loss()
        self.cost, self.cost_res, self.cost_prior = cost_all[0], cost_all[1], cost_all[2]
        
    def set_init(self, mk):
        self.mk = mk
        
    def descent_direction(self, cg_tol=None, cg_max=1000, method='cg_my', curvature_detector=False):
        self.model.update_m(self.mk, update_sol=False)
        gg = self.model.gradient(self.mk)
        self.grad, self.grad_res, self.grad_prior = gg[0], gg[1], gg[2]
        
        pre_cond = self.model.precondition_linear_operator()
        
        if cg_tol is None:
            ## if cg_tol is None, specify cg_tol according to the rule in the following paper:
            ## Learning physics-based models from data: perspective from inverse problems
            ## and model reduction, Acta Numerica, 2021. Page: 465-466
            norm_grad = np.sqrt(self.grad@self.M@self.grad)
            cg_tol = min(0.5, np.sqrt(norm_grad))
        atol = 0.1
        ## cg iteration will terminate when norm(residual) <= min(atol, cg_tol*|self.grad|)
             
        if method == 'cg_my':
            ## Note that the M and Minv option of cg_my are designed different to the methods 
            ## in scipy, e.g., Minv should be M in scipy
            ## Here, in order to keep the matrix to be symmetric and positive definite,
            ## we actually need to solve M H g = M (-grad) instead of H g = -grad
            self.g, info, k = cg_my(
                self.hessian_operator, -self.M@self.grad, Minv=pre_cond,
                tol=cg_tol, atol=atol, maxiter=cg_max, curvature_detector=True
                )
            if k == 1:  
                ## If the curvature is negative for the first iterate, use grad as the 
                ## search direction.
                self.g = -self.grad
            # print(info)
        elif method == 'bicgstab':
            ## It seems hardly to incoporate the curvature terminate condition 
            ## into the cg type methods in scipy. However, we keep the implementation.
            ## Since there may be some ways to add curvature terminate condition, and 
            ## cg type methods in scipy can be used as a basedline for testing linear problems.
            self.g, info = spsl.bicgstab(
                self.hessian_operator, -self.M@self.grad, M=pre_cond, tol=cg_tol, atol=atol, maxiter=cg_max,
                callback=None
                )
        elif method == 'cg':
            self.g, info = spsl.cg(
                self.hessian_operator, -self.M@self.grad, M=pre_cond, tol=cg_tol, atol=atol, maxiter=cg_max,
                callback=None
                )
        elif method == 'cgs':
            self.g, info = spsl.cgs(
                self.hessian_operator, -self.M@self.grad, M=pre_cond, tol=cg_tol, atol=atol, maxiter=cg_max,
                callback=None
                )       
        else:
            assert False, "method should be cg, cgs, bicgstab"
        
        self.hessian_terminate_info = info
        
    def step(self, method='armijo', show_step=False):
        if method == 'armijo':
            self.mk, cost_all, self.converged = self.armijo_line_search(self.mk, self.g, 
                                                       self.cost, show_step=show_step)
            self.cost, self.cost_res, self.cost_prior = cost_all[0], cost_all[1], cost_all[2]
        elif method == 'fixed':
            self.mk = self.mk + self.lr*self.g
        else:
            assert False, "method should be fixed or armijo"

    
















