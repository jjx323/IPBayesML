#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 21:03:32 2022

@author: jjx323
"""

# import numpy as np
import scipy.sparse.linalg as spsl
# import scipy.sparse as sps

import fenics as fe 
# import dolfin as dl

from core.misc import trans2spnumpy

#############################################################################
class Domain(object):
    def __init__(self, mesh_type='P', mesh_order=2):
        self.mesh_type = mesh_type
        self.mesh_order = mesh_order
        self._mesh = None
        self._function_space = None
        
    @property
    def mesh(self):
        return self._mesh
    
    @property
    def function_space(self):
        return self._function_space
    
    
class Domain2D(Domain):
    '''
    class Domain has to properties: mesh, function_space
    mesh: a square domain with uniform mesh
    function_space: can be specified by 'mesh_type' and 'mesh_order'
    '''
    def __init__(self, low_point=[0, 0], high_point=[1, 1], nx=100, ny=100, mesh_type='P', mesh_order=2):
        super().__init__(mesh_type, mesh_order)
        self._mesh = fe.RectangleMesh(fe.Point(low_point[0], low_point[1]), \
                                     fe.Point(high_point[0], high_point[1]), nx, ny)
        self._function_space = fe.FunctionSpace(self._mesh, self.mesh_type, self.mesh_order)
    
    def update(self, low_point=[0, 0], high_point=[1, 1], nx=100, ny=100, mesh_type='P', mesh_order=2):
        self.mesh_type = mesh_type
        self.mesh_order = mesh_order
        self._mesh = fe.RectangleMesh(fe.Point(low_point[0], low_point[1]), \
                                     fe.Point(high_point[0], high_point[1]), nx, ny)
        self._function_space = fe.FunctionSpace(self._mesh, mesh_type, mesh_order)
    

class Domain1D(Domain):
    def __init__(self, low_point=0, high_point=1, n=100, mesh_type='P', mesh_order=2):
        super().__init__(mesh_type, mesh_order)
        self._mesh = fe.IntervalMesh(n, low_point, high_point)
        self._function_space = fe.FunctionSpace(self._mesh, self.mesh_type, self.mesh_order)
        
    def update(self, low_point=0, high_point=1, n=100, mesh_type='P', mesh_order=2):
        self.mesh_type = mesh_type
        self.mesh_order = mesh_order
        self._mesh = fe.IntervalMesh(n, low_point, high_point)
        self._function_space = fe.FunctionSpace(self._mesh, mesh_type, mesh_order)
        

###########################################################################
class ModelBase(object):
    def __init__(self, d, domain_equ, prior, noise, equ_solver):
        self.d = d
        self.domain_equ = domain_equ
        self.prior = prior
        self.noise = noise 
        self.equ_solver = equ_solver 
        self._initialization()
        
    def _initialization(self):
        self.p = fe.Function(self.domain_equ.function_space)
        self.q = fe.Function(self.domain_equ.function_space)
        self.m = fe.Function(self.domain_equ.function_space)
        self.grad_residual, self.grad_prior = None, None
        self.hessian_residual, self.hessian_prior = None, None
        self.S = self.equ_solver.S
        u_ = fe.TrialFunction(self.domain_equ.function_space)
        v_ = fe.TestFunction(self.domain_equ.function_space)
        self.M_equ_ = fe.assemble(fe.inner(u_, v_)*fe.dx)
        self.M_equ = trans2spnumpy(self.M_equ_)
        u_ = fe.TrialFunction(self.domain_equ.function_space)
        v_ = fe.TestFunction(self.domain_equ.function_space)
        self.M_ = fe.assemble(fe.inner(u_, v_)*fe.dx)
        self.M = trans2spnumpy(self.M_)
        temp_fun = fe.Function(self.domain_equ.function_space)
        self.geometric_dim = temp_fun.geometric_dimension()

    def update_m(self, m_vec, update_sol):
        raise NotImplementedError

    def update_noise(self):
        raise NotImplementedError
    
    def update_d(self, d):
        self.d = d.copy()
        
    def loss_residual(self):
        temp = (self.S@self.p.vector()[:] - self.d)
        temp = temp@temp
        return 0.5*temp
    
    def loss_prior(self):
        return 0.5*self.prior.evaluate_CM_inner(self.m.vector()[:], self.m.vector()[:])
    
    def loss(self):
        loss_res = self.loss_residual()
        loss_prior = self.loss_prior()
        return loss_res + loss_prior, loss_res, loss_prior
        
    def eval_grad_residual(self, m_vec):
        raise NotImplementedError
    
    def eval_grad_prior(self, m_vec):
        return self.prior.evaluate_grad(m_vec)

    def gradient(self, m_vec):
        grad_res = self.eval_grad_residual(m_vec)
        grad_prior = self.eval_grad_prior(m_vec)
        return grad_res + grad_prior, grad_res, grad_prior
        
    def eval_hessian_res_vec(self, m_hat_vec):
        raise NotImplementedError
        
    def eval_hessian_prior_vec(self, m_vec):
        self.hessian_prior = self.prior.evaluate_hessian_vec(m_vec)
        return self.hessian_prior
    
    def hessian(self, m_vec):
        hessian_res = self.eval_hessian_res_vec(m_vec)
        hessian_prior = self.eval_hessian_prior_vec(m_vec)
        return hessian_res + hessian_prior
        
    def hessian_linear_operator(self):
        leng = self.M.shape[0]
        linear_ope = spsl.LinearOperator((leng, leng), matvec=self.hessian)
        return linear_ope
    
    def MxHessian(self, m_vec):
        ## Usually, algorithms need a symmetric matrix.
        ## Here, we calculate MxHessian to make a symmetric matrix. 
        return self.M@self.hessian(m_vec)
    
    def MxHessian_linear_operator(self):
        leng = self.M.shape[0]
        linear_op = spsl.LinearOperator((leng, leng), matvec=self.MxHessian)
        return linear_op
    
    def precondition(self, m_vec):
        return self.prior.precondition(m_vec)
    
    def precondition_linear_operator(self):
        leng = self.M.shape[0]
        linear_ope = spsl.LinearOperator((leng, leng), matvec=self.precondition)
        return linear_ope
    
    def precondition_inv(self, m_vec):
        return self.prior.precondition_inv(m_vec)
    
    def precondition_inv_linear_operator(self):
        leng = self.M.shape[0]
        linear_ope = spsl.LinearOperator((leng, leng), matvec=self.precondition_inv)
        return linear_ope

    