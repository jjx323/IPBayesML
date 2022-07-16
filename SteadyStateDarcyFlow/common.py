#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 21:45:50 2019

@author: jjx323
"""
import numpy as np
from scipy.special import gamma
import scipy.linalg as sl
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import fenics as fe
import dolfin as dl 

import sys, os
sys.path.append(os.pardir)
from core.probability import GaussianElliptic2
from core.model import ModelBase
from core.misc import my_project, trans2spnumpy, \
                      construct_measurement_matrix, make_symmetrize

    
###########################################################################
class EquSolver(object):
    def __init__(self, domain_equ, f, m, points):
        self.domain_equ = domain_equ
        self.V_equ = self.domain_equ.function_space
        self.mm = fe.interpolate(m, self.V_equ)
        self.exp_m = fe.Function(self.V_equ)
        self.exp_m.vector()[:] = my_project(dl.exp(self.mm), self.V_equ, flag='only_vec')
        self.f = fe.interpolate(f, self.V_equ)
        self.points = points
        
        self.u_, self.v_ = fe.TrialFunction(self.V_equ), fe.TestFunction(self.V_equ)
        self.K_ = fe.assemble(fe.inner(self.exp_m*fe.grad(self.u_), fe.grad(self.v_))*fe.dx)
        self.F_ = fe.assemble(self.f*self.v_*fe.dx)
        self.M_ = fe.assemble(fe.inner(self.u_, self.v_)*fe.dx)
        
        def boundary(x, on_boundary):
            return on_boundary
        
        self.bc = fe.DirichletBC(self.V_equ, fe.Constant('0.0'), boundary)
        self.bc.apply(self.K_)
        self.bc.apply(self.F_)
        
        self.K = trans2spnumpy(self.K_)
        self.M = trans2spnumpy(self.M_)
        self.F = self.F_[:]
        
        self.S = construct_measurement_matrix(points, self.V_equ)
        
        ## All of the program did not highly rely on FEniCS, 
        ## so the following FEniCS function will be treated only as helpping function
        self.sol_forward = fe.Function(self.V_equ)
        self.sol_adjoint = fe.Function(self.V_equ)
        self.sol_incremental = fe.Function(self.V_equ)
        self.sol_incremental_adjoint = fe.Function(self.V_equ)
        self.Fs = fe.Function(self.V_equ)
        self.m_hat = fe.Function(self.V_equ)
        
        ## All of the solutions will be treated as the solution interact with 
        ## other program
        self.sol_forward_vec = self.sol_forward.vector()[:]
        self.sol_adjoint_vec = self.sol_adjoint.vector()[:]
        self.sol_incremental_vec = self.sol_incremental.vector()[:]
        self.sol_incremental_adjoint_vec = self.sol_incremental_adjoint.vector()[:]
    
    def update_m(self, m_vec):
        self.mm.vector()[:] = np.array(m_vec)
        self.exp_m = fe.Function(self.V_equ)
        # self.exp_m.vector()[:] = fe.project(dl.exp(self.mm), self.V_equ).vector()[:]
        self.exp_m.vector()[:] = my_project(dl.exp(self.mm), self.V_equ, flag='only_vec')
        
        self.K_ = fe.assemble(fe.inner(self.exp_m*fe.grad(self.u_), fe.grad(self.v_))*fe.dx)
        self.bc.apply(self.K_)
        self.K = trans2spnumpy(self.K_)
        
    def get_data(self):
        val = self.S@self.sol_forward.vector()[:]
        return np.array(val) 

    def forward_solver(self, m_vec=None, method='numpy'):
        if type(m_vec) != type(None):
            self.update_m(m_vec)
    
        if method == 'FEniCS':
            fe.solve(self.K_, self.sol_forward.vector(), self.F_)
            self.sol_forward_vec = np.array(self.sol_forward.vector()[:])
        elif method == 'numpy':
            self.sol_forward_vec = spsl.spsolve(self.K, self.F)
            self.sol_forward_vec = np.array(self.sol_forward_vec)
            
        return self.sol_forward_vec
        
    def adjoint_solver(self, vec, m_vec=None, method='numpy'):
        if type(m_vec) != type(None):
            self.update_m(m_vec)
            
        Fs = -self.S.T@vec
        
        if method == 'FEniCS':
            self.Fs.vector()[:] = Fs
            fe.solve(self.K_, self.sol_adjoint.vector(), self.Fs.vector())
            self.sol_adjoint_vec = np.array(self.sol_adjoint.vector()[:])
        elif method == 'numpy':
            self.sol_adjoint_vec = np.array(spsl.spsolve(self.K, Fs))
            
        return self.sol_adjoint_vec
  
    def incremental_forward_solver(self, m_hat, sol_forward=None, method='numpy'):
        if type(sol_forward) == type(None):
            self.sol_forward.vector()[:] = self.sol_forward_vec 
        
        if method == 'FEniCS':
            self.m_hat.vector()[:] = np.array(m_hat)
            b_ = -fe.assemble(fe.inner(self.m_hat*self.exp_m*fe.grad(self.sol_forward), fe.grad(self.v_))*fe.dx)
            self.bc.apply(b_)
            fe.solve(self.K_, self.sol_incremental.vector(), b_)
            self.sol_incremental_vec = np.array(self.sol_incremental.vector()[:])
        elif method == 'numpy':
            b_ = fe.inner(self.exp_m*fe.grad(self.sol_forward)*self.u_, fe.grad(self.v_))*fe.dx
            b_ = fe.assemble(b_)   
            b_spnumpy = trans2spnumpy(b_)
            b = b_spnumpy@m_hat
            self.sol_incremental_vec = np.array(spsl.spsolve(self.K, -b))
            
        return self.sol_incremental_vec     
        
    def incremental_adjoint_solver(self, vec, m_hat, sol_adjoint=None, simple=False, method='numpy'):
        if type(sol_adjoint) == type(None):
            self.sol_adjoint.vector()[:] = self.sol_adjoint_vec 
            
        Fs = -self.S.T@vec
        if simple == False:
            if method == 'FEniCS':
                self.m_hat.vector()[:] = np.array(m_hat)
                bl_ = fe.assemble(fe.inner(self.m_hat*self.exp_m*fe.grad(self.sol_adjoint), fe.grad(self.v_))*fe.dx)
                self.Fs.vector()[:] = Fs
                fe.solve(self.K_, self.sol_incremental_adjoint.vector(), -bl_ + self.Fs.vector())
                self.sol_incremental_adjoint_vec = np.array(self.sol_incremental_adjoint.vector()[:])
            elif method == 'numpy':
                bl_ = fe.assemble(fe.inner(self.exp_m*fe.grad(self.sol_adjoint)*self.u_, fe.grad(self.v_))*fe.dx)
                bl_spnumpy = trans2spnumpy(bl_)
                bl = bl_spnumpy@m_hat
                self.sol_incremental_adjoint_vec = spsl.spsolve(self.K, -bl + Fs)
        elif simple == True:
            if method == 'FEniCS':
                self.Fs.vector()[:] = Fs
                fe.solve(self.K_, self.sol_incremental_adjoint.vector(), self.Fs.vector())
                self.sol_incremental_adjoint_vec = np.array(self.sol_incremental_adjoint.vector()[:])
            elif method == 'numpy':
                val = spsl.spsolve(self.K, Fs)
                self.sol_incremental_adjoint_vec = np.array(val)
                
        return self.sol_incremental_adjoint_vec

        
###########################################################################
class ModelDarcyFlow(ModelBase):
    def __init__(self, d, domain_equ, prior, noise, equ_solver):
        super().__init__(d, domain_equ, prior, noise, equ_solver)
        self.p = fe.Function(self.equ_solver.domain_equ.function_space)
        self.q = fe.Function(self.equ_solver.domain_equ.function_space)
        self.pp = fe.Function(self.equ_solver.domain_equ.function_space)
        self.qq = fe.Function(self.equ_solver.domain_equ.function_space)
        self.u_ = fe.TrialFunction(self.domain_equ.function_space)
        self.v_ = fe.TestFunction(self.domain_equ.function_space)
        self.m_hat = fe.Function(self.domain_equ.function_space)
        self.m = self.equ_solver.mm
        self.loss_residual_now = 0

    def update_m(self, m_vec, update_sol=True):
        self.equ_solver.update_m(m_vec)
        if update_sol == True:
            self.equ_solver.forward_solver()
    
    def updata_d(self, d):
        self.d = d
        
    def _time_noise_precision(self, vec):
        if type(self.noise.precision) != type(None):
            val = self.noise.precision@vec
        else:
            val = spsl.spsolve(self.noise.covariance, vec)
        return np.array(val)
        
    def loss_residual(self):
        temp = (self.S@self.equ_solver.sol_forward_vec - self.noise.mean - self.d)
        if type(self.noise.precision) != type(None): 
            temp = temp@(self.noise.precision)@temp
        else:
            temp = temp@(spsl.spsolve(self.noise.covariance, temp))
        self.loss_residual_now = 0.5*temp
        return self.loss_residual_now
    
    def loss_residual_L2(self):
        temp = (self.S@self.equ_solver.sol_forward_vec - self.d)
        temp = temp@temp
        return 0.5*temp
    
    def eval_grad_residual(self, m_vec):
        self.update_m(m_vec, update_sol=False)
        self.equ_solver.forward_solver()
        vec = self.S@self.equ_solver.sol_forward_vec - self.noise.mean - self.d
        vec = self._time_noise_precision(vec) 
        self.equ_solver.adjoint_solver(vec)
        self.p.vector()[:] = self.equ_solver.sol_forward_vec
        self.q.vector()[:] = self.equ_solver.sol_adjoint_vec
        b_ = fe.assemble(fe.inner(fe.grad(self.q), self.equ_solver.exp_m*fe.grad(self.p)*self.v_)*fe.dx)
        return spsl.spsolve(self.equ_solver.M, b_[:])
        
    def eval_hessian_res_vec(self, m_hat_vec):
        # self.m_hat.vector()[:] = m_hat_vec
        self.equ_solver.incremental_forward_solver(m_hat_vec)
        vec = self.S@self.equ_solver.sol_incremental_vec
        vec = self._time_noise_precision(vec)        
        self.equ_solver.incremental_adjoint_solver(vec, m_hat_vec, simple=False)
        self.pp.vector()[:] = self.equ_solver.sol_incremental_vec
        self.qq.vector()[:] = self.equ_solver.sol_incremental_adjoint_vec
        A1 = fe.assemble(fe.inner(self.m_hat*self.equ_solver.exp_m*fe.grad(self.p)*self.v_, \
                                  fe.grad(self.q))*fe.dx)
        A2 = fe.assemble(fe.inner(self.equ_solver.exp_m*fe.grad(self.p)*self.v_, 
                                  fe.grad(self.qq))*fe.dx)
                         # fe.grad(self.equ_solver.sol_incremental_adjoint))*fe.dx)
        A3 = fe.assemble(fe.inner(self.equ_solver.exp_m*fe.grad(self.q)*self.v_,
                                  fe.grad(self.pp))*fe.dx)
                         # fe.grad(self.equ_solver.sol_incremental))*fe.dx)
        
        A = A1[:] + A2[:] + A3[:]
        
        return spsl.spsolve(self.equ_solver.M, A)


            







