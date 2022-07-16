#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 21:51:36 2022

@author: jjx323
"""

import numpy as np
import fenics as fe
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
from core.model import Domain2D
from core.probability import GaussianElliptic2
from core.sample import pCN
from core.noise import NoiseGaussianIID
from core.misc import load_expre

from SteadyStateDarcyFlow.common import EquSolver, ModelDarcyFlow


## set data and result dir
DATA_DIR = './DATA/'
RESULT_DIR = './RESULT/pCN/'
noise_level = 0.01

## set the step length of MCMC
beta = 0.01
## set the total number of the sampling
length_total = 1e5

## domain for solving PDE
equ_nx = 100
domain_equ = Domain2D(nx=equ_nx, ny=equ_nx, mesh_type='CG', mesh_order=1)

## loading the truth for testing algorithms 
mesh_truth = fe.Mesh(DATA_DIR + 'saved_mesh_truth.xml')
V_truth = fe.FunctionSpace(mesh_truth, 'CG', 1)
truth_fun = fe.Function(V_truth, DATA_DIR + 'truth_fun.xml')
truth_fun = fe.interpolate(truth_fun, domain_equ.function_space)

## setting the prior measure
prior_measure = GaussianElliptic2(
    domain=domain_equ, alpha=1, a_fun=1, theta=0.1, boundary='Neumann'
    )

## loading coordinates of the measurement points
measurement_points = np.load(DATA_DIR + "measurement_points_2D.npy")

## setting the forward problem
f_expre = load_expre(DATA_DIR + 'f_2D.txt')
f = fe.interpolate(fe.Expression(f_expre, degree=2), domain_equ.function_space)

equ_solver = EquSolver(domain_equ=domain_equ, m=truth_fun, f=f, points=measurement_points)

## load the measurement data
d = np.load(DATA_DIR + "measurement_noise_2D" + "_" + str(noise_level) + ".npy")
d_clean = np.load(DATA_DIR + "measurement_clean_2D.npy")

## setting the noise
noise_level_ = noise_level*max(d_clean)
noise = NoiseGaussianIID(dim=len(measurement_points))
noise.set_parameters(variance=noise_level_**2)

## setting the Model
model = ModelDarcyFlow(
    d=d, domain_equ=domain_equ, prior=prior_measure, 
    noise=noise, equ_solver=equ_solver
    )

## define the function phi used in pCN
def phi(u_vec):
    model.update_m(u_vec.flatten(), update_sol=True)
    return model.loss_residual()

## set the path for saving results
samples_file_path = RESULT_DIR + 'beta_' + str(beta) + '/samples/'
os.makedirs(samples_file_path, exist_ok=True)
draw_path = RESULT_DIR + 'beta_' + str(beta) + '/draws/'
os.makedirs(draw_path, exist_ok=True)

## set pCN 
pcn = pCN(prior=model.prior, phi=phi, beta=beta, save_num=np.int64(1e3), path=samples_file_path)

## extract information from the chain to see how the algorithm works
global num_
num_ = 0
class CallBack(object):
    def __init__(self, num_=0, function_space=domain_equ.function_space, truth=truth_fun,
                 save_path=draw_path, length_total=length_total, phi=phi):
        self.num_ = num_
        self.fun = fe.Function(function_space)
        self.truth = truth
        self.save_path = save_path
        self.num_fre = 1000
        self.length_total = length_total
        self.phi = phi
        
    def callback_fun(self, params):
        # params = [uk, iter_num, accept_rate, accept_num]
        num = params[1]
        if num % self.num_fre == 0:
            print("-"*70)
            print('Iteration number = %d/%d' % (num, self.length_total), end='; ')
            print('Accept rate = %4.4f percent' % (params[2]*100), end='; ')
            print('Accept num = %d/%d' % (params[3] - self.num_, self.num_fre), end='; ')
            self.num_ = params[3]
            print('Phi = %4.4f' % self.phi(params[0]))
        
            self.fun.vector()[:] = params[0]
            fe.plot(self.truth, label='Truth')
            fe.plot(self.fun, label='Estimation')
            plt.legend()
            plt.show(block=False)
            plt.savefig(self.save_path + 'fig' + str(num) + '.png')
            plt.close()
 
callback = CallBack()

## start sampling
acc_rate, samples_file_path, _ = pcn.generate_chain(length_total=length_total, 
                                                    callback=callback.callback_fun)

## plot the trace of sampling function u
path_samples = samples_file_path
num_total = np.int64(len(os.listdir(path_samples)))
num_start = 0
trace_u = []

for i in range(num_start, num_total):
    temp = np.load(path_samples + 'sample_' + str(i) + '.npy')
    for data in temp:
        trace_u.append(data[10])
    
plt.plot(trace_u)
plt.show()











