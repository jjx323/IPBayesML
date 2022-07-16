#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 17:36:07 2022

@author: jjx323
"""

import numpy as np
import fenics as fe
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
from core.model import Domain1D
from core.probability import GaussianElliptic2
from core.noise import NoiseGaussianIID
from core.optimizer import GradientDescent, NewtonCG
from core.approximate_sample import rMAP

from SimpleSmooth.common import EquSolver, ModelSS


## set data and result dir
DATA_DIR = './DATA/'
RESULT_DIR = './RESULT/MAP/'
noise_level = 0.05

## domain for solving PDE
equ_nx = 300
domain_equ = Domain1D(n=equ_nx, mesh_type='CG', mesh_order=1)

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
measurement_points = np.load(DATA_DIR + "measurement_points_1D.npy")

## setting the forward problem
alpha = 0.01
equ_solver = EquSolver(domain_equ=domain_equ, alpha=alpha, \
                        points=np.array([measurement_points]).T, m=truth_fun)

## load the measurement data
d = np.load(DATA_DIR + "measurement_noise_1D" + "_" + str(noise_level) + ".npy")
d_clean = np.load(DATA_DIR + "measurement_clean_1D.npy")

## setting the noise
noise_level_ = noise_level*max(d_clean)
noise = NoiseGaussianIID(dim=len(measurement_points))
noise.set_parameters(variance=noise_level_**2)

## setting the Model
model = ModelSS(
    d=d, domain_equ=domain_equ, prior=prior_measure, 
    noise=noise, equ_solver=equ_solver
    )

## setting rMAP
rmap = rMAP(model)
rmap.prepare(num_eigval=30)
## |cost_pre - cost| <= eta, the optimizer stops
## method: specify the NewtonCG optimizer with 'cg_my' is used.
m_samples = rmap.generate_sample(num_samples=100, method='cg_my', eta=1e-2, cg_max=20, max_iter=50)

num_samples = m_samples.shape[0]
plt.figure()
fun = fe.Function(model.domain_equ.function_space)
fe.plot(truth_fun, linewidth=2)
for itr in range(num_samples):
    fun.vector()[:] = np.array(m_samples[itr, :])
    fe.plot(fun, alpha=0.1)
fe.plot(truth_fun, label='truth', linewidth=2)
fun.vector()[:] = rmap.map_point
fe.plot(fun, label='map', linewidth=2)
plt.legend()
plt.show()
















