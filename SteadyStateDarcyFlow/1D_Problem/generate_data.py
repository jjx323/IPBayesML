#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 21:52:40 2022

@author: jjx323
"""

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import matplotlib.pyplot as plt

import fenics as fe
import dolfin as dl

import sys, os
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
from core.model import Domain1D
from core.probability import GaussianElliptic2
from core.misc import save_expre

from SteadyStateDarcyFlow.common import EquSolver


DATA_DIR = './DATA/'

## domain for solving PDE
equ_nx = 1000
domain_equ = Domain1D(n=equ_nx, mesh_type='CG', mesh_order=1)

alpha = 1.0
a_fun = 1.0
prior_measure = GaussianElliptic2(
    domain=domain_equ, alpha=alpha, a_fun=a_fun, theta=0.1, boundary='Neumann'
    )

## generate a sample, set this sample as the ground truth
truth_fun = fe.Function(domain_equ.function_space)
truth_fun.vector()[:] = prior_measure.generate_sample()
#fe.plot(truth_fun)
## save the truth
os.makedirs(DATA_DIR, exist_ok=True)
np.save(DATA_DIR + 'truth_vec', truth_fun.vector()[:])
file1 = fe.File(DATA_DIR + "truth_fun.xml")
file1 << truth_fun
file2 = fe.File(DATA_DIR + 'saved_mesh_truth.xml')
file2 << domain_equ.function_space.mesh()

## load the ground truth
truth_fun = fe.Function(domain_equ.function_space)
truth_fun.vector()[:] = np.load(DATA_DIR + 'truth_vec.npy')

## specify the measurement points
coordinates = np.linspace(0, 1, 20)

## construct a solver to generate data
# f_expre = "sin(1*pi*x[0])"
f_expre = "1.0"  
f = fe.Expression(f_expre, degree=5)
save_expre(DATA_DIR + 'f_1D.txt', f_expre)

equ_solver = EquSolver(domain_equ=domain_equ, m=truth_fun, f=f, points=np.array([coordinates]).T,)

sol = fe.Function(domain_equ.function_space)
sol.vector()[:] = equ_solver.forward_solver()
clean_data = [sol(point) for point in coordinates]
np.save(DATA_DIR + 'measurement_points_1D', coordinates)
np.save(DATA_DIR + 'measurement_clean_1D', clean_data)
data_max = max(clean_data)
## add noise to the clean data
noise_levels = [0.01, 0.02, 0.03, 0.04, 0.05]
for noise_level in noise_levels:
    data = clean_data + noise_level*data_max*np.random.normal(0, 1, (len(clean_data),))
    path = DATA_DIR + 'measurement_noise_1D' + '_' + str(noise_level)
    np.save(path, data)

    











