#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 11:53:31 2022

@author: jjx323
"""

import numpy as np
# import scipy.linalg as sl
import scipy.sparse as sps
# import scipy.sparse.linalg as spsl
# import fenics as fe
import torch

############################################################################
class NoiseGaussianIID(object):
    def __init__(self, dim):
        assert type(dim) in [int, np.int32, np.int64]
        self.dim = dim
        self.mean = None
        self.covariance = None
        self.precision = None
        self.is_torch = False

    def set_parameters(self, mean=None, variance=None):
        if mean is None:
            # defalut value is zero
            self.mean = np.zeros(self.dim,)
        else:
            self.mean = mean

        if variance is None:
            # default value is identity matrix
            self.covariance = sps.eye(self.dim)
            self.precision = sps.eye(self.dim)
        else:
            assert variance >= 1e-15
            self.covariance = sps.eye(self.dim)*variance
            self.precision = sps.eye(self.dim)*(1.0/variance) 
    
    def to_tensor(self):
        assert type(self.covariance) != type(None)
        assert type(self.precision) != type(None)
        assert type(self.mean) != type(None)
        if type(self.covariance) == np.ndarray:
            self.covariance = torch.tensor(self.covariance, dtype=torch.float32)
        else:
            self.covariance = torch.tensor(self.covariance.todense(), dtype=torch.float32)
        if type(self.precision) == np.ndarray:
            self.precision = torch.tensor(self.precision, dtype=torch.float32)
        else:
            self.precision = torch.tensor(self.precision.todense(), dtype=torch.float32)
        self.mean = torch.tensor(self.mean, dtype=torch.float32)
        self.is_torch = True
        
    def to_numpy(self):
        self.covariance = np.array(self.covariance)
        self.precision = np.array(self.precision)
        self.mean = np.array(self.mean) 
        self.is_torch = False

    def update_paramters(self, mean=None, variance=None):
        if mean is not None:
            self.mean = mean
        if variance is not None:
            assert variance >= 1e-15
            self.covariance = sps.eye(self.dim)*variance
            self.precision = sps.eye(self.dim)*(1.0/variance) 
    
    def generate_sample(self):
        return self.mean + self.generate_sample_zero_mean()
    
    def generate_sample_zero_mean(self):
        if type(self.covariance) == torch.Tensor:
            a = torch.normal(0, 1, size=(self.dim,), dtype=torch.float32)
            B = torch.eye(self.dim, dtype=torch.float32)*torch.sqrt(self.covariance[0])
            sample = torch.mv(B, a)
        else:
            a = np.random.normal(0, 1, (self.dim,))
            B = sps.diags(np.sqrt(self.covariance.diagonal()))
            sample = np.array(B@a)

        return sample













