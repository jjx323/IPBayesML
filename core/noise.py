#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 11:53:31 2022

@author: Junxiong Jia
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
    
    def to_tensor(self, dtype=torch.float32):
        assert type(self.covariance) != type(None)
        assert type(self.precision) != type(None)
        assert type(self.mean) != type(None)
        if type(self.covariance) == np.ndarray:
            self.covariance = torch.tensor(self.covariance, dtype=dtype)
        else:
            self.covariance = torch.tensor(self.covariance.todense(), dtype=dtype)
        if type(self.precision) == np.ndarray:
            self.precision = torch.tensor(self.precision, dtype=dtype)
        else:
            self.precision = torch.tensor(self.precision.todense(), dtype=dtype)
        self.mean = torch.tensor(self.mean, dtype=dtype)
        self.is_torch = True
        
    def to_torch_cuda(self, dtype=torch.float32):
        assert type(self.covariance) != type(None)
        assert type(self.precision) != type(None)
        assert type(self.mean) != type(None)
        if type(self.covariance) == np.ndarray:
            self.covariance = torch.tensor(self.covariance, dtype=dtype).cuda()
        else:
            self.covariance = torch.tensor(self.covariance.todense(), dtype=dtype).cuda()
        if type(self.precision) == np.ndarray:
            self.precision = torch.tensor(self.precision, dtype=dtype).cuda()
        else:
            self.precision = torch.tensor(self.precision.todense(), dtype=dtype).cuda()
        self.mean = torch.tensor(self.mean, dtype=dtype).cuda()
        self.is_torch = True
    
    def to_torch(self, device="cpu"):
        if device == "cpu":
            self.to_tensor()
        elif device == "cuda":
            self.to_torch_cuda()
        else:
            raise NotImplementedError("device must be cpu or cuda")
        
    def to_numpy(self):
        self.covariance = np.array(self.covariance.cpu())
        self.precision = np.array(self.precision.cpu())
        self.mean = np.array(self.mean.cpu()) 
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
            device = self.mean.device
            a = torch.normal(0, 1, size=(self.dim,), dtype=torch.float32).to(device)
            B = torch.eye(self.dim, dtype=torch.float32)*torch.sqrt(self.covariance[0]).to(device)
            sample = torch.mv(B, a)
        else:
            a = np.random.normal(0, 1, (self.dim,))
            B = sps.diags(np.sqrt(self.covariance.diagonal()))
            sample = np.array(B@a)

        return sample













