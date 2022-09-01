#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 20:56:03 2022

@author: Junxiong Jia
"""

import numpy as np
import os

###################################################################
class pCN(object):
    '''
    M. Dashti, A. M. Stuart, The Bayesian Approch to Inverse Problems, 
    Hankbook of Uncertainty Quantification, 2017
    '''
    def __init__(self, prior, phi, beta=0.01, save_num=np.int64(1e4), path=None):
        
        assert hasattr(prior, "generate_sample") and hasattr(prior, "mean_vec")
        assert hasattr(prior, "generate_sample_zero_mean")
        
        self.prior = prior
        self.phi = phi
        self.beta = beta
        self.dt = (2*(2 - beta**2 - 2*np.sqrt(1-beta**2)))/(beta**2)
        self.save_num = save_num
        self.path = path
        if self.path is not None:
            if not os.path.exists(self.path):
                os.makedirs(self.path)
        
    def generate_chain(self, length_total=1e5, callback=None, uk=None, index=None):
        
        assert type(length_total) == type(1.0) or type(length_total) == type(np.array(1.0)) or type(length_total) == type(1)
        
        chain = []
        if uk is None:
            uk = self.prior.generate_sample()
        else:
            uk = uk
        chain.append(uk.copy())
        ac_rate = 0
        ac_num = 0
        
        def aa(u_new, phi_old):
            #return min(1, np.exp(self.phi(u_old)-self.phi(u_new)))
            #print(self.phi(u_old)-self.phi(u_new))
            phi_new = self.phi(u_new)
            assert phi_new <= 1e20, "The algorithm cannot work when phi > 1e20"
            panduan = np.exp(min(0.0, phi_old-phi_new))
            return panduan, phi_new
        
        si = 0
        if index == None: index = 0
        phi_old = 1e20   # a large enough number 
        i = 1
        m0 = self.prior.mean_vec
        while i <= length_total:
            a = np.sqrt(1-self.beta*self.beta)
            b = self.beta
            xik = self.prior.generate_sample_zero_mean()
            ## ref: Algorithms for Kullback-Leibler approximation of probability 
            ## measures in infinite dimensions, SIAM J. SCI. COMPUT, 2015
            vk = m0 + a*(uk - m0) + b*xik
            t, phi_new = aa(vk, phi_old)
            r = np.random.uniform(0, 1)
            if t >= r:
                chain.append(vk.copy())
                uk = vk.copy()
                ac_num += 1
                phi_old = phi_new
            else: 
                chain.append(uk.copy())
            ac_rate = ac_num/i 
            i += 1
            
            if self.path is not None:
                si += 1
                if np.int64(si) == np.int64(self.save_num):
                    si = 0
                    np.save(self.path + 'sample_' + np.str(np.int64(index)), chain)
                    del chain
                    chain = []
                    index += 1
        
            if callback is not None:
                callback([uk, i, ac_rate, ac_num])

        if self.path is None:
            return [chain, ac_rate]
        else:
            return [ac_rate, self.path, np.int64(index)]
        






