#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 10:36:55 2021

@author: sam
"""

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from tqdm import tqdm

class bootstrapTest():
    def __init__(self,category, canonical_set_traces, tau, n_pulse, isi,
                 integrate_time, experiment_list, notes=None, canonical_set = None):
        self.category  = category
        self.canonical_set_traces = canonical_set_traces
        self.tau = tau
        self.n_pulse = n_pulse
        self.isi = isi
        self.integrate_time = integrate_time
        self.experiment_list = experiment_list
        self.canonical_set = canonical_set
        if canonical_set is None:
            self.generate_integratedPulse()
        self.notes = notes

    def generate_integratedPulse(self):
        pulses = [[] for r in self.canonical_set_traces]
        for i,r in enumerate(self.canonical_set_traces):
            loc = np.argmin(np.abs(self.tau))
            for j in range(self.n_pulse):
                pulses[i].append(np.mean(r[:,loc:loc+self.integrate_time],axis=1))
                loc += self.isi
        self.canonical_set = [np.array(p) for p in pulses]
        return

    def generate_bootstrap(self,sample_size,ind,statistic,n_boot=1e5):
        bootstrap = []
#         for i in tqdm(range(int(n_boot)),position=0,leave=True):
        for i in range(int(n_boot)):
            bootstrap.append(statistic(np.random.choice(self.canonical_set[ind[0]][ind[1]],sample_size)))
        return bootstrap

    def sample_rank(self, sample, ind=None, statistic=np.mean, n_boot=1e5):
        sample_size = sample.size
        bootstrap = self.generate_bootstrap(sample_size, ind, statistic,n_boot=n_boot)
        observed = statistic(sample)
        return scipy.stats.percentileofscore(bootstrap,observed)

    def canonical_estimate(self, ind, statistic=np.mean, conf_interval=95,
                           sample_size=None, n_boot=1e5):
        if conf_interval is None:
            return statistic(self.canonical_set[ind[0]][ind[1]])
        if sample_size is None:
            sample_size = self.canonical_set[ind[0]][ind[1]].size
        bootstrap = self.generate_bootstrap(sample_size, ind, statistic,n_boot=n_boot)
        return np.mean(bootstrap), [np.percentile(bootstrap,(100-conf_interval)/2),
                                    np.percentile(bootstrap,conf_interval+(100-conf_interval)/2)]

    def plot_bootstrap(self,sample_size, ind, statistic=np.mean, ax=None,
                       bins=np.linspace(0,2,1000),c='k',conf_interval=95,n_boot=1e5):
        if ax is None:
            fig = plt.figure()
            ax = fig.gca()
        bootstrap = self.generate_bootstrap(sample_size, ind, statistic,n_boot=n_boot)
        y, _ = np.histogram(bootstrap, bins)
        y = y/y.sum()
        ax.plot(bins[1:],y,c=c)
        #95 CI of bootstrap
        y_sum = np.cumsum(y)
        lo = np.where(y_sum>(1-conf_interval/100)/2)[0][0]
        hi = np.where(y_sum>conf_interval/100+(1-conf_interval/100)/2)[0][0]
        plt.fill_between(bins[1:][lo:hi],np.zeros(hi-lo),y[lo:hi],
                          facecolor=c, alpha=.1)

def bootstrap_traces(data, sample_size=None, statistic=np.mean,
                     n_boot=1e3 ,conf_interval=95,):
    if sample_size is None:
        sample_size = data.shape[0]
    bootstrap = []
#     for i in tqdm(range(int(n_boot)),position=0,leave=True):
    for i in range(int(n_boot)):
        bootstrap.append(statistic(data[np.random.choice(np.arange(data.shape[0]),sample_size),:],axis=0))
    bootstrap = np.array(bootstrap)
    return np.mean(bootstrap,axis=0), [np.percentile(bootstrap,(100-conf_interval)/2,axis=0),
                                    np.percentile(bootstrap,conf_interval+(100-conf_interval)/2,axis=0)]

def timeDependentDifference(data,ref,n_boot=1e3,conf_interval=99):
    '''Returns whether data and ref are significantly deifferenct at each timepoint'''
    #'''Runs differently than the other measurements below (aka: not interchangable)'''
    pop_samp = lambda x: np.median(x[np.random.choice(np.arange(x.shape[0]),x.shape[0])],axis=0)
    diff = np.array([pop_samp(data) - pop_samp(ref) for i in range(int(n_boot))])
    diff = (np.percentile(diff,(100-conf_interval)/2,axis=0)>0) + (np.percentile(diff,conf_interval+(100-conf_interval)/2,axis=0)<0)
    return diff


def bootstrap_compare(data1, data2, operator=np.subtract, measurement=None, sample_size=None, statistic=np.mean,
                       n_boot=1e3, conf_interval=95,return_samples=False,**kwargs):
    '''bootstap comparison of samples from 2 datasets'''
    #measurement: the value being calculated from samples (see measurement.py)
    #    ***for efficiency, precalculate measurement and use None value for non-population averaged measures
    #operator: how we compare the measurements
    #statistic: what value of the distribution of operator results we care about (*irrelevant for population based measures)
    if sample_size is None:
        sample_size = data1.shape[0]#min(data1.shape[0],data2.shape[0])
    if measurement is None:
        measurement = lambda x: x
    bootstrap = []
    for i in tqdm(range(int(n_boot)),position=0,leave=True):
#     for i in range(int(n_boot)):
        bootstrap.append(statistic(
            operator(measurement(data1[np.random.choice(np.arange(data1.shape[0]),sample_size)],**kwargs),
                     measurement(data2[np.random.choice(np.arange(data2.shape[0]),sample_size)],**kwargs)
                    )
            ))
    bootstrap = np.array(bootstrap)
    if return_samples:
        return np.mean(bootstrap), [np.percentile(bootstrap,(100-conf_interval)/2),
                                    np.percentile(bootstrap,conf_interval+(100-conf_interval)/2)], bootstrap
    
    return np.mean(bootstrap), [np.percentile(bootstrap,(100-conf_interval)/2),
                                    np.percentile(bootstrap,conf_interval+(100-conf_interval)/2)]

def bootstrap_diff(data1, data2, measurement=None, sample_size=None, statistic=np.mean,
                       n_boot=1e3, conf_interval=95,return_samples=False,**kwargs):
        y,rng,boot = bootstrap_compare(data1, data2, np.subtract, measurement,
                sample_size, statistic,n_boot, conf_interval,return_samples=True,**kwargs)
        if return_samples:
            if rng[0]>0 or rng[1]<0:
                return y, rng, True, boot
            else:
                return y, rng, False, boot
        if rng[0]>0 or rng[1]<0:
            return y, rng, True
        else:
            return y, rng, False

def bootstrap_relative(data1, data2, measurement=None, sample_size=None, statistic=np.mean,
                       n_boot=1e3, conf_interval=95,**kwargs):
        y,rng = bootstrap_compare(data1, data2, np.divide, measurement,
                sample_size, statistic,n_boot, conf_interval,**kwargs)
        if rng[0]>1 or rng[1]<1:
            return y, rng, True
        else:
            return y, rng, False


def bootstrap(data,sample_size=None,statistic=np.median,conf_interval=95,n_boot=1e5,
              return_samples=False,**kwargs):
        if sample_size is None:
            sample_size = data.shape[0]
        bootstrap = []
        for i in range(int(n_boot)):
#             bootstrap.append(statistic(np.random.choice(data,sample_size)))
            bootstrap.append(statistic(data[np.random.choice(np.arange(data.shape[0]),sample_size)],**kwargs))
        if return_samples:
                return np.mean(bootstrap,axis=0), [np.percentile(bootstrap,(100-conf_interval)/2,axis=0),
                                                   np.percentile(bootstrap,conf_interval+(100-conf_interval)/2,axis=0)], bootstrap
        
        return np.mean(bootstrap,axis=0), [np.percentile(bootstrap,(100-conf_interval)/2,axis=0),
                                    np.percentile(bootstrap,conf_interval+(100-conf_interval)/2,axis=0)]
