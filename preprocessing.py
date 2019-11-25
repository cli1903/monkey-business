#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scipy.io import loadmat
from scipy.interpolate import interp1d
import numpy as np

"""
Created on Thu Nov 21 16:18:16 2019

"""

def align_kin(file):
    monkey = loadmat(file)
    
    kin = monkey["KIN"]
    #kin_lab = monkey["kin_labels"]
    kin_time = monkey["kin_times"]
    spike = monkey["spikes"]
    
    bin_size = 0.1
    
    aligned = []
    spks = []
    stacked_bins = []
    motion_labels = []
    
    for i in range(6):
        
        trial_stack = []
        trial_motion = []
        
        start_bins = np.arange(np.min(kin_time[:,i][0]), np.max(kin_time[:,i][0]), bin_size)
        
        tk = kin[:,i][0]
        
        
        spkcounts = []
        for j in range(260):
            spkcounts.append((np.histogram(spike[0][i][0][j], start_bins)[0])[1:].tolist())

        bintime = start_bins[2:]
        f = interp1d(kin_time[:,i][0][0], tk)
        
        kin_move = f(bintime)
        
        aligned.append(kin_move)
        spks.append(spkcounts)
        
        
        for j in range(0, len(bintime), 10):       
            if j + 10 >= len(bintime):
                break
            trial_stack.append(np.array(spkcounts)[:][j: j + 10])
            trial_motion.append(kin_move[:, j + 10])
            
    
        stacked_bins.append(trial_stack)
        motion_labels.append(trial_motion)
        
    return stacked_bins, motion_labels
                 

#monkey1 = 'COS071212_MOCAP.mat'
#monkey2 = 'GAR080710_MOCAP.mat'
    
fr, km = align_kin('COS071212_MOCAP.mat')
