#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scipy.io import loadmat
from scipy.interpolate import interp1d
import numpy as np

"""
Created on Thu Nov 21 16:18:16 2019
"""

def get_data(file):
    monkey = loadmat(file)
    
    print("loaded")
    
    kin = monkey["KIN"]
    kin_time = monkey["kin_times"]
    spike = monkey["spikes"]
    
    bin_size = 0.1
    
    aligned = []
    spks = []

    
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
        
        #print(np.array(spkcounts).shape)     
        
        for j in range(0, len(bintime), 50):       
            if j + 10 >= len(bintime):
                break
            trial_stack.append(np.array(spkcounts)[:, j : j + 10])
            trial_motion.append(np.array(kin_move)[:, j + 10])
        
            
        if i == 0:
            stacked_bins = np.array(trial_stack)
            motion_labels = np.array(trial_motion)
        else:
            stacked_bins = np.vstack((stacked_bins, np.array(trial_stack)))
            motion_labels = np.vstack((motion_labels, np.array(trial_motion)))
            
    stacked_bins = np.transpose(stacked_bins.astype('float32'), axes = [0, 2, 1])
    motion_labels = motion_labels.astype('float32')
    
    #print(stacked_bins.shape)
        
    return stacked_bins, motion_labels


#monkey1 = 'COS071212_MOCAP.mat'
#monkey2 = 'GAR080710_MOCAP.mat'
    
#monkey = loadmat(monkey1)

        
    
