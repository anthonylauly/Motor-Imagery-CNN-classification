# -*- coding: utf-8 -*-
"""
Created on Mon May 25 20:42:40 2020

@author: antho
"""
import numpy as np
import scipy.io as sio
from scipy.signal import butter

def get_data(subject, no_channels, no_trials, fs, windows_length, training=True, PATH=''):
    """ Load .mat files into numpy array
    
    Keyword arguments:
    subject -- (int) 1-9 indicating patient's subject
    no_channels -- number of eeg channels
    no_trials -- number of data samples
    fs -- frequency sample
    windows_length -- time length for MI extraction samples
    training -- (Bool) is it training or testing data
    PATH -- path where .mat file is located

    """
    
    labels = np.zeros(no_trials)
    data = np.zeros((no_trials, no_channels, windows_length*fs))

    NO_valid_trial = 0
    if training:
        tmp = sio.loadmat(PATH+'A0'+str(subject)+'T.mat')
        tmp_data = tmp['data']
        for ii in range(0,tmp_data.size):
            a_data1 = tmp_data[0,ii]
            a_data2=[a_data1[0,0]]
            a_data3=a_data2[0]
            a_X 		= a_data3[0]
            a_trial 	= a_data3[1]
            a_y 		= a_data3[2]
            a_artifacts = a_data3[5]
    
            for trial in range(0,a_trial.size):
                if (a_artifacts[trial] == 0):
                    data[NO_valid_trial,:,:] = np.transpose(a_X[int(a_trial[trial]):(int(a_trial[trial])+windows_length*fs),:no_channels])
                    labels[NO_valid_trial] = int(a_y[trial]) - 1
                    NO_valid_trial +=1
    
    else:
        tmp = sio.loadmat(PATH+'A0'+str(subject)+'E.mat')
        tmp_data = tmp['data']
        for ii in range(0,tmp_data.size):
            a_data1 = tmp_data[0,ii]
            a_data2=[a_data1[0,0]]
            a_data3=a_data2[0]
            a_X 		= a_data3[0]
            a_trial 	= a_data3[1]
            a_y 		= a_data3[2]
    
            for trial in range(0,a_trial.size):
                data[NO_valid_trial,:,:] = np.transpose(a_X[int(a_trial[trial]):(int(a_trial[trial])+windows_length*fs),:no_channels])
                labels[NO_valid_trial] = int(a_y[trial]) - 1
                NO_valid_trial +=1
        
    return data[0:NO_valid_trial,:,:], labels[0:NO_valid_trial]

def load_filterbank(bandwidth, fs, max_freq = 32, order = 2, ftype = 'butter'): 
    '''	Calculate Filters bank with Butterworth filter  

	Keyword arguments:
	bandwith -- (list/int) containing bandwiths ex. [2,4,8,16,32] or 4
	fs -- sampling frequency
    max_freq -- max freq used in filterbanks
    order -- The order of filter used
    ftype -- Type of digital filter used

	Return:	numpy array containing filters coefficients dimesnions 'butter': [N_bands,order,6] 'fir': [N_bands,order]
	'''
    f_bands = np.zeros((6,2)).astype(float)

    band_counter = 0
    
    if type(bandwidth) is list :
        for bw in bandwidth:
            startfreq = 7
            while (startfreq + bw <= max_freq): 
                f_bands[band_counter] = [startfreq, startfreq + bw]
    			
                if bw ==1: # do 1Hz steps
                    startfreq = startfreq +1
                elif bw == 2: # do 2Hz steps
                    startfreq = startfreq +2 
                else : # do 4 Hz steps if Bandwidths >= 4Hz
                    startfreq = startfreq +4
    
                band_counter += 1 
    
    if type(bandwidth) is int:
        startfreq = 7
        while (startfreq + bandwidth <= max_freq): 
            f_bands[band_counter] = [startfreq, startfreq + bandwidth]
            startfreq = startfreq + bandwidth
            band_counter += 1 
            
	# convert array to normalized frequency 
    f_band_nom = 2*f_bands[:band_counter]/fs
    n_bands = f_band_nom.shape[0]
    
    filter_bank = np.zeros((n_bands,order,6))

    for band_idx in range(n_bands):
        if ftype == 'butter': 
            filter_bank[band_idx] = butter(order, f_band_nom[band_idx], analog=False, btype='bandpass', output='sos')

    return filter_bank