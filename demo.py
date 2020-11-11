# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
@author: antho
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

import data_prep
import csp_extraction
import cnn_hyperparameter_search as chs
import cnn_train_test as ctt


PATH='D:/Semester 7/Tugas Akhir/Final Code/'

# =============================================================================
# Plot EEG Signal
# =============================================================================
import mne

raw = mne.io.read_raw_gdf(PATH+'datasets_gdf/A01T.gdf')
raw.plot()

# =============================================================================
# load spatial filter
# =============================================================================
subject = 2
no_channels = 22
no_trials = 288
fs = 250
window_length = 7

X,y = data_prep.get_data(subject, no_channels, no_trials, fs, window_length, 
                         True, PATH+'datasets/')

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.05, random_state = 350)

time_windows = time_windows = np.array([[2.5,3.5],
		 				  [3,4],
		 			   	[3.5,4.5],
		 				  [4,5],
		 				  [4.5,5.5],
		 				  [5,6],
		 			   	[2.5,4.5],
		 				  [3,5],
		 			 	  [3.5,5.5],
		 				  [4,6],
		 			 	  [2.5,6]])*250

filterbanks = data_prep.load_filterbank(bandwidth=4, fs=fs)

csp_filter = pd.read_csv(PATH+'spatial_filter/subject{}.csv'.format(subject))
csp_filter = csp_filter.values.reshape([11,len(filterbanks),no_channels,36])

csp_train = csp_extraction. extract_feature(X_train, csp_filter, 
                                            filterbanks, time_windows)

csp_val = csp_extraction.extract_feature(X_val, csp_filter, filterbanks, 
                                         time_windows)


# =============================================================================
# Test Data Evalutaion
# =============================================================================

raw = mne.io.read_raw_gdf(PATH+'datasets_gdf/A01E.gdf')
raw.plot()

X_test, y_test = data_prep.get_data(subject, no_channels, no_trials, fs, 
                                    window_length, False, PATH+'datasets/')

csp_test = csp_extraction.extract_feature(X_test, csp_filter, filterbanks,
                                          time_windows)

CNN_model = load_model('models/subject_{}_best_model.h5'.format(subject))
y_test = pd.get_dummies(y_test).values

test_acc = ctt.model_evaluation(CNN_model, csp_test, y_test)
