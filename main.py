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

subject = 9
no_channels = 22
no_trials = 288
fs = 250
window_length = 7

PATH='D:/Semester 7/Tugas Akhir/Final Code/'

X,y = data_prep.get_data(subject, no_channels, no_trials, fs, window_length, 
                         True, PATH+'datasets/')

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=300)

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

csp_filter = csp_extraction.generate_projection(X_train, y_train,filterbanks,
                                                time_windows)

csp_train = csp_extraction. extract_feature(X_train, csp_filter, 
                                            filterbanks, time_windows)

csp_val = csp_extraction.extract_feature(X_val, csp_filter, filterbanks, 
                                         time_windows)

y_train = pd.get_dummies(y_train).values
y_val = pd.get_dummies(y_val).values

# =============================================================================
# Grid Search Tuning
# =============================================================================
path_best_model = 'subject_{}/'.format(subject)

param_grid = chs.variable_init()
grid_result = chs.parameter_search(csp_train, y_train, param_grid)

pd(grid_result).to_csv('subject_{}_best_params'.format(subject))

hyperparams = np.array(pd.read_csv(path_best_model + 'best_cnn_params_{}.csv'.format(subject))).flatten()

# =============================================================================
# Train CNN Model
# =============================================================================
hyperparams = [1e-05, 1, 36, 16, 1, 32, 2, 550]
CNN_model = ctt.create_model(hyperparams)
CNN_model.save(PATH+'model_init/subject_{}.h5'.format(subject))
CNN_model = load_model(PATH+'model_init/subject_{}.h5'.format(subject))
ctt.train_model(csp_train, y_train, csp_val, y_val, CNN_model, 32, 550)

# =============================================================================
# Test Data Evalutaion
# =============================================================================
X_test, y_test = data_prep.get_data(subject, no_channels, no_trials, fs, 
                                    window_length, False, PATH+'datasets/')

csp_test = csp_extraction.extract_feature(X_test, csp_filter, filterbanks,
                                          time_windows)
y_test = pd.get_dummies(y_test).values

test_acc = ctt.model_evaluation(CNN_model, csp_test, y_test)
