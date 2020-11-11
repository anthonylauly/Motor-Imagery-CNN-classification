# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 21:49:42 2020

@author: antho
"""

from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

import numpy as np
import pandas as pd

def variable_init():
    '''
    '''
    num_learning_rate = [1e-03, 5e-04, 1e-04, 5e-05, 1e-05, 5e-06]
    num_conv_size_1 = [1, 2, 3, 5]
    num_conv_size_2 = [6, 36]
    num_conv_filters = [16, 24, 32, 36]
    num_conv_layers = [1, 2, 3, 4, 5]
    num_dense_nodes = [16, 24, 32, 48, 64, 128, 256, 512, 1024]
    num_dense_layers = [0, 1, 2, 3]
    #num_epochs = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550]
    
    param_grid  = dict(learning_rate=num_learning_rate, conv_size_1=num_conv_size_1, 
                      conv_size_2=num_conv_size_2, num_conv_filters=num_conv_filters, 
                      num_conv_layers=num_conv_layers, num_dense_nodes=num_dense_nodes, 
                      num_dense_layers=num_dense_layers)
    
    return param_grid

def create_model(learning_rate=5e-6, conv_size_1=2, conv_size_2=6, 
                 num_conv_filters=32, num_conv_layers=2, num_dense_nodes=512, 
                 num_dense_layers=2):
    '''
    Hyper-parameters:
    learning_rate:     Learning-rate for the optimizer.
    conv_size_1:       Conv layer kernel dimension 1
    conv_size_2:       Conv layer kernel dimension 2
    num_conv_filters:  Number of nodes in conv layer
    num_conv_layers:   Number of conv layers
    num_dense_nodes:   Number of nodes in each dense layer.
    num_dense_layers:  Number of dense layers.
    '''

    # Start construction of a Keras Sequential model.
    model = Sequential()

    # Add an input layer which is similar to a feed_dict in TensorFlow.
    # Note that the input-shape must be a tuple containing the image-size.
    model.add(InputLayer(input_shape=(11,6*36,1)))

    # First convolutional layer.
    # There are many hyper-parameters in this layer, but we only
    # want to optimize the activation-function in this example.
    for i in range(num_conv_layers):
        name = 'conv_layer_{0}'.format(i+1)
        
        model.add(Conv2D(kernel_size=[conv_size_1, conv_size_2], strides=[1, 6], 
                   filters=num_conv_filters, padding='same', 
                   activation='relu', name=name))

    # Flatten the 4-rank output of the convolutional layers
    # to 2-rank that can be input to a fully-connected / dense layer.
    model.add(Flatten())

    # Add fully-connected / dense layers.
    # The number of layers is a hyper-parameter we want to optimize.
    for i in range(num_dense_layers):
        # Name of the layer. This is not really necessary
        # because Keras should give them unique names.
        name = 'layer_dense_{0}'.format(i+1)
        
        # Add the dense / fully-connected layer to the model.
        # This has two hyper-parameters we want to optimize:
        # The number of nodes and the activation function.
        model.add(Dense(num_dense_nodes,
                    activation='relu',
                    name=name))
        
    # Last fully-connected / dense layer with softmax-activation
    # for use in classification.
    model.add(Dense(4, activation='softmax'))
    
    # Use the Adam method for training the network.
    # We want to find the best learning-rate for the Adam method.
    optimizer = Adam(lr=learning_rate)
    
    # In Keras we need to compile the model so it can be trained.
    model.compile(optimizer=optimizer,
          loss='categorical_crossentropy',
          metrics=['accuracy'])
    
    return model

def parameter_search(X, y, param_grids, num_epochs=100):
    model = KerasClassifier(build_fn=create_model, epochs=400, batch_size=10)
    grid = GridSearchCV(estimator=model, param_grid=param_grids, cv=5)
    
    grid_result = grid.fit(X, y)
    
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    params = grid_result.cv_results_['params']
    for mean, param in zip(means, params):
        print("%f  with: %r" % (mean, param))
        
    return grid_result




