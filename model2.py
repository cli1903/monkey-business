#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 20:19:38 2019
a
"""

import tensorflow as tf
import numpy as np

class Model(tf.keras.Model):
    def __init__(self, kin_markers):

        """
        The Model class predicts the next words in a sequence.
        Feel free to initialize any variables that you find necessary in the constructor.

        :param vocab_size: The number of unique words in the data
        """

        super(Model, self).__init__()

        self.batch_size = 100
        self.rnn_size = 300
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
        
        self.num_markers = kin_markers
        

        self.rnn = tf.keras.layers.LSTM(self.rnn_size, return_sequences = True, return_state = True)
        
        self.dense1 = tf.keras.layers.Dense(2000, activation = tf.nn.leaky_relu)
        self.dense2 = tf.keras.layers.Dense(1000, activation = tf.nn.leaky_relu)
        self.dense3 = tf.keras.layers.Dense(self.num_markers)
        
        self.drop1 = tf.keras.layers.Dropout(0.1)
        self.drop2 = tf.keras.layers.Dropout(0.3)

        self.flatten = tf.keras.layers.Flatten()
        


    def call(self, inputs):
        rnn, state1, state2 = self.rnn(inputs)
        d1 = self.flatten(self.dense1(rnn))
        d2 = self.dense2(self.drop1(d1))
        d3 = self.dense3(self.drop2(d2))
        
        return d3
    
    def loss(self, pred, labels):
        print(tf.reduce_mean(tf.keras.losses.MSE(labels, pred)))
        return tf.reduce_mean(tf.keras.losses.MSE(labels, pred))

