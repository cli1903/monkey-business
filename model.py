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

        self.window_size = 20
        self.batch_size = 100
        self.rnn_size = 300
        self.num_markers = kin_markers
        
        
        self.rnn = tf.keras.layers.LSTM(self.rnn_size, return_sequences = True, return_state = True)
        self.dense1 = tf.keras.layers.Dense(1000, activation = 'relu')
        self.dense2 = tf.keras.layers.Dense(self.num_markers)    
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)
        self.flatten = tf.keras.layers.Flatten()


    def call(self, inputs):
        # print(inputs.shape)
        rnn, _, _ = self.rnn(inputs)
        d1 = self.dense1(rnn)
        flattened = self.flatten(d1)
        d2 = self.dense2(flattened)
        # print(d2.shape)
        return d2
    
    def loss(self, pred, labels):
        print("loss:",tf.reduce_mean(tf.keras.losses.MSE(labels, pred)))
        return tf.reduce_mean(tf.keras.losses.MSE(labels, pred))