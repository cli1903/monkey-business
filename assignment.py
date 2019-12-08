#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 12:48:45 2019

@author: cindyli
"""

from __future__ import absolute_import
#from matplotlib import pyplot as plt
import numpy as np
from preprocessing import get_data
import tensorflow as tf
from model import Model

def train(model, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels.
    :param model: the initialized model to use for the forward
    pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training)
    :param train_inputs: train labels (all labels to use for training)
    :return: None
    '''

    # TODO: Iterate over the training inputs and labels, in model.batch_size increments
    # TODO: For every batch, compute then descend the gradients for the model's weights
    for i in range(0, len(train_labels), model.batch_size):
        inputs = train_inputs[i:i+model.batch_size]
        labels = train_labels[i:i+model.batch_size]
        with tf.GradientTape() as tape:
            pred = model.call(inputs)
            loss = model.loss(pred, labels)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            


def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. For this assignment,
    the inputs should be the entire test set, but in the future we will
    ask you to batch it instead.
    :param test_inputs: MNIST test data (all images to be tested)
    :param test_labels: MNIST test labels (all corresponding labels)
    :return: accuracy - Float (0,1)
    """

    # TODO: Iterate over the testing inputs and labels
    # TODO: Return accuracy across testing set

    pred = model.call(test_inputs)
    return model.loss(pred, test_labels)


def main():
    '''
    Read in MNIST data, initialize your model, and train and test your model
    for one epoch. The number of training steps should be your the number of
    batches you run through in a single epoch. You should receive a final accuracy on the testing examples of > 80%.
    :return: None
    '''

    NUM_EPOCHS = 200
    # TODO: load MNIST train and test examples into train_inputs, train_labels, test_inputs, test_labels
    fr, km = get_data('COS071212_MOCAP.mat')
    
    eighty_p = int(len(fr) * 0.8)
    
    train_inp = fr[:eighty_p]
    train_lab = km[:eighty_p]
    
    test_inp = fr[eighty_p:]
    test_lab = km[eighty_p:]
    

    # TODO: Create Model
    model = Model(29)

    # TODO: Train model by calling train() ONCE on all data
    
    indices = range(0, len(train_lab))
    for i in range(NUM_EPOCHS):
        tf.random.shuffle(train_lab)
        train_inp = tf.gather(train_inp, indices)
        train_lab = tf.gather(train_lab, indices)
        print("training")
        train(model, train_inp, train_lab)
        print("testing")
        results = test(model, test_inp, test_lab)

    # TODO: Test the accuracy by calling test() after running train()
    
    
    print("results:", results)



if __name__ == '__main__':
    main()