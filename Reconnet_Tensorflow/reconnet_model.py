#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 20:37:10 2019

@author: kaushik47
"""

import tensorflow as tf
import numpy as np

def reconnet_block(x, reuse=False, training=True):
    with tf.variable_scope('recon_block', reuse=reuse):
        e1 = tf.layers.dense(inputs=x, units=1089, activation=tf.nn.leaky_relu, 
                             kernel_initializer=tf.random_normal_initializer(stddev=0.05))
        e1 = tf.reshape(e1, (-1, 33,33,1))
        e2 = tf.layers.conv2d(e1, 64, 11, padding='same', activation=tf.nn.leaky_relu, 
                              kernel_initializer=tf.random_normal_initializer(stddev=0.1))
        e3 = tf.layers.conv2d(e2, 32, 1, padding='same', activation=tf.nn.leaky_relu, 
                              kernel_initializer=tf.random_normal_initializer(stddev=0.1))
        e4 = tf.layers.conv2d(e3, 1, 7, padding='same', activation=tf.nn.leaky_relu, 
                              kernel_initializer=tf.random_normal_initializer(stddev=0.1))
        e5 = tf.layers.conv2d(e4, 64, 11, padding='same', activation=tf.nn.leaky_relu, 
                              kernel_initializer=tf.random_normal_initializer(stddev=0.1))
        e6 = tf.layers.conv2d(e5, 32, 1, padding='same', activation=tf.nn.leaky_relu, 
                              kernel_initializer=tf.random_normal_initializer(stddev=0.1))
        e7 = tf.layers.conv2d(e6, 1, 7, padding='same', activation=tf.nn.leaky_relu, 
                              kernel_initializer=tf.random_normal_initializer(stddev=0.1))
        
        return e7
    
def normalize(v):
    return v / np.sqrt(v.dot(v))

def generate_phi():
    np.random.seed(333)
    phi = np.random.normal(size=(272, 1089))
    n = len(phi)
    
    # perform Gramm-Schmidt orthonormalization
    
    phi[0, :] = normalize(phi[0, :])
    
    for i in range(1, n):
        Ai = phi[i, :]
        for j in range(0, i):
            Aj = phi[j, :]
            t = Ai.dot(Aj)
            Ai = Ai - t * Aj
        phi[i, :] = normalize(Ai)
        
    return phi

def divide_img_to_blocks(image, stride=14, filter_size=33):
    (h,w) = image.shape
    image = image[0:h-h%3, 0:w-w%3]
    (h,w) = image.shape
    h_iters = ((h - filter_size) // stride) + 1
    w_iters = ((w - filter_size) // stride) + 1
    blocks = []
    for i in range(h_iters):
        for j in range(w_iters):
            blocks.append(image[stride*i:filter_size+stride*i, stride*j:filter_size+stride*j])
    
    return np.asarray(blocks)