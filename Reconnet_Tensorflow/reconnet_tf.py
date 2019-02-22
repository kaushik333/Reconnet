#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 20:23:23 2019

@author: kaushik47
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
import reconnet_model
from sklearn.model_selection import train_test_split
import os

#%% CREATE DATASET

# Read data and create patches
image_dataset = []
pics = sorted(os.listdir('./T91/'))
for pic in pics:
    image = cv2.imread('./T91/'+pic)
    image_lum = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    image_lum = cv2.normalize(image_lum.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    image_lum = image_lum[:,:,0]
    blocks = reconnet_model.divide_img_to_blocks(image_lum)
    image_dataset.append(blocks)
    
image_dataset = np.concatenate(image_dataset, axis=0)

A = scipy.io.loadmat('./phi_0_25_1089.mat')["phi"]

#A = reconnet_model.generate_phi()

labels = np.copy(image_dataset)

X_train, X_test, y_train, y_test = train_test_split(image_dataset, image_dataset, test_size=0.1, random_state=333)

# %% BUILD MODEL

graph1 = tf.Graph()
with graph1.as_default():
    tf.set_random_seed(1000)
    num = X_train.shape[0]
    train_dataset = tf.convert_to_tensor(X_train, dtype=tf.float32) 
    train_label = tf.convert_to_tensor(y_train, dtype=tf.float32)
    test_dataset = tf.convert_to_tensor(X_test, dtype=tf.float32) 
    test_label = tf.convert_to_tensor(y_test, dtype=tf.float32)
    
    dataset = tf.data.Dataset.from_tensor_slices(train_dataset)
    dataset = dataset.shuffle(num)
    dataset = dataset.batch(batch_size=128)
    epoch = 5000
    iterator = tf.data.Iterator.from_structure(dataset.output_types,
                                               dataset.output_shapes)
    next_element = iterator.get_next()
    training_init_op = iterator.make_initializer(dataset)   
    
    A1 = tf.constant(A.T, dtype=tf.float32)
    
    x = tf.placeholder(tf.float32, shape=[None, 33, 33], name='x')
    x_comp = tf.matmul(tf.reshape(x,(-1,33*33)), A1)
    
    output = reconnet_model.reconnet_block(x_comp)
    
    loss = tf.reduce_mean(tf.squared_difference(x, tf.squeeze(output)))
    
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 1e-4
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           100000, 0.9, staircase=True)
    
    # Optimizers
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    
# %% PERFORM TRAINING
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
with tf.Session(graph=graph1, config=config) as sess:
  sess.run(init)
#  saver.restore(sess, "./weights_1L_1D_other_template/ae_stn.ckpt")
  for i in range(1, epoch+1): #EPOCHS
    print("epoch no {}".format(i))
    sess.run(training_init_op)
    
    if i%20 == 0:
      print("-------Saving model--------")
      saver.save(sess, "./weights_TF_reconnet/weights.ckpt")    
    j=1    
    while(True):
      try:
        batch = sess.run(next_element)
        feed = {x : batch}
        sess.run(optimizer, feed_dict=feed)
          
        if j%30 ==0:
          t_loss = sess.run(loss, feed_dict=feed)
          print("Loss is: %.9f"%(t_loss))
          
        j+=1
         
      except tf.errors.OutOfRangeError:
        break
    
# %% TESTING

with tf.Session(graph=graph1) as sess:
    sess.run(init)
    saver = tf.train.Saver()
    saver.restore(sess, "./weights_TF_reconnet/weights.ckpt")
    for i in range(X_test.shape[0]):
        feed = {x : np.expand_dims(X_test[i,:,:], axis=0)}
        out = sess.run(output, feed_dict=feed)
        out = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX)
        out = np.squeeze(out)
        out = np.uint8(out)
        
        inp = cv2.normalize(X_test[i,:,:], None, 0, 255, cv2.NORM_MINMAX)
        inp = np.uint8(inp)
        
        plt.figure()
        plt.imshow(np.uint8(inp), cmap='gray')
        plt.show()
        
        plt.figure()
        plt.imshow(out, cmap='gray')
        plt.show()
        
        if i==10:
            break

# %% RECONSTRUCT IMAGES

test_image = cv2.imread('./test_parrot.jpg')
test_image_lum = cv2.cvtColor(test_image, cv2.COLOR_BGR2YCR_CB)
test_image_lum = test_image_lum[:,:,0]
#test_image_lum = cv2.normalize(test_image_lum.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
test_gray = cv2.imread('./test_parrot.jpg',0)

filter_size = 33
stride = 14
(h,w) = test_image_lum.shape
h_iters = ((h - filter_size) // stride) + 1
w_iters = ((w - filter_size) // stride) + 1
recon_img = np.zeros((h,w))

with tf.Session(graph=graph1) as sess:
    sess.run(init)
    saver = tf.train.Saver()
    saver.restore(sess, "./weights_TF_reconnet/weights.ckpt")

    for i in range(h_iters):
        for j in range(w_iters):
            feed = {x : np.expand_dims(test_image_lum[stride*i:filter_size+stride*i, stride*j:filter_size+stride*j], axis=0)}
            out = sess.run(output, feed_dict=feed)
#            out = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX)
            out = np.squeeze(out)
            out = np.uint8(out)
            recon_img[stride*i:filter_size+stride*i, stride*j:filter_size+stride*j] = out
            
    plt.figure()
    plt.imshow(recon_img, cmap='gray')
    plt.show()
    
    plt.figure()
    plt.imshow(test_gray, cmap='gray')
    plt.show()
    