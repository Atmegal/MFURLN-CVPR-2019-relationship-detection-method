# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 23:45:22 2018

@author: zhan
"""
import numpy as np
import scipy.io as sci        
data = np.load('./input/vrd_roidb.npz')
key = data.keys()
data1 = data[key[0]]
data_i = data1[()]['train_roidb']


confidence = np.ones([100, 100, 70])#Laplace smoothing
for i in range(len(data_i)):
    for j in range(len(data_i[i]['sub_gt'])):
        confidence[np.int(data_i[i]['sub_gt'][j]), np.int(data_i[i]['obj_gt'][j]), np.int(data_i[i]['rela_gt'][j])] +=1
        

confidence_obj = np.ones([100, 70])
for i in range(len(data_i)):
    for j in range(len(data_i[i]['sub_gt'])):
        confidence_obj[np.int(data_i[i]['obj_gt'][j]), np.int(data_i[i]['rela_gt'][j])] +=1
        

confidence_sub = np.ones([100, 70])
for i in range(len(data_i)):
    for j in range(len(data_i[i]['sub_gt'])):
        confidence_sub[np.int(data_i[i]['sub_gt'][j]), np.int(data_i[i]['rela_gt'][j])] +=1
        
#normalize
for i in range(100):
    for j in range(100):
        if np.sum(confidence[i,j])>0:
            confidence[i,j] = confidence[i,j]/np.sum(confidence[i,j])
            

for i in range(100):
        if np.sum(confidence_sub[i]) > 0:
            confidence_sub[i] = confidence_sub[i]/np.sum(confidence_sub[i])
        if np.sum(confidence_obj[i]) > 0:
            confidence_obj[i] = confidence_obj[i]/np.sum(confidence_obj[i])

np.savez('./input/VRD/language_inter', sub_obj = confidence, sub = confidence_sub, obj = confidence_obj)

           
