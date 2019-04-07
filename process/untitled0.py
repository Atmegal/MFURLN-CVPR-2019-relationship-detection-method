# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 15:46:02 2018

@author: hp
"""

import numpy as np        
data = np.load('/media/hp/189EA2629EA23860/work/visual_relation/vtranse/input/vg_roidb2.npz')
key = data.keys()
data1 = data[key[0]]
data_i = data1[()]['train_roidb'][:-5000]


confidence = np.ones([200, 200, 100])
for i in range(len(data_i)):
    for j in range(len(data_i[i]['sub_gt'])):
        confidence[np.int(data_i[i]['sub_gt'][j]), np.int(data_i[i]['obj_gt'][j]), np.int(data_i[i]['rela_gt'][j])] +=1
        

confidence_obj = np.ones([200, 100])
for i in range(len(data_i)):
    for j in range(len(data_i[i]['sub_gt'])):
        confidence_obj[np.int(data_i[i]['obj_gt'][j]), np.int(data_i[i]['rela_gt'][j])] +=1
        

confidence_sub = np.ones([200, 100])
for i in range(len(data_i)):
    for j in range(len(data_i[i]['sub_gt'])):
        confidence_sub[np.int(data_i[i]['sub_gt'][j]), np.int(data_i[i]['rela_gt'][j])] +=1
        

for i in range(100):
    for j in range(100):
        if np.sum(confidence[i,j])>0:
            confidence[i,j] = confidence[i,j]/np.sum(confidence[i,j])
            

for i in range(100):
        if np.sum(confidence_sub[i]) > 0:
            confidence_sub[i] = confidence_sub[i]/np.sum(confidence_sub[i])
        if np.sum(confidence_obj[i]) > 0:
            confidence_obj[i] = confidence_obj[i]/np.sum(confidence_obj[i])
np.savez('/media/hp/189EA2629EA23860/work/visual_relation/vtranse_new/fuse_res/inter_VG', sub_obj = confidence, sub = confidence_sub, obj = confidence_obj)
