# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 12:51:54 2018

@author: zhan
"""

# data for predicate detection
import numpy as np 
import xlwt
import h5py
import cv2
from model.config import cfg 
from model.ass_fun import *
import json
import scipy.io as sci

N_each_pred = cfg.VRD_BATCH_NUM
file_path_use = '/media/hp/189EA2629EA23860/work/visual_relation/dataset/VRD/zero_shot.mat'
file_path = [file_path_use]

test_image_path = cfg.DIR + 'dataset/VRD/sg_dataset/sg_test_images/'
image_path = [test_image_path]

save_path ='./input/zero_vrd_roidb.npz'

for r in range(1):
	file_path_use = file_path[r]
	image_path_use = image_path[r]
	save_path_use = save_path[r]
	roidb = []
	data=sci.loadmat(file_path_use)
	image_name = data.keys()
	len_img = 1000
	t = 0
	num = 0
	for image_id in range(len_img):
		if (image_id+1)%1000 == 0:
			print('image id is {0}'.format(image_id+1))
		roidb_temp = {}
		image_full_path = image_path_use + data['imagePath'][0][image_id][0]
		im = cv2.imread(image_full_path)
		if type(im) == type(None):
			continue

		im_shape = np.shape(im)
		im_h = im_shape[0]
		im_w = im_shape[1]

		roidb_temp['image'] = image_full_path
		roidb_temp['width'] = im_w
		roidb_temp['height'] = im_h

		d = data['gt_tuple_label'][0][image_id]
		relation_length = len(d)
		if relation_length == 0:
			continue
		sb_new = np.zeros(shape=[relation_length,4])
		ob_new = np.zeros(shape=[relation_length,4])
		rela = np.zeros(shape=[relation_length,])
		obj = np.zeros(shape=[relation_length,])
		subj = np.zeros(shape=[relation_length,])

		for relation_id in range(relation_length):
			relation = d[relation_id]

			obj[relation_id] = relation[2]-1
			subj[relation_id] = relation[0]-1
			rela[relation_id] = relation[1]-1
   
                	d1 = data['gt_sub_bboxes'][0][image_id][relation_id]
                	ob_new[relation_id][0:4] = data['gt_obj_bboxes'][0][image_id][relation_id]
                	sb_new[relation_id][0:4] = data['gt_sub_bboxes'][0][image_id][relation_id]
		roidb_temp['sub_box_gt'] = sb_new + 0.0
		roidb_temp['obj_box_gt'] = ob_new + 0.0
		roidb_temp['sub_gt'] = subj + 0.0
		roidb_temp['obj_gt'] = obj + 0.0
		roidb_temp['rela_gt'] = rela + 0.0
		roidb_temp['index_pred'] = generate_batch(len(rela), 50)
		roidb.append(roidb_temp)
		num = num+1
print(num)
test_roidb = roidb

roidb = {}
roidb['zero_test_roidb'] = test_roidb

np.savez(save_path, roidb=roidb)
