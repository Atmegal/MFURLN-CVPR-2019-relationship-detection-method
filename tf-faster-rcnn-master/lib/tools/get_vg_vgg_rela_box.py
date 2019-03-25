from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.test import test_net_vg
from model.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, get_output_tb_dir
import argparse
import pprint
import numpy as np
import sys

import tensorflow as tf
from nets.vgg16 import vgg16

output_dir = '/media/zhan/189EA2629EA23860/work/visual_relation/vtranse/tf-faster-rcnn-master/lib/dete_pred_vg.npz'
save_path = '/media/zhan/189EA2629EA23860/work/visual_relation/vtranse/tf-faster-rcnn-master/lib/vg_detected_box.npz'
model_path = '/media/zhan/189EA2629EA23860/vrd-master/tf-faster-rcnn-master/lib/output/vgg16/vg/vgg16_faster_rcnn_iter_1050000.ckpt'

num_classes = 201
vg_roidb = np.load('vg_roidb.npz')
roidb_temp = vg_roidb['roidb']
roidb = roidb_temp[()]
train_roidb = roidb['train_roidb']
val_roidb = roidb['val_roidb']
test_roidb = roidb['test_roidb']
N_train = len(train_roidb)
N_temp = np.int32(N_train/2)
print(N_temp)
train_roidb_temp = train_roidb[0:N_temp]
net = vgg16()
net.create_architecture("TEST", num_classes, tag = 'default', 
		                   anchor_scales=cfg.ANCHOR_SCALES, 
		                   anchor_ratios=cfg.ANCHOR_RATIOS)
saver = tf.train.Saver()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
	saver.restore(sess, model_path)
	train_detected_box = test_net_vg(sess, net, train_roidb_temp, output_dir, num_classes, max_per_image=300, thresh=0.05)
	test_detected_box = test_net_vg(sess, net, test_roidb, output_dir, num_classes, max_per_image=300, thresh=0.05)
	val_detected_box = test_net_vg(sess, net, val_roidb, output_dir, num_classes, max_per_image=300, thresh=0.05)
vg_detected_box = {'train_detected_box': train_detected_box,
			'val_detected_box': val_detected_box,
					'test_detected_box': test_detected_box}
np.savez(save_path, vg_detected_box = vg_detected_box)
