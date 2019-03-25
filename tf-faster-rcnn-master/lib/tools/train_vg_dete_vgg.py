from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.train_val import train_net
from model.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, get_output_tb_dir
import argparse
import pprint
import numpy as np
import sys
import os
import tensorflow as tf
from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from nets.mobilenet_v1 import mobilenetv1

class imdb(object):
	def __init__(self, num_classes):
		self.num_classes = num_classes

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

output_dir = 'output/vgg16/vgg_2/default'
tb_dir = 'output/vgg16/vgg_2/tb'
N_obj = 201
vg_roidb = np.load('/media/hp/189EA2629EA23860/work/visual_relation/tf-faster-rcnn-master_1/lib/vg_roidb.npz')
roidb_temp = vg_roidb['roidb']
roidb = roidb_temp[()]
train_roidb = roidb['train_roidb']
test_roidb = roidb['val_roidb']

vg_imdb = imdb(N_obj)
net = vgg16()
roidb = train_roidb
valroidb = test_roidb
	
pretrained_model = 'output/vgg16/coco_2014_train+coco_2014_valminusminival/vgg16_faster_rcnn_iter_1190000.ckpt'

train_net(net, vg_imdb, roidb, valroidb, output_dir, tb_dir, pretrained_model = pretrained_model, max_iters = 1200000)
