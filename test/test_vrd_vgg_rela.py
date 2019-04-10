from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf 
import numpy as np 
from model.config import cfg 
from model.ass_fun import *
from net.vrd_rela import MFURLN

N_cls = cfg.VRD_NUM_CLASS
N_rela = cfg.VRD_NUM_RELA 
N_each_batch = cfg.VRD_BATCH_NUM

index_sp = False
index_cls = False

vnet = MFURLN()
vnet.create_graph(N_each_batch, index_sp, index_cls, N_cls, N_rela)

roidb_path = './input/vrd_rela.npz'
res_path =  './tf-faster-rcnn-master/lib/output/VRD/vgg16_faster_rcnn_iter_60000.ckpt'
save_path = './result/vrd/rela/vrd_roid_rela.ckpt'
test_path = './result/vrd/rela/vrd_roid_rela.npz'


saver = tf.train.Saver(max_to_keep = 200)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
roidb_read = read_roidb(roidb_path)
test_roidb = roidb_read['test_roidb']
N_test = len(test_roidb)

rR50 = 0.
rR100 = 0.
pR50 = 0.
pR100 = 0.
rRso_50 = 0.
rRso_100 = 0.
pRso_50 = 0.
pRso_100 = 0.
num = 0.
saver = tf.train.Saver()
with tf.Session() as sess:
	init = tf.global_variables_initializer()
	sess.run(init)
	saver.restore(sess, model_path)

	pred_roidb = []
	for roidb_id in range(N_test):
		roidb_use = test_roidb[roidb_id]
		if len(roidb_use['rela_gt']) == 0:
			pred_roidb.append({})
			print(roidb_id + 1)
			continue

		pred_rela, pred_rela_score, pred_rela_all, sub_rela, obj_rela, sub_score, obj_score = vnet.test_rela(sess, roidb_use)

		sub_score = np.reshape( roidb_use['sub_score'], np.shape(pred_rela_score) )
		obj_score = np.reshape( roidb_use['obj_score'], np.shape(pred_rela_score) )
	
		pred_rela_score = pred_rela_score *sub_score *obj_score
		pred_roidb_temp = {'pred_rela': pred_rela, 'pred_rela_score': pred_rela_score, 'pred_rela_all':pred_rela_all,
							'sub_box_dete': roidb_use['sub_box_dete'], 'obj_box_dete': roidb_use['obj_box_dete'],
							'sub_dete': roidb_use['sub_dete']-1, 'obj_dete': roidb_use['obj_dete']-1}
		pred_roidb.append(pred_roidb_temp)
   		rela_R50, rela_num_right50 = rela_recall1([test_roidb[roidb_id]], [pred_roidb_temp], 50)
   		rela_R100, rela_num_right100 = rela_recall1([test_roidb[roidb_id]], [pred_roidb_temp], 100)
   		phrase_R50, rela_num_right50 = phrase_recall1([test_roidb[roidb_id]], [pred_roidb_temp], 50)
   		phrase_R100, rela_num_right100 = phrase_recall1([test_roidb[roidb_id]], [pred_roidb_temp], 100)
   		rR50 = rR50 + rela_R50
   		rR100 = rR100 + rela_R100
   		pR50 = pR50 + phrase_R50
   		pR100 = pR100 + phrase_R100
		num = num+rela_num_right50

   		if roidb_id%50 == 0:
       			print(roidb_id, rR50,rR100, pR50,pR100, num)
       			print('rela_R50: {0}, rela_R100: {1}'.format(rR50/num, rR100/num))
			print('phrase_R50: {0}, phrase_R100: {1}'.format(pR50/num, pR100/num))
		del pred_roidb_temp 
		gc.collect()	
print('final rela_R50: {0}, rela_R100: {1}'.format(rR50/num, rR100/num))
print('final phrase_R50: {0}, phrase_R100: {1}'.format(pR50/num, pR100/num))
roidb = {}
roidb['pred_roidb'] = pred_roidb
np.savez(save_path, roidb=roidb)
print('rela_R50: {0}, rela_R100: {1}, p_R50: {2}, p_R100: {3}'.format(rela_R50, rela_R100, phrase_R50, phrase_R100))

	
