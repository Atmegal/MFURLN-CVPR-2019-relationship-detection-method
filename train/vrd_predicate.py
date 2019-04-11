#for build_rd_network2
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf 
import numpy as np 
from model.config import cfg 
from net.vrd_predicate import MFURLN
from random import shuffle
from model.ass_fun import *
N_cls = cfg.VRD_NUM_CLASS
N_rela = cfg.VRD_NUM_RELA 
N_each_batch = cfg.VRD_BATCH_NUM


index_sp = False
index_cls = False

vnet = MFURLN()
vnet.create_graph(N_each_batch, index_sp, index_cls, N_cls, N_rela)

roidb_path = './input/vrd_roidb.npz'
#roidb_path = '/media/hp/189EA2629EA23860/work/visual_relation/vtranse/input/vrd_roidb.npz'
res_path =  './tf-faster-rcnn-master/lib/output/VRD/vgg16_faster_rcnn_iter_60000.ckpt'
model_path = './result/vrd/predicate/vrd_roid_predicate.ckpt'
test_path = './result/vrd/predicate/vrd_roid_predicate.npz'


roidb_read = read_roidb(roidb_path)
train_roidb = roidb_read['train_roidb'][:3500]
roidb_path = './input/vrd_roidb.npz'
roidb_read = read_roidb(roidb_path)
validation_roidb = roidb_read['val_roidb']
test_roidb = roidb_read['test_roidb']
N_train = len(train_roidb)
N_test = len(test_roidb)
N_round = 13
N_show = 500
N_save = N_train
N_val = len(validation_roidb)

total_var = tf.trainable_variables()
restore_var = [var for var in total_var if 'vgg_16' in var.name]
cls_score_var = [var for var in total_var if 'cls_score' in var.name]
res_var = [item for item in restore_var if item not in cls_score_var]

RD_var = [var for var in total_var if 'RD' in var.name]

saver_res = tf.train.Saver(var_list = restore_var )

saver = tf.train.Saver(max_to_keep = 200)
for var in RD_var:
	print(var)

lr_init = 0.0002
global_step = tf.Variable(0, trainable = False)
learning_rate = tf.train.exponential_decay(lr_init, global_step = global_step, decay_steps = 4000, decay_rate = 0.5)
learning_rate = tf.reduce_max([learning_rate, 0.00001])

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train_loss = vnet.losses['rd_loss']
RD_train = optimizer.minimize(train_loss, var_list = RD_var, global_step = global_step)


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
config = tf.ConfigProto()

f1 = open('result_for_test.txt', "a") 
f1.write('For new')
f1.write('\n') 
f1.close() 
R50_val  = R50_test = 0
merged = tf.summary.merge_all()
with tf.Session(config = config) as sess:
	init = tf.global_variables_initializer()
	sess.run(init)
	saver_res.restore(sess, res_path)

	t = 0.0
	rd_loss = 0.0
	acc = 0.0
	best_acc_val = 0.0
	N_t = range(N_train)
	stop = 0
	for r in range(N_round):
		t = 0.0
		rd_loss = 0.0
		acc = 0.0
		shuffle(N_t)
		#N_t = []
		for roidb_id in N_t:
			roidb_use = train_roidb[roidb_id]
			if len(roidb_use['rela_gt']) == 0:
				continue
			rd_loss_temp, acc_temp = vnet.train_predicate(sess, roidb_use, RD_train, t, t)
			rd_loss = rd_loss + rd_loss_temp
			acc = acc + acc_temp
			t = t + 1.0
			if t % N_show == 0:
				lr = sess.run(learning_rate)
				print("t: {0}, rd_loss: {1}, acc: {2}, lr: {3}".format(t, rd_loss/t, acc/t, lr))
		rd_loss_val = 0.0
		acc_val = 0.0
		t_val = 0.0
		rec_val = 0.0
		t_rec = 0.0
		pred_roidb = []

		for val_id in range(N_val):
			#pdb.set_trace()
			roidb_use = validation_roidb[val_id]
			if len(roidb_use['rela_gt']) == 0:
				continue
			rd_loss_temp, acc_temp, all_value = vnet.test_predicate(sess, roidb_use, val_id)
			t_val = t_val + 1
			pred_roidb_temp = {'pred_rela': rd_loss_temp, 'pred_rela_score': acc_temp,
							'sub_box_dete': roidb_use['sub_box_gt'], 'obj_box_dete': roidb_use['obj_box_gt'],
							'sub_dete': roidb_use['sub_gt'], 'obj_dete': roidb_use['obj_gt']}
			pred_roidb.append(pred_roidb_temp)
		R50_val, num_right50 = rela_recall(validation_roidb, pred_roidb, 50)
		print("val: acc: {0}".format(R50_val))
		if R50_val > best_acc_val:
			best_acc_val = R50_val
			save_path = model_path
			saver.save(sess, save_path)
			print("saving model")
			stop = 0
			rd_loss_test = 0.0
			acc_test = 0.0
			t_test = 0.0
			rec_test = 0.0
			t_rec = 0.0
			pred_roidb = []
			for test_id in range(N_test):
				roidb_use = test_roidb[test_id]
				if len(roidb_use['rela_gt']) == 0:
					continue
				pred_rela, pred_rela_score, pred_rela_all  = vnet.test_predicate(sess, roidb_use, test_id)

				pred_roidb_temp = {'pred_rela': pred_rela, 'pred_rela_score': pred_rela_score,'pred_rela_all': pred_rela_all,
							'sub_box_dete': roidb_use['sub_box_gt'], 'obj_box_dete': roidb_use['obj_box_gt'],
							'sub_dete': roidb_use['sub_gt'], 'obj_dete': roidb_use['obj_gt']}
				pred_roidb.append(pred_roidb_temp)
			R50_test, num_right50 = rela_recall(test_roidb, pred_roidb, 50)
			save_path = test_path 
			roidb = {}
			roidb['pred_roidb'] = pred_roidb
			np.savez(save_path, roidb=roidb)
			print("test: acc: {0}".format( R50_test))
		#stop += 1
		#if stop > 2:
		#	break		
    		f1 = open('result_for_test.txt', "a")  
       		f1.write(str(r))
    		f1.write('\t')
   		f1.write('detection prediction: ')
		f1.write('\t')
   		f1.write('train:')
		f1.write('\t')
    		f1.write(str(rd_loss/t))
		f1.write('\t')
    		f1.write(str(acc/t))
		f1.write('\t')
   		f1.write('validation:')
		f1.write('\t')
    		f1.write(str(R50_val))
		f1.write('\t')
   		f1.write('test:')
		f1.write('\t')
    		f1.write(str(R50_test))
		f1.write('\n')
    		f1.close() 
		print("tra: rd_loss: {0}, acc: {1}".format(rd_loss/t, acc/t))
    		f1 = open('result_train.txt', "a")  
    		f1.write(str(r))
    		f1.write('\t')
   		f1.write(str(t))
    		f1.write('\t')
   		f1.write('detection prediction: ')
		f1.write('\t')
    		f1.write(str(rd_loss/t))
		f1.write('\t')
		f1.write(str(acc/t))
		f1.write('\n')
    		f1.close() 
    






