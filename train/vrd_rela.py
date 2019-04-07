from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf 
import numpy as np 
from model.config import cfg 
from model.ass_fun_1 import *
from net.vrd_rela import VTranse
from random import shuffle
from fuse_res.func import rela_recall1, phrase_recall1, rela_recall2, phrase_recall2
N_cls = cfg.VRD_NUM_CLASS
N_rela = cfg.VRD_NUM_RELA
N_each_batch = cfg.VRD_BATCH_NUM_RELA
lr_init = cfg.VRD_LR_INIT

index_sp = False
index_cls = False

vnet = VTranse()
vnet.create_graph(N_each_batch, index_sp, index_cls, N_cls, N_rela)

roidb_path = './input/vrd_rela.npz'
res_path =  './tf-faster-rcnn-master/lib/output/VRD/vgg16_faster_rcnn_iter_60000.ckpt'
model_path = './result/vrd/vrd_roid_rela.ckpt'
test_path = './result/vrd/vrd_roid_rela.npz'
roidb_read = read_roidb(roidb_path)
train_roidb = roidb_read['train_roidb']
val_roidb = roidb_read['val_roidb']

N_train = len(train_roidb)
N_val = len(val_roidb)

N_round = cfg.VRD_TRAIN_ROUND
N_show = 500
N_save = N_train 

total_var = tf.trainable_variables()
restore_var = [var for var in total_var if 'vgg_16' in var.name]
cls_score_var = [var for var in total_var if 'cls_score' in var.name]
res_var = [item for item in restore_var if item not in cls_score_var]

saver_res = tf.train.Saver(var_list = restore_var)

RD_var = [var for var in total_var if 'RD' in var.name]

saver = tf.train.Saver(max_to_keep = 200)

for var in RD_var:
    print(var)

lr_init = 0.0002
global_step = tf.Variable(0, trainable = False)
learning_rate = tf.train.exponential_decay(lr_init, global_step = global_step, decay_steps = 5000, decay_rate = 0.5)
learning_rate = tf.reduce_max([learning_rate, 0.00001])
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate  )
train_loss = vnet.losses['rd_loss']

RD_train = optimizer.minimize(train_loss, var_list = RD_var , global_step = global_step)

rela_R50 = 0.0
rela_R100 = 0.0
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    saver_res.restore(sess, res_path)
    t = 0.0
    rd_loss = 0.0
    acc = 0.0
    best_val = 0.0
    N_t = range(N_train)
    stop = 0
    for r in range(N_round):
        t = 0.0
        rd_loss = 0.0
        acc = 0.0
        shuffle(N_t)
	if stop == 2:
		break
        for roidb_id in N_t:
            roidb_use = train_roidb[roidb_id]
            if len(roidb_use['rela_dete']) == 0:
                continue
            a = 3
            rd_loss_temp, acc_temp = vnet.train_rela(sess, roidb_use, RD_train, a, t)
            rd_loss = rd_loss + rd_loss_temp
            acc = acc + acc_temp
            t = t + 1.0
            if t % N_show == 0:
                lr = sess.run(learning_rate)
                print("t: {0}, rd_loss: {1}, acc: {2}, lr: {3}".format(t, rd_loss/t, acc/t, lr))
            if t%N_save == 0:
                rd_loss_val = 0.0
                acc_val = 0.0
                pred_roidb = []
                rR50 = 0.
                rR100 = 0.
                pR50 = 0.
                pR100 = 0.
                num = 0.
		rRso_50  = 0.
		rRso_100  = 0.
		pRso_50 = 0.
		pRso_100 = 0.
                for val_id in range(N_val):
                    roidb_use = val_roidb[val_id]
                    if len(roidb_use['rela_dete']) == 0:
                        continue

                    pred_rela, pred_rela_score, pred_rela_all, sub_rela, obj_rela, sub_score, obj_score  = vnet.test_rela(sess, roidb_use)

                    sub_score = np.reshape( roidb_use['sub_score'], np.shape(pred_rela_score) )
                    obj_score = np.reshape( roidb_use['obj_score'], np.shape(pred_rela_score) )

                    pred_rela_score = pred_rela_score *sub_score *obj_score
                    pred_roidb_temp = {'pred_rela': pred_rela, 'pred_rela_score': pred_rela_score, 'pred_rela_all':pred_rela_all,
                            'sub_box_dete': roidb_use['sub_box_dete'], 'obj_box_dete': roidb_use['obj_box_dete'],
                            'sub_dete': roidb_use['sub_dete']-1, 'obj_dete': roidb_use['obj_dete']-1}

                    rela_R50, rela_num_right50 ,_= rela_recall1([val_roidb[val_id]], [pred_roidb_temp], 50)
                    rela_R100, rela_num_right100 ,_= rela_recall1([val_roidb[val_id]], [pred_roidb_temp], 100)

                    phrase_R50, rela_num_right50 = phrase_recall1([val_roidb[val_id]], [pred_roidb_temp], 50)
                    phrase_R100, rela_num_right100 = phrase_recall1([val_roidb[val_id]], [pred_roidb_temp], 100)


                    rR50 = rR50 + rela_R50
                    rR100 = rR100 + rela_R100
                    pR50 = pR50 + phrase_R50
                    pR100 = pR100 + phrase_R100

                    num = num+rela_num_right50
                    pred_roidb.append(pred_roidb_temp)
                    if (val_id+1)%50 == 0:
                            print("Rela detection:", val_id+1, rR50,rR100, pR50,pR100, num)

                rela_R50 = rR50/num
                rela_R100 = rR100/num
                p_R50= pR50/num
                p_R100= pR100/num
                print("val: rela_R50: {0}, rela_R100: {1}, phrase_R50: {2}, phrase_R100: {3}".format(rela_R50, rela_R100, p_R50, p_R100))
                if rela_R50 + rela_R100 > best_val:
                    best_val = rela_R50 + rela_R100
                    save_path = save_path
                    print("saving model to {0}".format(save_path))
		    stop = 0
                    saver.save(sess, save_path)
		else:
		    stop +=1


        f1 = open('rela_validation_test.txt', "a")
        f1.write( save_path)
        f1.write( '\n')
        f1.write(str(r))
        f1.write('\t')
        f1.write('train: ')
        f1.write('\t')
        f1.write(str(rd_loss/t))
        f1.write('\t')
        f1.write(str(acc/t))
        f1.write('\t')
        f1.write('validation: ')
        f1.write('\t')
        f1.write(str(rela_R50))
        f1.write('\t')
        f1.write(str(rela_R100))
        f1.write('\t')
        f1.write(str(p_R50))
        f1.write('\t')
        f1.write(str(p_R100))
        f1.write('\t')
        f1.write(str(rela_so_R50))
        f1.write('\t')
        f1.write(str(rela_so_R100))
        f1.write('\t')
        f1.write(str(p_so_R50))
        f1.write('\t')
        f1.write(str(p_so_R100))
        f1.write('\n')
        f1.close()








