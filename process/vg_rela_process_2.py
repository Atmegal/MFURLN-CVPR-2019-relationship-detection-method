#data for relationship detection
#need vg_roidb.npz and vg_detected_box.npz
import numpy as np 
from model.config import cfg 
from model.ass_fun import *
import gc
N_each_batch = cfg.VG_BATCH_NUM_RELA
N_each_pair = cfg.VG_AU_PAIR
iou_l = cfg.VG_IOU_TRAIN
roidb_path = cfg.DIR + 'MFURLN/input/vg_roidb2.npz'
detected_box_path = cfg.DIR + 'MFURLN/input/vg_detected_box.npz'

roidb_read = read_roidb(roidb_path)
#train_roidb = roidb_read['train_roidb'][:-5000]
val_roidb = roidb_read['train_roidb'][-5000:]
test_roidb = roidb_read['test_roidb']
del roidb_read
gc.collect()
roidb_read = read_roidb(detected_box_path)
#train_detected_box = roidb_read['train_detected_box'][:-5000]
val_detected_box = roidb_read['train_detected_box'][-5000:]
test_detected_box = roidb_read['test_detected_box']
roidb_read = []
#N_train = len(train_roidb)
N_val = len(val_roidb)
N_test = len(test_roidb)
del roidb_read
gc.collect()
roidb = []
for i in range(N_val):
	if (i+1)%100 == 0:
		print(i+1)
	val_roidb_use = val_roidb[i]
	val_detected_box_use = val_detected_box[i]
	roidb_temp = generate_test_rela_roidb(val_roidb_use, val_detected_box_use, N_each_batch)
	roidb.append(roidb_temp)
val_roidb_new = roidb

roidb = []
for i in range(N_test):
	if (i+1)%100 == 0:
		print(i+1)
	test_roidb_use = test_roidb[i]
	test_detected_box_use = test_detected_box[i]
	roidb_temp = generate_test_rela_roidb(test_roidb_use, test_detected_box_use, N_each_batch)
	roidb.append(roidb_temp)
test_roidb_new = roidb

roidb = {}
roidb['test_roidb'] = test_roidb_new
save_path = cfg.DIR + 'MFURLN/input/vg_rela_roidb_test.npz'
np.savez(save_path, roidb=roidb)
roidb = {}
roidb['val_roidb'] = val_roidb_new
save_path = cfg.DIR + 'MFURLN/input/vg_rela_roidb_val.npz'
np.savez(save_path, roidb=roidb)

