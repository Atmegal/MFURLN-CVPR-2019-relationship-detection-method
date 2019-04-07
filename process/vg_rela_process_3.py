#data for relationship detection
#need vg_roidb.npz and vg_detected_box.npz
import numpy as np 
from model.config import cfg 
from model.ass_fun import *
import gc
roidb = []
num = 0
t = 0
j = 0
train_roidb_new = []
for i in range(14):
	save_path = cfg.DIR + 'MFURLN/input/vg_process/vg_rela_roidb_'+str(j)+'.npz'
	roidb_read = read_roidb(save_path)
	train_roidb = roidb_read['train_roidb']
	print(len(train_roidb[0]['sub_dete']))
	for i in range(len(train_roidb)):

	train_roidb_new.extend(train_roidb)
	del roidb_read
	del train_roidb
	gc.collect()
	print(len(train_roidb_new))
save_path = cfg.DIR + 'vtranse/input/vg_rela_roidb_train.npz'
roidb = {}
roidb['train_roidb'] = train_roidb_new
#roidb['test_roidb'] = test_roidb_new
#roidb['val_roidb'] = val_roidb_new

np.savez(save_path, roidb=roidb)
