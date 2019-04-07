from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
from random import shuffle
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
import numpy as np
from model.config import cfg 
from model.ass_fun import *
import pdb
from tensorflow.python.ops import array_ops
class MFURLN(object):
	def __init__(self):
		self.predictions = {}
		self.losses = {}
		self.layers = {}
		self.feat_stride = [16, ]
		self.scope = 'vgg_16'

	def create_graph(self, N_each_batch, index_sp, index_cls, num_classes, num_predicates):
		self.image = tf.placeholder(tf.float32, shape=[1, None, None, 3])
		self.sbox = tf.placeholder(tf.float32, shape=[None, 4])
		self.obox = tf.placeholder(tf.float32, shape=[None, 4])

############################################3
		self.vbox = tf.placeholder(tf.float32, shape=[None, 4])

##################################################
		self.sp_info = tf.placeholder(tf.float32, shape=[None, 8])

		self.rela_label = tf.placeholder(tf.int32, shape=[None,num_predicates])

##############################################################

#############################################################	
		self.sub_cls = tf.placeholder(tf.int32, shape=[None,])
		self.obj_cls = tf.placeholder(tf.int32, shape=[None,])

		self.keep_prob = tf.placeholder(tf.float32)

		self.object = np.load('/media/hp/189EA2629EA23860/work/NLP/oList_word_embedding.npy')
		self.robject = np.load('/media/hp/189EA2629EA23860/work/NLP/rList_word_embedding.npy')
		conf = np.load('/media/hp/189EA2629EA23860/work/visual_relation/vtranse_new/fuse_res/inter.npz')
		self.sub_obj_pred = conf['sub_obj']
		self.sub_pred = conf['sub']
		self.obj_pred = conf['obj']

		self.index_sp = index_sp
		self.index_cls = index_cls
		self.num_classes = num_classes
		self.num_predicates = num_predicates
		self.N_each_batch = N_each_batch

		self.build_dete_network()
		self.build_rd_network()
		self.add_rd_loss()

	def build_dete_network(self, is_training=True):
		net_conv = self.image_to_head(is_training)
		sub_pool5 = self.crop_pool_layer(net_conv, self.sbox, "sub_pool5")
		ob_pool5 = self.crop_pool_layer(net_conv, self.obox, "ob_pool5")
		v_pool5 = self.crop_pool_layer(net_conv, self.vbox, "v_pool5")
		

		sub_fc7 = self.head_to_tail(sub_pool5,is_training,  scope = self.scope, reuse = False)
		ob_fc7 = self.head_to_tail(ob_pool5,is_training,  scope = self.scope,reuse = True)
		v_fc7 = self.head_to_tail(v_pool5, is_training,  scope = self.scope,reuse = True)

		with tf.variable_scope(self.scope, self.scope):
			# region classification
			sub_cls_prob, sub_cls_pred, sub_cls_score = self.region_classification(sub_fc7, is_training, reuse = False)
		with tf.variable_scope(self.scope, self.scope):
			# region classification
			ob_cls_prob, ob_cls_pred, ob_cls_score = self.region_classification(ob_fc7, is_training, reuse = True)

		self.predictions['sub_cls_prob'] = sub_cls_prob
		self.predictions['sub_cls_pred'] = sub_cls_pred
		self.predictions['ob_cls_prob'] = ob_cls_prob
		self.predictions['ob_cls_pred'] = ob_cls_pred
		self.predictions['sub_cls_score'] = sub_cls_score
		self.predictions['ob_cls_score'] = ob_cls_score
		self.layers['sub_pool5'] = sub_pool5
		self.layers['ob_pool5'] = ob_pool5
		self.layers['v_pool5'] = v_pool5
		self.layers['sub_fc7'] = tf.concat([sub_fc7], 1)
		self.layers['ob_fc7'] = tf.concat([ob_fc7], 1)
		self.layers['v_fc7'] = tf.concat([v_fc7], 1)

	def image_to_head(self, is_training, reuse=False):
		with tf.variable_scope(self.scope, self.scope, reuse=reuse):
			net = slim.repeat(self.image, 2, slim.conv2d, 64, [3, 3], 
				trainable=is_training, scope='conv1')
			net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool1')
			net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3],
				trainable=is_training, scope='conv2')
			net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool2')
			net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3],
				trainable=is_training, scope='conv3')
			net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3')
			net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
				trainable=is_training, scope='conv4')
			net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool4')
			net_conv = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
				trainable=is_training, scope='conv5')

			self.layers['head'] = net_conv
			return net_conv

	def head_to_tail(self, pool5, is_training, scope, reuse=False):
		with tf.variable_scope(scope, scope, reuse=reuse):
			pool5_flat = slim.flatten(pool5, scope='flatten')
			fc6 = slim.fully_connected(pool5_flat, 4096, scope='fc6')
			fc6 = slim.dropout(fc6, keep_prob=self.keep_prob, is_training=True, 
					scope='dropout6')
			fc7 = slim.fully_connected(fc6, 4096, scope='fc7')
			fc7 = slim.dropout(fc7, keep_prob=self.keep_prob, is_training=True, 
					scope='dropout7')
			return fc7


	def crop_pool_layer(self, bottom, rois, name, size = 1 ):
		"""
		Notice that the input rois is a N*4 matrix, and the coordinates of x,y should be original x,y times im_scale. 
		"""
		with tf.variable_scope(name) as scope:
			n=tf.to_int32(tf.shape(rois)[0])
			batch_ids = tf.zeros([n,],dtype=tf.int32)
			# Get the normalized coordinates of bboxes
			bottom_shape = tf.shape(bottom)
			print("cls_score's shape: {0}".format(bottom.get_shape()))
			height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self.feat_stride[0])
			width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self.feat_stride[0])
			x1 = tf.slice(rois, [0, 0], [-1, 1], name="x1") / width
			y1 = tf.slice(rois, [0, 1], [-1, 1], name="y1") / height
			x2 = tf.slice(rois, [0, 2], [-1, 1], name="x2") / width
			y2 = tf.slice(rois, [0, 3], [-1, 1], name="y2") / height
			# Won't be back-propagated to rois anyway, but to save time
			bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], 1))
			crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [cfg.POOLING_SIZE*2*size, cfg.POOLING_SIZE*2*size], method='bilinear',
											 name="crops")
			pooling = max_pool(crops, 2, 2, 2, 2, name="max_pooling")
			print("cls_score's shape: {0}".format(pooling.get_shape()))
		return pooling



	def region_classification(self, fc7, is_training, reuse = False):
		cls_score = slim.fully_connected(fc7, self.num_classes, 
										 activation_fn=None, scope='cls_score', reuse=reuse)
		print("cls_score's shape: {0}".format(cls_score.get_shape()))
		cls_prob = tf.nn.softmax(cls_score, name="cls_prob")
		cls_pred = tf.argmax(cls_score, axis=1, name="cls_pred")

		return cls_prob, cls_pred, cls_score

	def build_rd_network(self):
		v_fc = self.layers['v_fc7']
		sub_fc = self.layers['sub_fc7']	
     		ob_fc = self.layers['ob_fc7']


 		sp_info = tf.concat([ self.sp_info ], axis = 1)

		sub_fc = slim.fully_connected(sub_fc, cfg.VTR.VG_R, 
										 activation_fn=tf.nn.relu, scope='RD_s1')			 
		ob_fc = slim.fully_connected(ob_fc, cfg.VTR.VG_R, 
										 activation_fn=tf.nn.relu, scope='RD_o1')
		v_fc = slim.fully_connected(v_fc, cfg.VTR.VG_R, 
										 activation_fn=tf.nn.relu, scope='RD_v1')	 
		sp_info = slim.fully_connected(sp_info, cfg.VTR.VG_R, 
										 activation_fn=tf.nn.relu, scope='RD_p1')

		vector_dic = tf.Variable(self.object, trainable=False, name='VD_vo')
		sub_onehot = tf.matmul(tf.one_hot(self.sub_cls-1, self.num_classes-1), vector_dic)
		obj_onehot = tf.matmul(tf.one_hot(self.obj_cls-1, self.num_classes-1), vector_dic)

		label_s = slim.fully_connected(sub_onehot, cfg.VTR.VG_R, 
										 activation_fn=tf.nn.relu, scope='RD_ls1')
		label_o = slim.fully_connected(obj_onehot , cfg.VTR.VG_R, 
										 activation_fn=tf.nn.relu, scope='RD_lo1')

		v_f1 =  slim.fully_connected( tf.concat([sub_fc, ob_fc, v_fc], axis = 1), cfg.VTR.VG_R, 
										 activation_fn=tf.nn.relu, scope='RD_1v', reuse = False)

		v_f2 =  slim.fully_connected( tf.concat([sp_info], axis = 1), cfg.VTR.VG_R, 
										 activation_fn=tf.nn.relu, scope='RD_2v', reuse = False)

		v_f3 =  slim.fully_connected( tf.concat([label_s, label_o], axis = 1), cfg.VTR.VG_R,
										 activation_fn=tf.nn.relu, scope='RD_3v', reuse = False)

		rela_label = self.rela_label#tf.one_hot( self.rela_label, self.num_predicates)

		full_fc = tf.concat([v_f1, v_f2, v_f3], 1)

		full_fc_d = slim.fully_connected( full_fc, 100, 
										 activation_fn=tf.nn.relu, scope='RD_full_d', reuse = reuse)

		rela_score_full_d = slim.fully_connected(full_fc_d, 1,  
										 activation_fn=None, scope='RD_final_full_d', reuse = reuse)

		full_fc_p = slim.fully_connected(tf.concat([full_fc, full_fc_d], axis = 1), cfg.VTR.VG_R, 
										 activation_fn=tf.nn.relu, scope='RD_full_p', reuse = reuse)
		rela_score_full = slim.fully_connected(full_fc_p, self.num_predicates-1,  
										 activation_fn=None, scope='RD_final_full_p', reuse = reuse)

		self.predictions['rela_pred_all'] = tf.concat([tf.nn.sigmoid(rela_score_full), tf.nn.sigmoid(rela_score_full_1)], axis = 1)

		rela_label, rela_label_1 = tf.split(rela_label, [self.num_predicates-1, 1], 1)

		rela_label_3 = 1 - rela_label_1

		r1 = tf.nn.sigmoid(rela_score_full)


		in_de_loss = - tf.reduce_sum( rela_label_1 * tf.log(tf.clip_by_value( tf.nn.sigmoid(rela_score_full_1), 1e-8, 1.0))) - tf.reduce_sum( ( rela_label_3) * tf.log(tf.clip_by_value(1 - tf.nn.sigmoid(rela_score_full_1), 1e-8, 1.0)))

		rela_loss =  - tf.reduce_sum(  rela_label * tf.log(tf.clip_by_value(r1, 1e-8, 1.0))) - tf.reduce_sum( tf.reshape(tf.reduce_sum((1 - rela_label) * tf.log(tf.clip_by_value(1 - r1, 1e-8, 1.0)), 1), [-1, 1]) * rela_label_3) * 0.5


		self.layers['rd_loss'] =  in_de_loss  + rela_loss
		self.layers['rela_score_f'] = rela_score_full
		self.layers['rela_score_is_pair'] = rela_score_full_d

	def add_rd_loss(self):
		regularizer = tf.contrib.layers.l2_regularizer(0.0001)
		we = tf.trainable_variables()
		RD_var = [var for var in we if 'RD' in var.name]
		rela_label =self.rela_label
		rela_score =  self.predictions['rela_pred_all']


		sub_label =tf.one_hot(self.sub_cls, self.num_classes)
		obj_label =tf.one_hot(self.obj_cls, self.num_classes)

		rd_loss = tf.contrib.layers.apply_regularization(regularizer, weights_list=RD_var) + self.layers['rd_loss'] #* 100 
	
		self.losses['rd_loss'] = rd_loss
		prediction = tf.one_hot(tf.argmax(rela_score, 1),  self.num_predicates)
		correct_prediction = tf.not_equal(tf.argmax(tf.multiply(prediction, tf.cast(rela_label, tf.float32)), 1), False)

		self.losses['acc'] = tf.reduce_mean( tf.cast(correct_prediction, tf.float32) )

		rela_pred = tf.argmax(rela_score, 1)
		self.predictions['rela_pred_train'] = rela_pred
		rela_score = tf.nn.sigmoid(self.layers['rela_score_f']) 

		rela_pred = tf.argmax(rela_score, 1)
		self.predictions['rela_pred'] = rela_pred
		rela_max_prob = tf.reduce_max(rela_score, 1)

		r_0 = tf.nn.sigmoid(self.layers['rela_score_is_pair'])
	
		r_0 = 1 - tf.reshape(r_0, tf.shape(rela_max_prob))	

		self.predictions['rela_max_prob'] = rela_max_prob * r_0 


	def train_rela(self, sess, roidb_use, RD_train, N_t, t):
		im, im_scale, w, h, im_visual, w1, h1 = im_preprocess(roidb_use['image'])
		batch_size = self.N_each_batch
		batch_num = np.min(len(roidb_use['right_index'])-1, 0)/batch_size
		#N_each_batch
		batch_num_w = np.min(len(roidb_use['wrong_index'])-1, 0)/(self.N_each_batch)
		#batch_id_w = 0
		RD_loss = 0.0
		acc = 0.0
		shuffle(roidb_use['right_index'])
		shuffle(roidb_use['wrong_index'])
		num = 0.
		for batch_id in range(np.int32(batch_num) + 1):
                  
			blob = get_blob_rela2(roidb_use, im_scale, w, h, self.index_sp, batch_size, batch_id, 0, N_t, w1, h1)
			
			if len(blob) == 0:
				print(batch_id)
				#pdb.set_trace()
				continue
			feed_dict = {self.image: im,
				self.keep_prob: 0.5,
				self.sbox: blob['sub_box'], 
				self.obox: blob['obj_box'], 
				self.vbox: blob['vis_box'],
					self.rela_label: blob['rela'],
					self.sub_cls: blob['sub_gt'],
					self.obj_cls: blob['obj_gt'],
						self.sp_info: blob['sp_info']
						}	
			_, losses = sess.run([RD_train,self.losses], feed_dict = feed_dict)
			RD_loss += losses['rd_loss']
			acc += losses['acc']
			num += 1
			if t% 1000 == 0:
				print(losses['ls'] )

		if num == 0:
			return RD_loss, acc
		RD_loss = RD_loss/(num)
		acc = acc/num

		return RD_loss, acc

	def test_rela(self, sess, roidb_use):
		im, im_scale , w, h, im_visual, w1, h1 = im_preprocess(roidb_use['image'])
		batch_num = len(roidb_use['index_rela'])/self.N_each_batch
		pred_rela = np.zeros([len(roidb_use['index_rela']),])
		pred_rela_score = np.zeros([len(roidb_use['index_rela']),])
		pred_rela_all = np.zeros([len(roidb_use['index_rela']),self.num_predicates * 1 ])
		sub_rela = np.zeros([len(roidb_use['index_rela']),])
		sub_score = np.zeros([len(roidb_use['index_rela']),])
		obj_rela = np.zeros([len(roidb_use['index_rela']),])
		obj_score = np.zeros([len(roidb_use['index_rela']),])		
		for batch_id in range(np.int32(batch_num)):
			blob = get_blob_rela1(roidb_use, im_scale, w, h, w1, h1, self.index_sp, self.N_each_batch, batch_id)
			is_pair = np.array((blob['rela'] == 70 )+ 0)
			feed_dict = {self.image: im, 
				self.keep_prob: 1,
				self.sbox: blob['sub_box'], 
				self.obox: blob['obj_box'], 
				self.vbox: blob['vis_box'],
					self.sub_cls: blob['sub_gt'],
					self.obj_cls: blob['obj_gt']
						self.sp_info: blob['sp_info']
						}
			predictions = sess.run(self.predictions, feed_dict = feed_dict)
			#w1 = sess.run(self.layers['w'], feed_dict = feed_dict)
			pred_rela[batch_id*self.N_each_batch:(batch_id+1)*self.N_each_batch] = predictions['rela_pred'][:]
			sub_rela[batch_id*self.N_each_batch:(batch_id+1)*self.N_each_batch] = predictions['sub_dete'][:]
			obj_rela[batch_id*self.N_each_batch:(batch_id+1)*self.N_each_batch] = predictions['obj_dete'][:]
			sub_score[batch_id*self.N_each_batch:(batch_id+1)*self.N_each_batch] = predictions['sub_score'][:]
			obj_score[batch_id*self.N_each_batch:(batch_id+1)*self.N_each_batch] = predictions['obj_score'][:]

			pred_rela_score[batch_id*self.N_each_batch:(batch_id+1)*self.N_each_batch] = predictions['rela_max_prob'][:]
			pred_rela_all[batch_id*self.N_each_batch:(batch_id+1)*self.N_each_batch,:] = predictions['rela_pred_all'][:,:]
		N_rela = len(roidb_use['rela_dete'])
		pred_rela = pred_rela[0:N_rela]
		pred_rela_score = pred_rela_score[0:N_rela]
		pred_rela_all = pred_rela_all[0:N_rela,:]
		sub_rela = sub_rela[0:N_rela]
		obj_rela = obj_rela[0:N_rela]
		sub_score = sub_score[0:N_rela]
		obj_score = obj_score[0:N_rela]

		return pred_rela, pred_rela_score, pred_rela_all, sub_rela, obj_rela, sub_score, obj_score
def conv(x, h, w, K, s_y, s_x, name, relu = True, reuse=False, padding='SAME'):
	"""
	Args:
		x: input
		h: height of filter
		w: width of filter
		K: number of filters
		s_y: stride of height of filter
		s_x: stride of width of filter
	"""
	#c means the number of input channels
	c = int(x.get_shape()[-1])

	with tf.variable_scope(name, reuse=reuse) as scope:
		weights = tf.get_variable('weights', shape=[h,w,c,K])
		biases = tf.get_variable('biases', shape=[K])

		conv_value = tf.nn.conv2d(x, weights, strides = [1,s_y,s_x,1], padding = padding)
		add_baises_value = tf.reshape(tf.nn.bias_add(conv_value, biases), tf.shape(conv_value))
		if relu==True:
			relu_value = tf.nn.relu(add_baises_value, name=scope.name)
		else:
			relu_value = add_baises_value

		return relu_value

def fc(x,K,name,relu=True,reuse=False):
	"""
	Args:
		x: input
		K: the dimension of the output
	"""
	#c means the number of input channels

	c = int(x.get_shape()[1])
	with tf.variable_scope(name, reuse=reuse) as scope:
		weights = tf.get_variable('weights', shape=[c,K])
		biases = tf.get_variable('biases',shape=[K])
		relu_value = tf.nn.xw_plus_b(x,weights,biases,name = scope.name)
		if relu:
			result_value = tf.nn.relu(relu_value)
		else:
			result_value = relu_value
		return result_value

def max_pool(x, h, w, s_y, s_x, name, padding='SAME'):
	return tf.nn.max_pool(x, ksize=[1,h,w,1], strides=[1, s_x, s_y, 1], padding=padding, name=name)

def avg_pool(x, h, w, s_y, s_x, name, padding='SAME'):
	return tf.nn.avg_pool(x, ksize=[1,h,w,1], strides=[1, s_x, s_y, 1], padding=padding, name=name)

def dropout(x, keep_prob):
	return tf.nn.dropout(x, keep_prob)

def leaky_relu(x, alpha):
	return tf.maximum(x, alpha * x)
def conv_op(input_op,name,kh,kw,n_out,dh,dw,p):
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+"kernel",
                                 shape=[kh,kw,n_in,n_out],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(input_op, kernel, (1,dh,dw,1),padding='SAME')
        bias_init_val = tf.constant(0.0, shape=[n_out],dtype=tf.float32)
        biases = tf.Variable(bias_init_val , trainable=True , name='bias')
        z = tf.nn.bias_add(conv,biases)
        activation = tf.nn.relu(z,name=scope)
        return activation

def fc_op(input_op,name,n_out,p):
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+'kernel',
                                 shape=[n_in,n_out],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.Variable(tf.constant(0.1,shape=[n_out],dtype=tf.float32),name='bias')
        activation = tf.nn.relu_layer(input_op,kernel,biases,name=scope)
        return activation

def mpool_op(input_op,name,kh,kw,dh,dw):
    return tf.nn.max_pool(input_op,ksize=[1,kh,kw,1],strides=[1,dh,dw,1],padding='VALID',name=name)
