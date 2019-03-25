# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
try:
  import cPickle as pickle
except ImportError:
  import pickle
import os
import math
import pdb
from utils.timer import Timer
from utils.cython_nms import nms
from utils.blob import im_list_to_blob

from model.config import cfg, get_output_dir
from model.bbox_transform import clip_boxes, bbox_transform_inv

def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)

def _get_blobs(im):
  """Convert an image and RoIs within that image into network inputs."""
  blobs = {}
  blobs['data'], im_scale_factors = _get_image_blob(im)

  return blobs, im_scale_factors

def _clip_boxes(boxes, im_shape):
  """Clip boxes to image boundaries."""
  # x1 >= 0
  boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
  # y1 >= 0
  boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
  # x2 < im_shape[1]
  boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
  # y2 < im_shape[0]
  boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
  return boxes

def _rescale_boxes(boxes, inds, scales):
  """Rescale boxes according to image rescaling."""
  for i in range(boxes.shape[0]):
    boxes[i,:] = boxes[i,:] / scales[int(inds[i])]

  return boxes

def im_detect(sess, net, im):
  blobs, im_scales = _get_blobs(im)
  assert len(im_scales) == 1, "Only single-image batch implemented"

  im_blob = blobs['data']
  blobs['im_info'] = np.array([im_blob.shape[1], im_blob.shape[2], im_scales[0]], dtype=np.float32)

  cls_score, scores, bbox_pred, rois, features = net.test_image(sess, blobs['data'], blobs['im_info'])
#################################################################
  cls_score = np.reshape(cls_score, [scores.shape[0], -1]) 
  features = np.reshape(features, [scores.shape[0], -1]) 
#####################################################################
  boxes = rois[:, 1:5] / im_scales[0]
  scores = np.reshape(scores, [scores.shape[0], -1])
  bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])

  if cfg.TEST.BBOX_REG:
    # Apply bounding-box regression deltas
    box_deltas = bbox_pred
    pred_boxes = bbox_transform_inv(boxes, box_deltas)
    pred_boxes = _clip_boxes(pred_boxes, im.shape)
  else:
    # Simply repeat the boxes, once for each class
    pred_boxes = np.tile(boxes, (1, scores.shape[1]))

  return scores, pred_boxes, cls_score, features

def apply_nms(all_boxes, thresh):
  """Apply non-maximum suppression to all predicted boxes output by the
  test_net method.
  """
  num_classes = len(all_boxes)
  num_images = len(all_boxes[0])
  nms_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
  for cls_ind in range(num_classes):
    for im_ind in range(num_images):
      dets = all_boxes[cls_ind][im_ind]
      if dets == []:
        continue

      x1 = dets[:, 0]
      y1 = dets[:, 1]
      x2 = dets[:, 2]
      y2 = dets[:, 3]
      scores = dets[:, 4]
      inds = np.where((x2 > x1) & (y2 > y1) & (scores > cfg.TEST.DET_THRESHOLD))[0]
      dets = dets[inds,:]
      if dets == []:
        continue

      keep = nms(dets, thresh)
      if len(keep) == 0:
        continue
      nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
  return nms_boxes

def test_net(sess, net, imdb, weights_filename, max_per_image=100, thresh=0.05):
  np.random.seed(cfg.RNG_SEED)
  """Test a Fast R-CNN network on an image database."""
  num_images = len(imdb.image_index)
  # all detections are collected into:
  #  all_boxes[cls][image] = N x 5 array of detections in
  #  (x1, y1, x2, y2, score)
  all_boxes = [[[] for _ in range(num_images)]
         for _ in range(imdb.num_classes)]
  all_scores = [[[] for _ in range(num_images)]
         for _ in range(imdb.num_classes)]
  all_features = [[[] for _ in range(num_images)]
         for _ in range(imdb.num_classes)]

  output_dir = get_output_dir(imdb, weights_filename)
  print("output_dir is:", output_dir)
  print("\n")
  # timers
  _t = {'im_detect' : Timer(), 'misc' : Timer()}

  for i in range(num_images):
    im = cv2.imread(imdb.image_path_at(i))

    _t['im_detect'].tic()
    scores, boxes, cls_score, features = im_detect(sess, net, im)
    _t['im_detect'].toc()

    _t['misc'].tic()

    # skip j = 0, because it's the background class
    for j in range(1, imdb.num_classes):
      inds = np.where(scores[:, j] > thresh)[0]
      cls_scores = scores[inds, j]
      cls_boxes = boxes[inds, j*4:(j+1)*4]
      cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
        .astype(np.float32, copy=False)
      keep = nms(cls_dets, cfg.TEST.NMS)
      cls_dets = cls_dets[keep, :]
      sco = cls_score[inds]
      sco = sco[keep]
      fea = features[inds]
      fea = fea[keep]
      all_boxes[j][i] = cls_dets
      all_scores[j][i] = sco
      all_features[j][i] = fea

    # Limit to max_per_image detections *over all classes*
    if max_per_image > 0:
      image_scores = np.hstack([all_boxes[j][i][:, -1]
                    for j in range(1, imdb.num_classes)])
      if len(image_scores) > max_per_image:
        image_thresh = np.sort(image_scores)[-max_per_image]
        for j in range(1, imdb.num_classes):
          keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
          all_boxes[j][i] = all_boxes[j][i][keep, :]
    _t['misc'].toc()

    print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
        .format(i + 1, num_images, _t['im_detect'].average_time,
            _t['misc'].average_time))

  det_file = os.path.join(output_dir, 'detections.pkl')
  with open(det_file, 'wb') as f:
    pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

  print('Evaluating detections')
  imdb.evaluate_detections(all_boxes, output_dir)


def extract_net(sess, net, imdb, weights_filename, roidb, max_per_image=100, thresh=0.05):
  #change from test_net, and this function is used to extract features from conv5 of vgg16
  np.random.seed(cfg.RNG_SEED)
  """Test a Fast R-CNN network on an image database."""
  num_images = len(imdb.image_index)
  # all detections are collected into:
  #  all_boxes[cls][image] = N x 5 array of detections in
  #  (x1, y1, x2, y2, score)
  all_boxes = [[[] for _ in range(num_images)]
         for _ in range(imdb.num_classes)]

  output_dir = get_output_dir(imdb, weights_filename)
  print("output_dir is:", output_dir)
  print("\n")
  # timers
  _t = {'im_detect' : Timer(), 'misc' : Timer()}

  data_input = []
  #N_save is the number of processed images which will be saved into one file
  N_save = 100
  for i in range(1000):
    im = cv2.imread(imdb.image_path_at(i))

    _t['im_detect'].tic()
    feature_maps = im_extract(sess, net, im)

    fea = np.squeeze(feature_maps)
    data_temp = {}
    data_temp['img_path'] = imdb.image_path_at(i)
    data_temp['box'] = roidb[i]['boxes']
    data_temp['img_shape'] = np.shape(im)
    data_temp['fea'] = fea
    data_input.append(data_temp)
    if (i+1) % N_save == 0:
      file_name_of_input = '/home/yangxu/project/rd/input/input'+ format(int((i+1)/N_save), '03')+'.npz'
      np.savez(file_name_of_input, data_input=data_input)
      data_input = []



    _t['im_detect'].toc()

    _t['misc'].tic()


    print('im_extract: {:d}/{:d} {:.3f}s {:.3f}s' \
        .format(i + 1, num_images, _t['im_detect'].average_time,
            _t['misc'].average_time))
    print("i is: {0}, the path of image is {1}, shape of the image is {2}, the shape of the feature map is{3}\n".format(i, imdb.image_path_at(i), np.shape(im), np.shape(fea)))



def im_extract(sess, net, im):
  #extract feature maps of conv5 from vgg16
  blobs, im_scales = _get_blobs(im)
  assert len(im_scales) == 1, "Only single-image batch implemented"

  im_blob = blobs['data']

  feature_maps = net.extract_head(sess, blobs['data'])

  return feature_maps

def test_net_vg(sess, net, roidb, output_dir, num_classes, max_per_image=100, thresh=0.05):
  np.random.seed(cfg.RNG_SEED)
  num_images = len(roidb)
  # all detections are collected into:
  #  all_boxes[cls][image] = N x 5 array of detections in
  #  (x1, y1, x2, y2, score)
  all_boxes = [[[] for _ in range(num_images)]
         for _ in range(num_classes)]
  all_scores = [[[] for _ in range(num_images)]
         for _ in range(num_images)]
  all_features = [[[] for _ in range(num_images)]
         for _ in range(num_images)]
  print("output_dir is:", output_dir)
  print("\n")
  # timers
  _t = {'im_detect' : Timer(), 'misc' : Timer()}
  dete_pred = []

  for i in range(num_images):
    print("i: {0}/{1}".format(i, num_images))
    dete_pred_temp = {'image': roidb[i]['image'],
                      'gt_boxes': roidb[i]['boxes'],
                      'gt_classes': roidb[i]['gt_classes']}

    im = cv2.imread(roidb[i]['image'])

    _t['im_detect'].tic()
    scores, boxes, cls_score, features = im_detect(sess, net, im)
    _t['im_detect'].toc()

    _t['misc'].tic()

    # skip j = 0, because it's the background class
    test = np.array([])
    testb = np.array([])
    testl = np.array([])    
    for j in range(1, num_classes):
      inds = np.where(scores[:, j] > thresh)[0]
      cls_scores = scores[inds, j]
      cls_boxes = boxes[inds, j*4:(j+1)*4]
      cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
        .astype(np.float32, copy=False)
      keep = nms(cls_dets, cfg.TEST.NMS)
      cls_dets = cls_dets[keep, :]
      test = np.append(cls_dets, test )
      testb = np.append(cls_scores[keep], testb )
      testl = np.append(cls_scores[keep] *0 +j, testl )      
     # sco = cls_score[inds]
    #  sco = sco[keep]
     # fea = features[inds]
     # fea = fea[keep]
      all_boxes[j][i] = cls_dets
     # all_scores[j][i] = sco
     # all_features[j][i] = fea
    #  pdb.set_trace()
    # Limit to max_per_image detections *over all classes*
    if max_per_image > 0:
      image_scores = np.hstack([all_boxes[j][i][:, -1]
                    for j in range(1, num_classes)])
      if len(image_scores) > max_per_image:
        image_thresh = np.sort(image_scores)[-max_per_image]
        for j in range(1, num_classes):
          keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
          all_boxes[j][i] = all_boxes[j][i][keep, :]
       #   all_features[j][i] = all_features[j][i][keep, :]
       #   all_scores[j][i] = all_scores[j][i][keep, :]
    pdb.set_trace()          
    _t['misc'].toc()

    pred_boxes = []
    pred_features = []
    pred_scores = []
    # this is for background
    pred_boxes.append([])
    pred_features.append([])
    pred_scores.append([])
    for j in range(1,num_classes):
      pred_boxes.append(all_boxes[j][i])
    #  pred_features.append(all_features[j][i])
     # pred_scores.append(all_scores[j][i])
    dete_pred_temp['pred_boxes'] = pred_boxes
   # dete_pred_temp['pred_features'] = pred_features
   # dete_pred_temp['pred_scores'] = pred_scores
    dete_pred.append(dete_pred_temp)

  np.savez(output_dir,dete_pred_vg = dete_pred)
  return dete_pred


    
