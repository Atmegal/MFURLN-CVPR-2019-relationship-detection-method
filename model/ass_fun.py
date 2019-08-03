import numpy as np
import cv2
import os
from model.config import cfg
import PIL.Image as Image
import pdb
from random import shuffle


def generate_batch(N_total, N_each):
    """
	This file is used to generate index of the training batch.

	Arg:
		N_total:
		N_each:
	out_put:
		index_box: the corresponding index
	"""
    num_batch = np.int32(N_total / N_each)
    if N_total % N_each == 0:
        index_box = range(N_total)
    else:
        index_box = np.empty(shape=[N_each * (num_batch + 1)], dtype=np.int32)
        index_box[0:N_total] = range(N_total)
        N_rest = N_each * (num_batch + 1) - N_total
        index_box[N_total:] = np.random.randint(0, N_total, N_rest)
    return index_box


def generate_batch_bal(labels, N_each):
    N_total = len(labels)
    num_batch = np.int32(N_total / N_each)
    if N_total % N_each == 0:
        index_box = range(N_total)
    else:
        index_box = np.empty(shape=[N_each * (num_batch + 1)], dtype=np.int32)
        index_box[0:N_total] = range(N_total)
        N_rest = N_each * (num_batch + 1) - N_total

        unique_labels = np.unique(labels, axis=0)
        N_unique = len(unique_labels)
        num_label = np.zeros([N_unique, ])
        for ii in range(N_unique):
            num_label[ii] = np.sum(labels == unique_labels[ii])
        prob_label = np.sum(num_label) / num_label
        prob_label = prob_label / np.sum(prob_label)
        index_rest = np.random.choice(N_unique, size=[N_rest, ], p=prob_label)
        for ii in range(N_rest):
            ind = index_rest[ii]
            ind2 = np.where(labels == unique_labels[ind])[0]
            a = np.random.randint(len(ind2))
            index_box[N_total + ii] = ind2[a]
    return index_box


def read_roidb(roidb_path):
    roidb_file = np.load(roidb_path)
    key = roidb_file.keys()[0]
    roidb_temp = roidb_file[key]
    roidb = roidb_temp[()]
    return roidb


def compute_iou(box, proposal):
    """
	compute the IoU between box with proposal
	Arg:
		box: [x1,y1,x2,y2]
		proposal: N*4 matrix, each line is [p_x1,p_y1,p_x2,p_y2]
	output:
		IoU: N*1 matrix, every IoU[i] means the IoU between
			 box with proposal[i,:]
	"""
    len_proposal = np.shape(proposal)[0]
    IoU = np.empty([len_proposal, 1])
    for i in range(len_proposal):
        xA = max(box[0], proposal[i][0])
        yA = max(box[1], proposal[i][1])
        xB = min(box[2], proposal[i][2])
        yB = min(box[3], proposal[i][3])

        if xB < xA or yB < yA:
            IoU[i, 0] = 0
        else:
            area_I = (xB - xA) * (yB - yA)
            area1 = (box[2] - box[0]) * (box[3] - box[1])
            area2 = (proposal[i, 2] - proposal[i, 0]) * (proposal[i, 3] - proposal[i, 1])
            IoU[i, 0] = area_I / float(area1 + area2 - area_I)
    return IoU


def compute_iou_each(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    if xB < xA or yB < yA:
        IoU = 0
    else:
        area_I = (xB - xA) * (yB - yA)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        IoU = area_I / float(area1 + area2 - area_I)
    return IoU


def im_preprocess(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    im_orig = image.astype(np.float32, copy=True)
    im_orig -= np.array([[[102.9801, 115.9465, 122.7717]]])

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    target_size = cfg.IM_SIZE
    max_size = cfg.IM_MAX_SIZE
    im_scale = float(target_size) / float(im_size_min)

    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)
    im_shape_new = np.shape(im)
    im_use = np.zeros([1, im_shape_new[0], im_shape_new[1], im_shape_new[2]])
    im_use[0, :, :, :] = im
    sh = im.shape
    h = sh[0]
    w = sh[1]

    return im_use, im_scale, w, h


def generate_phrase_min_box(sbox, obox):
    N_box = len(sbox)
    phrase = np.zeros([N_box, 4])
    for i in range(N_box):
        phrase[i, 0] = max(sbox[i, 0], obox[i, 0])
        phrase[i, 1] = max(sbox[i, 1], obox[i, 1])
        phrase[i, 2] = min(sbox[i, 2], obox[i, 2])
        phrase[i, 3] = min(sbox[i, 3], obox[i, 3])
        if phrase[i, 2] < phrase[i, 0] or phrase[i, 3] < phrase[i, 1]:
            continue
        else:
            phrase[i] = phrase[i] * 0
    return phrase


def get_blob_pred(roidb_use, im_scale, N_each_batch):
    blob = {}
    sub_box = roidb_use['sub_box_gt'] * im_scale
    obj_box = roidb_use['obj_box_gt'] * im_scale
    rela = np.int32(roidb_use['rela_gt'])
    size = len(roidb_use['rela_gt'])

    index_use = np.array(range(size))
    size = len(index_use)

    sub_box_use = sub_box[index_use, :]
    obj_box_use = obj_box[index_use, :]
    rela_use = rela[index_use]

    blob['sub_box'] = sub_box_use
    blob['obj_box'] = obj_box_use

    sp1 = np.reshape(sub_box_use[:, 0], [size, 1])
    sp2 = np.reshape(sub_box_use[:, 1], [size, 1])
    sp3 = np.reshape(sub_box_use[:, 2], [size, 1])
    sp4 = np.reshape(sub_box_use[:, 3], [size, 1])

    op1 = np.reshape(obj_box_use[:, 0], [size, 1])
    op2 = np.reshape(obj_box_use[:, 1], [size, 1])
    op3 = np.reshape(obj_box_use[:, 2], [size, 1])
    op4 = np.reshape(obj_box_use[:, 3], [size, 1])

    blob['rela'] = rela_use

    blob['sub_gt'] = np.int32(roidb_use['sub_gt'][index_use])
    blob['obj_gt'] = np.int32(roidb_use['obj_gt'][index_use])

    v1 = np.reshape(np.min(np.hstack([op1, sp1]), 1), [size, 1])
    v2 = np.reshape(np.min(np.hstack([op2, sp2]), 1), [size, 1])
    v3 = np.reshape(np.max(np.hstack([op3, sp3]), 1), [size, 1])
    v4 = np.reshape(np.max(np.hstack([op4, sp4]), 1), [size, 1])
    vis_box_max = np.hstack([v1, v2, v3, v4])
    blob['vis_box'] = vis_box_max

    sv1 = (sp1 - v1) / (v3 - v1)
    sv2 = (sp2 - v2) / (v4 - v2)
    sv3 = (sp3 - v1) / (v3 - v1)
    sv4 = (sp4 - v2) / (v4 - v2)

    ov1 = (op1 - v1) / (v3 - v1)
    ov2 = (op2 - v2) / (v4 - v2)
    ov3 = (op3 - v1) / (v3 - v1)
    ov4 = (op4 - v2) / (v4 - v2)

    blob['sp_info'] = np.hstack([sv1, sv2, sv3, sv4, ov1, ov2, ov3, ov4])

    return blob


def get_blob_rela(roidb_use, im_scale, N_each_batch, batch_id, N_t):
    blob = {}
    sub_box = np.array(roidb_use['sub_box_dete']) * im_scale
    obj_box = np.array(roidb_use['obj_box_dete']) * im_scale
    rela = np.int32(roidb_use['rela_dete'])

    r = np.array(roidb_use['right_index'][batch_id * N_each_batch: (batch_id + 1) * N_each_batch], dtype=np.int64)

    shuffle(roidb_use['wrong_index'])
    r2 = np.array(roidb_use['wrong_index'][:np.int(len(r) * N_t)], dtype=np.int64)

    if len(r2) == 0 and len(r) != 0:
        index_use = r
    elif len(r2) != 0 and len(r) == 0:
        index_use = r2
    elif len(r2) == 0 and len(r) == 0:
        return blob
    else:
        index_use = np.concatenate([r, r2])

    size = len(index_use)

    sub_box_use = sub_box[index_use]
    obj_box_use = obj_box[index_use]
    rela_use = rela[index_use]

    blob['sub_box'] = sub_box_use
    blob['obj_box'] = obj_box_use

    blob['rela'] = rela_use
    blob['sub_gt'] = np.int32(np.array(roidb_use['sub_dete'])[index_use]) - 1
    blob['obj_gt'] = np.int32(np.array(roidb_use['obj_dete'])[index_use]) - 1

    sp1 = np.reshape(sub_box_use[:, 0], [size, 1])
    sp2 = np.reshape(sub_box_use[:, 1], [size, 1])
    sp3 = np.reshape(sub_box_use[:, 2], [size, 1])
    sp4 = np.reshape(sub_box_use[:, 3], [size, 1])

    op1 = np.reshape(obj_box_use[:, 0], [size, 1])
    op2 = np.reshape(obj_box_use[:, 1], [size, 1])
    op3 = np.reshape(obj_box_use[:, 2], [size, 1])
    op4 = np.reshape(obj_box_use[:, 3], [size, 1])

    v1 = np.reshape(np.min(np.hstack([op1, sp1]), 1), [size, 1])
    v2 = np.reshape(np.min(np.hstack([op2, sp2]), 1), [size, 1])
    v3 = np.reshape(np.max(np.hstack([op3, sp3]), 1), [size, 1])
    v4 = np.reshape(np.max(np.hstack([op4, sp4]), 1), [size, 1])
    vis_box = np.hstack([v1, v2, v3, v4])
    blob['vis_box'] = vis_box

    sv1 = (sp1 - v1) / (v3 - v1)
    sv2 = (sp2 - v2) / (v4 - v2)
    sv3 = (sp3 - v1) / (v3 - v1)
    sv4 = (sp4 - v2) / (v4 - v2)

    ov1 = (op1 - v1) / (v3 - v1)
    ov2 = (op2 - v2) / (v4 - v2)
    ov3 = (op3 - v1) / (v3 - v1)
    ov4 = (op4 - v2) / (v4 - v2)

    blob['sp_info'] = np.hstack([sv1, sv2, sv3, sv4, ov1, ov2, ov3, ov4])

    return blob


def get_blob_rela_t(roidb_use, im_scale, N_each_batch, batch_id):
    blob = {}
    sub_box = np.array(roidb_use['sub_box_dete']) * im_scale
    obj_box = np.array(roidb_use['obj_box_dete']) * im_scale
    index = roidb_use['index_rela']

    index_use = index[batch_id * N_each_batch: (batch_id + 1) * N_each_batch]
    size = len(index_use)
    sub_box_use = sub_box[index_use]
    obj_box_use = obj_box[index_use]

    blob['sub_box'] = sub_box_use
    blob['obj_box'] = obj_box_use

    blob['sub_gt'] = np.int32(np.array(roidb_use['sub_dete'])[index_use]) - 1
    blob['obj_gt'] = np.int32(np.array(roidb_use['obj_dete'])[index_use]) - 1

    sp1 = np.reshape(sub_box_use[:, 0], [size, 1])
    sp2 = np.reshape(sub_box_use[:, 1], [size, 1])
    sp3 = np.reshape(sub_box_use[:, 2], [size, 1])
    sp4 = np.reshape(sub_box_use[:, 3], [size, 1])

    op1 = np.reshape(obj_box_use[:, 0], [size, 1])
    op2 = np.reshape(obj_box_use[:, 1], [size, 1])
    op3 = np.reshape(obj_box_use[:, 2], [size, 1])
    op4 = np.reshape(obj_box_use[:, 3], [size, 1])

    v1 = np.reshape(np.min(np.hstack([op1, sp1]), 1), [size, 1])
    v2 = np.reshape(np.min(np.hstack([op2, sp2]), 1), [size, 1])
    v3 = np.reshape(np.max(np.hstack([op3, sp3]), 1), [size, 1])
    v4 = np.reshape(np.max(np.hstack([op4, sp4]), 1), [size, 1])
    vis_box = np.hstack([v1, v2, v3, v4])
    blob['vis_box'] = vis_box

    sv1 = (sp1 - v1) / (v3 - v1)
    sv2 = (sp2 - v2) / (v4 - v2)
    sv3 = (sp3 - v1) / (v3 - v1)
    sv4 = (sp4 - v2) / (v4 - v2)

    ov1 = (op1 - v1) / (v3 - v1)
    ov2 = (op2 - v2) / (v4 - v2)
    ov3 = (op3 - v1) / (v3 - v1)
    ov4 = (op4 - v2) / (v4 - v2)

    blob['sp_info'] = np.hstack([sv1, sv2, sv3, sv4, ov1, ov2, ov3, ov4])

    return blob


def pred_recall(test_roidb, pred_roidb, N_recall):
    N_right = 0.0
    N_total = 0.0
    N_data = len(test_roidb)
    for i in range(N_data):
        gt_rela = test_roidb[i]['rela_gt']
        if len(gt_rela) == 0:
            continue
        pred_rela = pred_roidb[i]['pred_rela']
        pred_rela_score = pred_roidb[i]['pred_rela_score']
        N_rela = len(gt_rela)
        N_total = N_total + N_rela
        if N_rela <= N_recall:
            N_right = N_right + np.sum(np.float32(gt_rela == pred_rela))
        else:
            sort_score = np.sort(pred_rela_score)[::-1]
            thresh = sort_score[N_recall]
            for j in range(N_rela):
                if pred_rela_score[j] >= thresh:
                    N_right = N_right + np.float32(gt_rela[j] == pred_rela[j])

    acc = N_right / N_total
    return acc


def rela_recall(test_roidb, pred_roidb, N_recall):
    N_right = 0.0
    N_total = 0.0
    N_data = len(test_roidb)
    # print(N_data)
    num_right = np.zeros([N_data, ])
    for i in range(N_data):
        rela_gt = test_roidb[i]['rela_gt']
        if len(rela_gt) == 0:
            continue
        sub_gt = test_roidb[i]['sub_gt']
        obj_gt = test_roidb[i]['obj_gt']
        sub_box_gt = test_roidb[i]['sub_box_gt']
        obj_box_gt = test_roidb[i]['obj_box_gt']

        pred_rela = pred_roidb[i]['pred_rela']
        pred_rela_score = pred_roidb[i]['pred_rela_score']
        sub_dete = pred_roidb[i]['sub_dete']
        obj_dete = pred_roidb[i]['obj_dete']
        sub_box_dete = pred_roidb[i]['sub_box_dete']
        obj_box_dete = pred_roidb[i]['obj_box_dete']

        N_rela = len(rela_gt)
        N_total = N_total + N_rela

        N_pred = len(pred_rela)
        sort_score = -np.sort(-np.reshape(pred_rela_score, [1, -1]))
        if N_recall >= N_pred:
            thresh = -1
        else:
            thresh = sort_score[0][N_recall]
        detected_gt = np.zeros([N_rela, ])
        for j in range(N_pred):
            if pred_rela_score[j] <= thresh:
                continue

            for k in range(N_rela):
                if detected_gt[k] == 1:
                    continue
                if (sub_gt[k] == sub_dete[j]) and (obj_gt[k] == obj_dete[j]) and (rela_gt[k] == pred_rela[j]):
                    s_iou = compute_iou_each(sub_box_dete[j], sub_box_gt[k])
                    o_iou = compute_iou_each(obj_box_dete[j], obj_box_gt[k])
                    if (s_iou >= 0.5) and (o_iou >= 0.5):
                        detected_gt[k] = 1
                        N_right = N_right + 1
                        num_right[i] = num_right[i] + 1

                # print(i,j, thresh, N_recall)
                # print(thresh, pred_rela_score[j])

    acc = N_right / N_total
    print(N_right)
    print(N_total)
    return acc, num_right


def phrase_recall(test_roidb, pred_roidb, N_recall):
    N_right = 0.0
    N_total = 0.0
    N_data = len(test_roidb)
    print(N_data)
    num_right = np.zeros([N_data, ])
    for i in range(N_data):
        rela_gt = test_roidb[i]['rela_gt']
        if len(rela_gt) == 0:
            continue

        N_rela = len(rela_gt)
        sub_gt = test_roidb[i]['sub_gt']
        obj_gt = test_roidb[i]['obj_gt']
        sub_box_gt = test_roidb[i]['sub_box_gt']
        obj_box_gt = test_roidb[i]['obj_box_gt']
        phrase_gt = generate_phrase_box(sub_box_gt, obj_box_gt)

        pred_rela = pred_roidb[i]['pred_rela']
        pred_rela_score = pred_roidb[i]['pred_rela_score']
        sub_dete = pred_roidb[i]['sub_dete']
        obj_dete = pred_roidb[i]['obj_dete']
        sub_box_dete = pred_roidb[i]['sub_box_dete']
        obj_box_dete = pred_roidb[i]['obj_box_dete']
        phrase_dete = generate_phrase_box(sub_box_dete, obj_box_dete)
        N_pred = len(pred_rela)

        N_total = N_total + N_rela

        sort_score = -np.sort(-np.reshape(pred_rela_score, [1, -1]))
        if N_recall >= N_pred:
            thresh = -1
        else:
            thresh = sort_score[0][N_recall]

        detected_gt = np.zeros([N_rela, ])
        for j in range(N_pred):
            if pred_rela_score[j] <= thresh:
                continue

            for k in range(N_rela):
                if detected_gt[k] == 1:
                    continue
                if (sub_gt[k] == sub_dete[j]) and (obj_gt[k] == obj_dete[j]) and (rela_gt[k] == pred_rela[j]):
                    iou = compute_iou_each(phrase_dete[j], phrase_gt[k])
                    if iou >= 0.5:
                        detected_gt[k] = 1
                        N_right = N_right + 1
                        num_right[i] = num_right[i] + 1

    acc = N_right / N_total
    print(N_right)
    print(N_total)
    return acc, num_right


def rela_recall_r(test_roidb, pred_roidb, N_recall):
    N_right = 0.0
    N_total = 0.0
    N_data = len(test_roidb)
    num_right = np.zeros([N_data, ])
    for i in range(N_data):
        rela_gt = test_roidb[i]['rela_gt']
        if len(rela_gt) == 0:
            continue
        sub_gt = test_roidb[i]['sub_gt']
        obj_gt = test_roidb[i]['obj_gt']
        sub_box_gt = test_roidb[i]['sub_box_gt']
        obj_box_gt = test_roidb[i]['obj_box_gt']

        pred_rela = pred_roidb[i]['pred_rela']
        pred_rela_score = pred_roidb[i]['pred_rela_score']
        sub_dete = pred_roidb[i]['sub_dete']
        obj_dete = pred_roidb[i]['obj_dete']
        sub_box_dete = pred_roidb[i]['sub_box_dete']
        obj_box_dete = pred_roidb[i]['obj_box_dete']

        N_rela = len(rela_gt)
        N_total = N_total + N_rela

        N_pred = len(pred_rela)
        sort_score = -np.sort(-np.reshape(pred_rela_score, [1, -1]))

        if N_recall >= N_pred:
            thresh = -1
        else:
            thresh = sort_score[0][N_recall]

        detected_gt = np.zeros([N_rela, ])
        detected_gt_1 = np.zeros([N_rela, ])
        for j in range(N_pred):
            if pred_rela_score[j] <= thresh:
                continue
            maxk = 0
            positionk = -1
            for k in range(N_rela):
                if detected_gt[k] == 1:
                    continue
                if (sub_gt[k] == sub_dete[j]) and (obj_gt[k] == obj_dete[j]) and (rela_gt[k] == pred_rela[j]):
                    s_iou = compute_iou_each(sub_box_dete[j], sub_box_gt[k])
                    o_iou = compute_iou_each(obj_box_dete[j], obj_box_gt[k])
                    iou = np.max([s_iou, o_iou])
                    if (s_iou >= 0.5) and (o_iou >= 0.5) and iou > maxk:
                        maxk = iou
                        positionk = k
            if positionk > -1:
                detected_gt[positionk] = 1
        N_right = N_right + np.sum(detected_gt)

    acc = N_right / N_total
    return N_right, N_total


def phrase_recall_r(test_roidb, pred_roidb, N_recall):
    N_right = 0.0
    N_total = 0.0
    N_data = len(test_roidb)
    num_right = np.zeros([N_data, ])
    for i in range(N_data):
        rela_gt = test_roidb[i]['rela_gt']
        if len(rela_gt) == 0:
            continue

        N_rela = len(rela_gt)
        sub_gt = test_roidb[i]['sub_gt']
        obj_gt = test_roidb[i]['obj_gt']
        sub_box_gt = test_roidb[i]['sub_box_gt']
        obj_box_gt = test_roidb[i]['obj_box_gt']
        phrase_gt = generate_phrase_box(sub_box_gt, obj_box_gt)

        pred_rela = pred_roidb[i]['pred_rela']
        pred_rela_score = pred_roidb[i]['pred_rela_score']
        sub_dete = pred_roidb[i]['sub_dete']
        obj_dete = pred_roidb[i]['obj_dete']
        sub_box_dete = pred_roidb[i]['sub_box_dete']
        obj_box_dete = pred_roidb[i]['obj_box_dete']
        phrase_dete = generate_phrase_box(sub_box_dete, obj_box_dete)
        N_pred = len(pred_rela)

        N_total = N_total + N_rela

        sort_score = -np.sort(-np.reshape(pred_rela_score, [1, -1]))
        if N_recall >= N_pred:
            thresh = -1
        else:
            thresh = sort_score[0][N_recall]
        detected_gt = np.zeros([N_rela, ])
        for j in range(N_pred):
            if pred_rela_score[j] <= thresh:
                continue
            maxk = 0
            positionk = -1
            for k in range(N_rela):
                if detected_gt[k] == 1:
                    continue
                if (sub_gt[k] == sub_dete[j]) and (obj_gt[k] == obj_dete[j]) and (rela_gt[k] == pred_rela[j]):
                    iou = compute_iou_each(phrase_dete[j], phrase_gt[k])
                    if iou >= 0.5 and iou > maxk:
                        maxk = iou
                        positionk = k
            if positionk > -1:
                detected_gt[positionk] = 1
        N_right = N_right + np.sum(detected_gt)

    acc = N_right / N_total

    return N_right, N_total


def print_pred_res(test_roidb, pred_roidb, res_name, N_rela):
    N_pred_rela = np.zeros([N_rela, ])
    N_right_rela = np.zeros([N_rela, ])
    N_gt_rela = np.zeros([N_rela, ])
    N_acc = np.zeros([N_rela, ])
    N_pred_other = np.zeros([N_rela, N_rela])

    N_pred = len(pred_roidb)
    for i in range(N_pred):
        pred_rela = np.int32(pred_roidb[i]['pred_rela'])
        gt_rela = np.int32(test_roidb[i]['rela_gt'])
        for k in range(len(gt_rela)):
            N_pred_rela[pred_rela[k]] += 1
            N_gt_rela[gt_rela[k]] += 1
            N_pred_other[gt_rela[k]][pred_rela[k]] += 1
            if pred_rela[k] == gt_rela[k]:
                N_right_rela[gt_rela[k]] += 1

    for k in range(N_rela):
        if N_gt_rela[k] == 0:
            N_acc[k] = -1
            continue
        N_acc[k] = (N_right_rela[k] + 0.0) / N_gt_rela[k]

    text_file = open(res_name, "aw")
    for k in range(N_rela):
        text_file.write('k: {0}, acc: {1}\n'.format(k, N_acc[k]))
    for k in range(N_rela):
        text_file.write('k: {0}, num of pred: {1}\n'.format(k, N_pred_rela[k]))
    for k in range(N_rela):
        text_file.write('k: {0}, others: {1}\n'.format(k, np.where(N_pred_other[k] > 0)[0]))
    text_file.close()


def generate_phrase_box(sbox, obox):
    N_box = len(sbox)
    phrase = np.zeros([N_box, 4])
    for i in range(N_box):
        phrase[i, 0] = min(sbox[i, 0], obox[i, 0])
        phrase[i, 1] = min(sbox[i, 1], obox[i, 1])
        phrase[i, 2] = max(sbox[i, 2], obox[i, 2])
        phrase[i, 3] = max(sbox[i, 3], obox[i, 3])
    return phrase


def extract_detected_box(dete_box):
    N_cls = len(dete_box['pred_boxes'])
    pred_box = dete_box['pred_boxes']
    detected_cls = np.zeros([1000, ])
    detected_box = np.zeros([1000, 4])
    detected_score = np.zeros([1000, ])
    t = 0
    for i in range(N_cls):
        l = len(pred_box[i])
        if l == 0:
            continue
        box_temp = pred_box[i]
        detected_box[t:t + l] = box_temp[:, 0:4]
        detected_cls[t:t + l] = i
        detected_score[t:t + l] = box_temp[:, 4]
        t = t + l
    detected_box = detected_box[0:t]
    detected_cls = detected_cls[0:t]
    detected_score = detected_score[0:t]
    return detected_box, detected_cls, detected_score


def extract_roidb_box(roidb):
    sbox = roidb['sub_box_gt']
    obox = roidb['obj_box_gt']
    sub_gt = np.reshape(roidb['sub_gt'], [len(roidb['sub_box_gt']), 1])
    obj_gt = np.reshape(roidb['obj_gt'], [len(roidb['obj_box_gt']), 1])
    sub_box_gt = np.concatenate((roidb['sub_box_gt'], sub_gt), axis=1)
    obj_box_gt = np.concatenate((roidb['obj_box_gt'], obj_gt), axis=1)
    roidb_box = np.concatenate((sub_box_gt, obj_box_gt), axis=0)

    roidb_cls = np.concatenate((roidb['sub_gt'], roidb['obj_gt']), axis=0)
    roidb_rela = roidb['rela_gt']
    N_rela = len(roidb_rela)
    unique_boxes, unique_inds = np.unique(roidb_box, axis=0, return_index=True)
    unique_boxes1 = np.hsplit(unique_boxes, [4, 1])
    unique_boxes = unique_boxes1[0]
    unique_cls = roidb_cls[unique_inds]
    rela_box_index = np.zeros([N_rela, 2])
    for i in range(N_rela):
        rela_box_index[i, 0] = np.where(unique_boxes == sbox[i])[0][0]
        rela_box_index[i, 1] = np.where(unique_boxes == obox[i])[0][0]
    return unique_cls, unique_boxes, roidb_rela, rela_box_index


def generate_au_box(unique_boxes, detected_box, iou_l):
    N_unique = len(unique_boxes)
    au_box = []
    for i in range(N_unique):
        box_temp = unique_boxes[i]
        iou = compute_iou(box_temp, detected_box)
        index_temp = np.where(iou > iou_l)[0]
        box_use = detected_box[index_temp]
        box_use = np.vstack((box_use, box_temp))
        au_box.append(box_use)
    return au_box


def generate_rela_info(au_box, index, N_each_pair):
    s_id = np.int32(index[0])
    o_id = np.int32(index[1])
    sbox = au_box[s_id]
    obox = au_box[o_id]
    N_s = len(sbox)
    N_o = len(obox)
    sa = np.random.randint(0, N_s, [N_each_pair, ])
    oa = np.random.randint(0, N_o, [N_each_pair, ])
    sbox_use = sbox[sa]
    obox_use = obox[oa]
    return sbox_use, obox_use


def generate_phrase_box1(sbox, obox):
    N_box = 1
    phrase = np.zeros([1, 4])

    phrase[0, 0] = min(sbox[0], obox[0])
    phrase[0, 1] = min(sbox[1], obox[1])
    phrase[0, 2] = max(sbox[2], obox[2])
    phrase[0, 3] = max(sbox[3], obox[3])
    return phrase


def generate_train_rela_roidb(roidb, dete_box, iou_l, N_each_batch, N_each_pair):
    unique_cls, unique_boxes, roidb_rela, rela_box_index = extract_roidb_box(roidb)
    detected_box, detected_cls, detected_score = extract_detected_box(dete_box)
    detected_cls = np.concatenate((detected_cls, unique_cls + 1), axis=0)
    detected_box = np.concatenate((detected_box, unique_boxes), axis=0)
    detected_score = np.concatenate((detected_score, unique_cls * 0 + 1), axis=0)
    N_dete = len(detected_box)
    sub_box_dete = np.zeros([N_dete * (N_dete - 1), 4])
    obj_box_dete = np.zeros([N_dete * (N_dete - 1), 4])
    sub_dete = np.zeros([N_dete * (N_dete - 1), ])
    obj_dete = np.zeros([N_dete * (N_dete - 1), ])
    sub_score = np.zeros([N_dete * (N_dete - 1), ])
    obj_score = np.zeros([N_dete * (N_dete - 1), ])
    t = 0
    for i in range(N_dete):
        for j in range(N_dete):
            if j == i:
                continue
            sub_box_dete[t] = detected_box[i]
            obj_box_dete[t] = detected_box[j]
            sub_dete[t] = detected_cls[i]
            obj_dete[t] = detected_cls[j]
            sub_score[t] = detected_score[i]
            obj_score[t] = detected_score[j]
            t = t + 1
    sub_box_dete = sub_box_dete[0:t]
    obj_box_dete = obj_box_dete[0:t]

    sub_dete = sub_dete[0:t]
    obj_dete = obj_dete[0:t]

    sub_score = sub_score[0:t]
    obj_score = obj_score[0:t]

    sub_box = []
    obj_box = []
    sub_dete1 = []
    obj_dete1 = []
    sub_score1 = []
    obj_score1 = []
    rela_dete = []
    is_pair = []
    right_index = []
    wrong_index = []

    for i in range(len(sub_box_dete)):
        # pdb.set_trace()
        s_IOU = np.reshape(compute_iou(sub_box_dete[i], roidb['sub_box_gt']), np.shape(roidb['rela_gt']))
        o_IOU = np.reshape(compute_iou(obj_box_dete[i], roidb['obj_box_gt']), np.shape(roidb['rela_gt']))
        f_IOU1 = (s_IOU > 0.5) * (o_IOU > 0.5) + 0.

        s = sub_dete[i] - roidb['sub_gt'] - 1  #######################################
        o = obj_dete[i] - roidb['obj_gt'] - 1  #################################

        s_gt = np.reshape((s == 0) + 0., np.shape(roidb['rela_gt']))
        o_gt = np.reshape((o == 0) + 0., np.shape(roidb['rela_gt']))
        f_IOU = f_IOU1 * s_gt * o_gt

        if np.sum(f_IOU) > 0:
            rel = roidb['rela_gt'][f_IOU == 1]
            sub_box.append(sub_box_dete[i])
            obj_box.append(obj_box_dete[i])
            sub_dete1.append(sub_dete[i])
            obj_dete1.append(obj_dete[i])
            sub_score1.append(sub_score[i])
            obj_score1.append(obj_score[i])
            is_pair.append(1)
            right_index.append(len(sub_box) - 1)
            a = np.zeros([71])
            for k in range(len(rel)):
                a[np.int(rel[k])] = 1
            rela_dete.append(a)
        else:
            sub_box.append(sub_box_dete[i])
            obj_box.append(obj_box_dete[i])
            sub_dete1.append(sub_dete[i])
            obj_dete1.append(obj_dete[i])
            sub_score1.append(sub_score[i])
            obj_score1.append(obj_score[i])
            a = np.zeros([71])
            a[70] = 1
            rela_dete.append(a)
            is_pair.append(0)
            wrong_index.append(len(sub_box) - 1)

    sub_box_dete = np.array(sub_box, dtype=np.float32)
    obj_box_dete = np.array(obj_box, dtype=np.float32)
    sub_dete2 = np.array(sub_dete1, dtype=np.float32)
    obj_dete2 = np.array(obj_dete1, dtype=np.float32)

    index_rela = generate_batch(len(rela_dete), N_each_batch)

    roidb_temp = {'image': roidb['image'], 'rela_gt': roidb_rela,
                  'sub_box_dete': sub_box_dete, 'obj_box_dete': obj_box_dete, 'sub_dete': sub_dete2,
                  'obj_dete': obj_dete2, 'rela_dete': rela_dete, 'is_pair': is_pair, 'right_index': right_index,
                  'wrong_index': wrong_index, 'sub_score': np.array(sub_score1), 'obj_score': np.array(obj_score1)}
    # print(len(roidb_temp['rela_dete']), len(roidb_temp['sub_dete']),len(roidb_temp['obj_dete']))
    return roidb_temp, N_dete


def generate_test_rela_roidb(roidb, dete_box, N_each_batch):
    detected_box, detected_cls, detected_score = extract_detected_box(dete_box)
    N_dete = len(detected_box)
    sub_box_dete = np.zeros([N_dete * (N_dete - 1), 4])
    obj_box_dete = np.zeros([N_dete * (N_dete - 1), 4])
    sub_dete = np.zeros([N_dete * (N_dete - 1), ])
    obj_dete = np.zeros([N_dete * (N_dete - 1), ])
    sub_score = np.zeros([N_dete * (N_dete - 1), ])
    obj_score = np.zeros([N_dete * (N_dete - 1), ])

    t = 0
    for i in range(N_dete):
        for j in range(N_dete):
            if j == i:
                continue
            sub_box_dete[t] = detected_box[i]
            obj_box_dete[t] = detected_box[j]
            sub_dete[t] = detected_cls[i]
            obj_dete[t] = detected_cls[j]
            sub_score[t] = detected_score[i]
            obj_score[t] = detected_score[j]
            t = t + 1
    sub_box_dete = sub_box_dete[0:t]
    obj_box_dete = obj_box_dete[0:t]
    sub_dete = sub_dete[0:t]
    obj_dete = obj_dete[0:t]
    sub_score = sub_score[0:t]
    obj_score = obj_score[0:t]
    index_rela = generate_batch(len(sub_box_dete), N_each_batch)
    rela_dete = np.zeros([len(sub_box_dete), ])

    roidb_temp = {'image': roidb['image'], 'rela_gt': roidb['rela_gt'], 'sub_gt': roidb['sub_gt'],
                  'obj_gt': roidb['obj_gt'],
                  'sub_box_dete': sub_box_dete, 'obj_box_dete': obj_box_dete, 'sub_dete': sub_dete,
                  'obj_dete': obj_dete,
                  'detected_box': detected_box, 'detected_cls': detected_cls, 'sub_box_gt': roidb['sub_box_gt'],
                  'obj_box_gt': roidb['obj_box_gt'], 'sub_score': sub_score, 'obj_score': obj_score,
                  'index_rela': index_rela,
                  'rela_dete': rela_dete, 'detected_score': detected_score}

    return roidb_temp

