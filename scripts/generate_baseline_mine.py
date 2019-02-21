import base64
import numpy as np
import cv2
import csv
import json
import os
import caffe
import sys
from scipy.ndimage import zoom
import random

random.seed(1)
import gc
import logging

csv.field_size_limit(sys.maxsize)
from skimage import io
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect, _get_blobs
from fast_rcnn.nms_wrapper import nms
import urllib

FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features', 'attrs', 'objects']
KARPATHY_SPLITS = './data/coco_splits/karpathy_%s_images.txt'  # train,val,test

PROTOTXT = 'baseline/test.prototxt'
WEIGHTS = 'baseline/resnet101_faster_rcnn_final.caffemodel'

IMAGE_DIR = 'data/images/'


def load_karpathy_splits(dataset='train'):
    imgIds = set()
    with open(KARPATHY_SPLITS % dataset) as data_file:
        for line in data_file:
            imgIds.add(int(line.split()[-1]))
    return imgIds


def load_image_ids(image_folder):
    ''' Map image ids to file paths. '''
    id_to_path = {}
    filenames = os.listdir(image_folder)
    for file in filenames:
        name = file.split('.')[0]
        id_to_path[name] = image_folder + file
    print 'Loaded %d image ids' % len(id_to_path)
    return id_to_path


caffe_root = ''  # this file should be run from REPO_ROOT/scripts

# Reduce the max number of region proposals, so that the bottom-up and top-down models can
# both fit on a 12GB gpu -> this may cause some demo captions to differ slightly from the
# generated outputs of ./experiments/caption_lstm/train.sh

cfg['TEST']['RPN_POST_NMS_TOP_N'] = 150  # Previously 300 for evaluations reported in the paper

rcnn_weights = caffe_root + 'demo/resnet101_faster_rcnn_final.caffemodel'

caption_weights = caffe_root + 'demo/lstm_iter_60000.caffemodel.h5'  # cross-entropy trained
caption_weights_scst = caffe_root + 'demo/lstm_scst_iter_1000.caffemodel.h5'  # self-critical trained

if os.path.isfile(rcnn_weights):
    print('Faster R-CNN weights found.')
else:
    print( 'Downloading Faster R-CNN weights...')
    url = "https://storage.googleapis.com/bottom-up-attention/resnet101_faster_rcnn_final.caffemodel"
    urllib.urlretrieve(url, rcnn_weights)

if os.path.isfile(caption_weights):
    print('Caption weights found.')
else:
    print('Downloading Caption weights...')
    url = "https://storage.googleapis.com/bottom-up-attention/%s" % caption_weights.split('/')[-1]
    urllib.urlretrieve(url, caption_weights)

if os.path.isfile(caption_weights_scst):
    print('Caption weights found.')
else:
    print('Downloading Caption weights...')
    url = "https://storage.googleapis.com/bottom-up-attention/%s" % caption_weights_scst.split('/')[-1]
    urllib.urlretrieve(url, caption_weights_scst)

MIN_BOXES = 10
MAX_BOXES = 100
# Code for getting features from Faster R-CNN
net = caffe.Net(PROTOTXT, caffe.TEST, weights=WEIGHTS)


def get_detections_from_im(image_id, image_path):
    global net
    im = cv2.imread(image_path)
    conf_thresh = 0.2
    # shape (rows, columns, channels)
    scores, _, _ = im_detect(net, im)

    # Keep the original boxes, don't worry about the regresssion bbox outputs
    rois = net.blobs['rois'].data.copy()
    # unscale back to raw image space
    _, im_scales = _get_blobs(im, None)
    cls_boxes = rois[:, 1:5] / im_scales[0]
    cls_prob = net.blobs['cls_prob'].data
    attr_prob = net.blobs['attr_prob'].data
    pool5 = net.blobs['pool5_flat'].data
    pool5_unflat = net.blobs['pool5'].data

    # Keep only the best detections
    max_conf = np.zeros((rois.shape[0]))
    for cls_ind in range(1, cls_prob.shape[1]):
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = np.array(nms(dets, cfg.TEST.NMS))
        max_conf[keep] = np.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])

    keep_boxes = np.where(max_conf >= conf_thresh)[0]
    if len(keep_boxes) < MIN_BOXES:
        keep_boxes = np.argsort(max_conf)[::-1][:MIN_BOXES]
    elif len(keep_boxes) > MAX_BOXES:
        keep_boxes = np.argsort(max_conf)[::-1][:MAX_BOXES]
    objects = np.argmax(cls_prob[keep_boxes][:, 1:], axis=1)
    attrs = np.argmax(attr_prob[keep_boxes][:, 1:], axis=1)

    res = {
        'image_id': image_id,
        'image_w': np.size(im, 1),
        'image_h': np.size(im, 0),
        'num_boxes': len(keep_boxes),
        'boxes': base64.b64encode(cls_boxes[keep_boxes]),
        'features': base64.b64encode(pool5[keep_boxes]),
        'objects': base64.b64encode(objects),
        'attrs': base64.b64encode(attrs)
    }
    return res


def run(image_folder, outfile):
    tsv_files = outfile

    id_to_path = load_image_ids(image_folder)
    caffe.set_mode_gpu()
    caffe.set_device(0)

    out_file = tsv_files
    with open(out_file, 'wb') as resnet_tsv_out:
        print 'Writing to %s' % out_file
        resnet_writer = csv.DictWriter(resnet_tsv_out, delimiter='\t', fieldnames=FIELDNAMES)
        count = 0
        for image_id in id_to_path.keys():
            if image_id == '':
                continue

            count += 1
            resnet_baseline = get_detections_from_im(image_id, id_to_path[image_id])
            resnet_writer.writerow(resnet_baseline)
            #   if count % 1000 == 0:
            print        '%d / %d' % (count, len(id_to_path.keys()))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--image_folder", type=str, default='data/images/test2015/', help="Path to image folder")
    parser.add_argument("--out_file", type=str, default='data/7w.tsv',
                        help="Path to .tsv file which contains extracted image features")
    args = parser.parse_args()

    run(args.image_folder, args.out_file)