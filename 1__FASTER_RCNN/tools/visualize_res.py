#!/usr/bin/env python

import _init_paths
from fast_rcnn.test import test_net
from demo import vis_detections
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
from fast_rcnn.nms_wrapper import nms
import caffe
import time, os, sys
import cPickle
import matplotlib.pyplot as plt
import Image, ImageDraw
import cv2

if __name__ == '__main__':
    imdb_name = 'imagenet_2015_val'
    #test_name = 'imagenet_bvlc_fast_rcnn_stage1_iter_125000'
    test_name = 'vgg16_fast_rcnn_iter_800000'
    det_name = '../output/default/' + imdb_name + '/' + test_name + '/detections.pkl'
    cfg.TRAIN.PROPOSAL_METHOD = 'gt'
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    
    imdb = get_imdb(imdb_name)
    all_boxes = cPickle.load(open(det_name,'rb'))
    
    for i in xrange(len(imdb._image_index)) :
        im_path = imdb.image_path_at(i)
        im = cv2.imread( im_path )
        cls_set = []
        det_set = []
        
        for cls_ind in xrange(1,len(imdb.classes)):
            dets = all_boxes[cls_ind][i]
            keep = nms(dets, 0.3)
            dets = dets[keep, :]
            cls = imdb._class_name[cls_ind]
            cls_set.append(cls)
            det_set.append(dets)
        
        vis_detections(im, cls_set, det_set, 0.7)
        plt.show()
             
