#!/usr/bin/env python

import _init_paths
from fast_rcnn.test import test_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
import caffe
import time, os, sys
import cPickle
import matplotlib.pyplot as plt
import Image, ImageDraw
import cv2

if __name__ == '__main__':
    #imdb_name = 'imagenet_2015_train'
    #proposal_name = '../output/default/' + imdb_name + '/imagenet_bvlc_rpn_stage2_iter_1000000_proposals.pkl'
    imdb_name = 'voc_2007_trainval'
    proposal_name = '../output/default/' + imdb_name + '/vgg16_rpn_stage2_iter_80000_proposals.pkl'
    cfg.TRAIN.PROPOSAL_METHOD = 'gt'
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    imdb = get_imdb(imdb_name)
    rpn = cPickle.load(open(proposal_name,'rb'))

    for i in xrange(300,len(imdb._image_index)) :
        im_path = imdb.image_path_at(i)
        regions = rpn[i]
        import pdb
        pdb.set_trace()
        plt.ion()
        for j in range( 10 ) : #regions.shape[0] ) :
            im = cv2.imread( im_path )
            cv2.rectangle(im,(regions[j][0],regions[j][1]),(regions[j][2],regions[j][3]),(0,255,0),2)
            im = im[:, :, (2, 1, 0)]
            plt.imshow(im)
            plt.show()
            _ = raw_input("Press [enter] to continue.") # wait for input from the user
             
