# --------------------------------------------------------
# Fast R-CNN for ILSVRC DETECTION
# Copyright Lunit Inc.
# Written by PAENG
# --------------------------------------------------------

import datasets
import datasets.pascal_voc
import os
import datasets.imdb
import xml.dom.minidom as minidom
from xml.etree.ElementTree import Element
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import h5py
import PIL
class imagenet(datasets.imdb):
    def __init__(self, image_set, year):
        datasets.imdb.__init__(self, 'imagenet_' + year + '_' + image_set)
        self._year = year
        self._image_set = image_set
        self._devkit_path = os.path.join( self._get_default_path(), 'devkit' )
        self._data_path = os.path.join( self._get_default_path(), 'Data', 'DET', self._image_set )
        self._annot_path = os.path.join( self._get_default_path(), 'Annotations', 'DET', self._image_set )
        if image_set is 'test' :
            self._data_path = '' # no data files
            self._annot_path = '' # no gt files
        self._classes = ('__background__',) # always index 0
        synsets = sio.loadmat( os.path.join(self._devkit_path,'data','meta_det.mat') )
        synsets = synsets['synsets'].squeeze()
        for i in range(200) :
            self._classes += (str(synsets[i][1][0]),)
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.JPEG'
        self._image_index = self._load_image_set_index()
        self._wh = self._load_image_width_height()
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb

        # PASCAL specific config options
        self.config = {'cleanup'  : True,
                       'use_salt' : True,
                       'top_k'    : 2000,
                       'use_diff' : False,
                       'rpn_file' : None}
        
        assert os.path.exists(self._annot_path), \
                'IMAGENET annotation path does not exist: {}'.format(self._annot_path)
        assert os.path.exists(self._data_path), \
                'IMAGENET data path does not exist: {}'.format(self._data_path)
    
    def _get_default_path(self):
        """
        Return the default path where IMAGENET is expected to be installed.
        """
        return os.path.join(datasets.ROOT_DIR, 'data', 'ILSVRC' + self._year)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path
    
    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file. (val, test)
        """
        image_set_file = os.path.join(self._get_default_path(), 'ImageSets', 'DET', 
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.split()[0] for x in f.readlines()]
        return image_index
    
    def _load_image_width_height(self) :
        cache_file = os.path.join(self.cache_path, self.name + '_img_wh.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                img_wh = cPickle.load(fid)
            print '{} image wh loaded from {}'.format(self.name, cache_file)
            return img_wh
        img_wh = []
        for index in self._image_index :
            wh = self.load_image_wh(index)
            img_wh.append(wh)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(img_wh, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote image wh to {}'.format(cache_file)
        return img_wh

    def load_image_wh(self, index):
        """
        Load the width and height
        """
        filename = os.path.join(self._annot_path, index + '.xml')
        assert os.path.exists(filename), \
                'Path does not exist: {}'.format(image_set_file)
        def get_data_from_tag(node, tag):
            return node.getElementsByTagName(tag)[0].childNodes[0].data

        with open(filename) as f:
            data = minidom.parseString(f.read())

        size = data.getElementsByTagName('size')
        iw = float(get_data_from_tag(size[0], 'width'))
        ih = float(get_data_from_tag(size[0], 'height'))
        out = (iw, ih)
        return out

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb
        gt_roidb = [self._load_imagenet_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb
    
    def _load_imagenet_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the IMAGENET
        format.
        """
        filename = os.path.join(self._annot_path, index + '.xml')
        def get_data_from_tag(node, tag):
            return node.getElementsByTagName(tag)[0].childNodes[0].data

        with open(filename) as f:
            data = minidom.parseString(f.read())

        objs = data.getElementsByTagName('object')
        size = data.getElementsByTagName('size')
        iw = float(get_data_from_tag(size[0], 'width'))
        ih = float(get_data_from_tag(size[0], 'height'))
        num_objs = len(objs)
        
        assert num_objs != 0, \
               'No objects in ground truth information'

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            x1 = float(get_data_from_tag(obj, 'xmin')) 
            y1 = float(get_data_from_tag(obj, 'ymin')) 
            x2 = float(get_data_from_tag(obj, 'xmax')) 
            y2 = float(get_data_from_tag(obj, 'ymax')) 
            cls = self._class_to_ind[
                    str(get_data_from_tag(obj, "name")).lower().strip()]
            # to avoid wrong annotation
            if x1 < 0 :
                x1 = 0
            if y1 < 0 :
                y1 = 0
            # exception ( 1-based annotation --> 0-based )
            if x2 >= iw :
                x2 = iw-1
            if y2 >= ih :
                y2 = ih-1
            if x2 <= x1 or y2 <= y1 : # can't define bbox
                print index
                assert False, \
                       'Cannot define bounding box'
          
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
        
        overlaps = scipy.sparse.csr_matrix(overlaps)
        
        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False}
    
    def rpn_roidb(self):
        if self._image_set == 'train' :
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = datasets.imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        if self._image_set == 'train':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def _load_selective_search_roidb(self, gt_roidb):
        
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search.pkl')
        filename = os.path.abspath(os.path.join(self.cache_path, '..',
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
               'Selective search data not found at: {}'.format(filename)
        
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                box_list = cPickle.load(fid)
            print '{} selective search loaded from {}'.format(self.name, cache_file)
        else :
            ss_data = h5py.File(filename)
            box_list = []
            for i in range(ss_data['boxes'].shape[1]) :
                if i % 1000 == 0 :
                    print '[LOADING SS BOXES] %d th image...' % (i+1)
                tmp = [ ss_data[element[i]][:] for element in ss_data['boxes'] ]
                tmp = tmp[0].transpose()
                box_list.append(tmp[:, (1, 0, 3, 2)] - 1)
            
            with open(cache_file, 'wb') as fid:
                cPickle.dump(box_list, fid, cPickle.HIGHEST_PROTOCOL)
            print 'wrote selective search bboxes to  {}'.format(cache_file)

        return self.create_roidb_from_box_list(box_list, gt_roidb)


    def _write_imagenet_results_file(self, all_boxes, output_dir):
        # VOCdevkit/results/VOC2007/Main/comp4-44503_det_test_aeroplane.txt
        # path = os.path.join( output_dir, 'results', 'VOC' + self._year )
        filename = output_dir + 'det_' + self._image_set + '.txt'
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(self.image_index):
                for cls_ind, cls in enumerate(self.classes):
                    if cls == '__background__':
                        continue
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(im_ind+1, cls_ind, dets[k, -1],
                                   dets[k, 0], dets[k, 1] ,
                                   dets[k, 2], dets[k, 3] ))
        print 'Writing IMAGENET DET results file: {}'.format(filename)
    
    def evaluate_detections(self, all_boxes, output_dir):
        import pdb
        pdb.set_trace()
        self._write_imagenet_results_file(all_boxes, output_dir)
        #self._do_matlab_eval(comp_id, output_dir)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    d = datasets.imagenet_det('train','2015')
    res = d.roidb
    from IPython import embed; embed()
