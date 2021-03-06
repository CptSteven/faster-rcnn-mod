# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick

# Modified by Steven to suit an ImageNet-style dataset. Main 
# lie in __init__ where _classes are read from classfile and 
# in _load_annotations where object is check whether it within
# the classfile, plus some other minor changes.
# --------------------------------------------------------

import datasets
import datasets.pascal_voc
import os
import datasets.imdb
import xml
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess

class pascal_voc(datasets.imdb):
    def __init__(self, image_set, devkit_path, imdir=None, listfile_path=None, clsfile=None):
        datasets.imdb.__init__(self, image_set)
        #self._year = year
        #import pdb
        #pdb.set_trace()
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
                            else devkit_path
        self._data_path = imdir#os.path.join(self._devkit_path, 'Images')
        self._anno_path = os.path.join(self._devkit_path, "Annotations")        
        if listfile_path == None:
            if image_set == 'test':
                self._image_set_file = os.path.join(self._devkit_path, "testset.lst")
            else:
                self._image_set_file = os.path.join(self._devkit_path, "trainset.lst")
        else:
            self._image_set_file = listfile_path
        if clsfile == None:
            self._cls_path = os.path.join(self._devkit_path, "classes.lst")
        else:
            self._cls_path = clsfile
        with open(self._cls_path) as fcls:
            #class set, more robust
            cls_list = [line.strip() for line in fcls if line.strip()]
        self._classes = cls_list
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        #self._image_ext = '.jpg'	#to-be-modified, ext should be configurable
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb
        #currently use dev-mode
        self._image_dir = os.path.join(self._devkit_path, "Images")

        # PASCAL specific config options
        self.config = {'cleanup'  : False, #Mod True,
                       'use_salt' : True,
                       'top_k'    : 2000,
                       'use_diff' : False,
                       'rpn_file' : None}

        assert os.path.exists(self._devkit_path), \
                'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        #assert os.path.exists(self._data_path), \
        #        'Path does not exist: {}'.format(self._data_path)
        size_ratio_file_raw = os.path.join(self._devkit_path, 'size_ratio_raw.txt')
        if os.path.exists(size_ratio_file_raw):
            os.remove(size_ratio_file_raw)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        index in the form of class-dir/filename.xml. Read image path
        from the xml file to ensure correctness and generalization
        """
        xml_path = os.path.join(self._anno_path, index)
        try:
            dom = minidom.parse(xml_path)
        except (EnvironmentError, xml.parsers.expat.ExpatError) as err:
            print 'import error:{}'.format(err)
            exit(1)
        filepath = dom.getElementsByTagName("filepath")[0].childNodes[0].data

        if filepath.strip()[0] == '/':
            image_path = filepath
        else:
            image_path = os.path.join(self._data_path, filepath)
        assert os.path.exists(image_path), \
                'Path does not exist: {}\n{}'.format(image_path,self._data_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        #image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
        #                              self._image_set + '.txt')
        assert os.path.exists(self._image_set_file), \
                'Path does not exist: {}'.format(self._image_set_file)
        with open(self._image_set_file) as f:
            image_index = [x.strip() for x in f.readlines() 
                           if x.strip() and x.strip().rsplit(".",1)[-1] == "xml"]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(datasets.ROOT_DIR, 'data', 'VOCdevkit' + self._year)

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        ##disable cache
        #cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if False and  os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        cand_gt_roidb = [self._load_pascal_annotation(index)
                    for index in self._image_index]
        gt_roidb = [itm for itm in cand_gt_roidb if itm != None]
        exc = [c for c,i in enumerate(cand_gt_roidb) if i == None]
        exc.reverse()
        for i in exc:
            self._image_index.pop(i)

        size_ratio_file_raw = os.path.join(self._devkit_path, 'size_ratio_raw.txt')
        with open(size_ratio_file_raw,'w') as f:
            for ind,i in enumerate(gt_roidb):
                for c,j in enumerate(i['boxes']):
                    r = float(j[3]-j[1] + 1) / float(j[2]-j[0] + 1)
                    cls = i['gt_classes'][c]
                    f.write('{}:{}:{}:{},{},{},{}\n'.format(cls, r, self._image_index[ind], j[0],j[1],j[2],j[3]))

        return gt_roidb
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)
        return gt_roidb

    #seems not necessary in faster rcnn, to comment
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

        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def rpn_roidb(self):
        if  self._image_set != 'test':		#to-be-decided testset name
            gt_roidb = self.gt_roidb()
            #print "-----",len(gt_roidb),"------"
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

#    def _load_selective_search_roidb(self, gt_roidb):
#        filename = os.path.abspath(os.path.join(self.cache_path, '..',
#                                                'selective_search_data',
#                                                self.name + '.mat'))
#        assert os.path.exists(filename), \
#               'Selective search data not found at: {}'.format(filename)
#        raw_data = sio.loadmat(filename)['boxes'].ravel()
#
#        box_list = []
#        for i in xrange(raw_data.shape[0]):
#            box_list.append(raw_data[i][:, (1, 0, 3, 2)] - 1)
#
#        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._anno_path, index )
        # print 'Loading: {}'.format(filename)
        def get_data_from_tag(node, tag):
            return node.getElementsByTagName(tag)[0].childNodes[0].data

        with open(filename) as f:
            data = minidom.parseString(f.read())

        size_node = data.getElementsByTagName("size")[0]
        width = float(get_data_from_tag(size_node, "width"))
        height = float(get_data_from_tag(size_node, "height"))
       # im_ratio = height / width

       # if im_ratio < 0.25 or im_ratio > 4:
       #     return None

        objs = data.getElementsByTagName('object')
        if not self.config['use_diff']:
            # Exclude the samples labeled as difficult
            non_diff_objs = [obj for obj in objs
                             if int(get_data_from_tag(obj, 'difficult')) == 0]
            if len(non_diff_objs) != len(objs):
                print 'Removed {} difficult objects' \
                    .format(len(objs) - len(non_diff_objs))
            objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        ix = 0
        for obj in objs:
            # Make pixel indexes 0-based
            x1 = float(get_data_from_tag(obj, 'xmin')) #- 1
            y1 = float(get_data_from_tag(obj, 'ymin')) #- 1
            x2 = float(get_data_from_tag(obj, 'xmax')) #- 1
            y2 = float(get_data_from_tag(obj, 'ymax')) #- 1
            #ensure valid box
            x1 = 0 if x1<0 else (x1 if x1 <= width-1 else width - 1)
            x2 = 0 if x2<0 else (x2 if x2 <= width-1 else width - 1)
            y1 = 0 if y1<0 else (y1 if y1 <= height-1 else height - 1)
            y2 = 0 if y2<0 else (y2 if y2 <= height-1 else height - 1)
            if x1 > x2:
                t = x1
                x1 = x2
                x2 = t
            if y1 > y2:
                t = y1
                y1 = y2
                y2 = t

            name = str(get_data_from_tag(obj, "name")).lower().strip()
            if not name in self._class_to_ind:
                num_objs -= 1
                continue
            
            #r = float(y2-y1) / float(x2-x1)
            #if r < 0.7 or r > 1.4:
            #    num_objs -= 1
            #    continue
            cls = self._class_to_ind[name]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            ix += 1

        boxes.resize(num_objs,4)
        gt_classes.resize(num_objs)
        overlaps.resize(num_objs,self.num_classes)
        overlaps = scipy.sparse.csr_matrix(overlaps)

        if num_objs > 0:
            return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False}
        return None

    def _write_voc_results_file(self, all_boxes):
        use_salt = self.config['use_salt']
        comp_id = 'comp4'
        if use_salt:
            comp_id += '-{}'.format(os.getpid())

        # VOCdevkit/results/VOC2007/Main/comp4-44503_det_test_aeroplane.txt
        path = os.path.join(self._devkit_path, 'results', 'Main')
        if not os.path.exists(path):
            os.makedirs(path)
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} VOC results file'.format(cls)
            filename = os.path.join(path, comp_id + '_'+ 'det_'
                                    + self._image_set + '_' + cls + '.txt')
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))
        return comp_id

    def _do_matlab_eval(self, comp_id, output_dir='output'):
        #rm_results = self.config['cleanup']
        rm_results = False
        path = os.path.join(os.path.dirname(__file__),
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(datasets.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\',{:d}); quit;"' \
               .format(self._devkit_path, comp_id,
                       self._image_set, output_dir, int(rm_results))
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir):
        comp_id = self._write_voc_results_file(all_boxes)
        self._do_matlab_eval(comp_id, output_dir)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    d = datasets.pascal_voc('trainval', '2007')
    res = d.roidb
    from IPython import embed; embed()
