#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

#CLASSES = ('__background__', 'bag')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return 0
    color = (0,100,0)
    for i in inds:
        confi = dets[i, -1]
        box = [int(x+0.5) for x in dets[i]]
        p0 = (box[0], box[1])
        p1 = (box[2], box[3])
        tag = class_name + ': '+ str(confi)
        cv2.rectangle(im, p0, p1, color)
        cv2.putText(im, tag, (p0[0]+15, p0[1]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
    return len(inds)

def detect(net, im):
    """ """


    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    res = 0
    global CLASSES
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        res += vis_detections(im, cls, dets, thresh=CONF_THRESH)
    return im,res

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('imdir' , help='Folder whose elements are images')
    parser.add_argument('outdir')
    parser.add_argument('--testset')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--def', dest='demo_net', help='Network to use [vgg16]')
    parser.add_argument('--net', help='caffemodel to use')
    parser.add_argument('--cls')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    #prototxt = os.path.join(cfg.ROOT_DIR, 'models', NETS[args.demo_net][0],
    #                        'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    #caffemodel = os.path.join(cfg.ROOT_DIR, 'data', 'faster_rcnn_models',
    #                          NETS[args.demo_net][1])
    prototxt = args.demo_net
    caffemodel = args.net

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    with open(args.cls) as fc:
        CLASSES = ['__background__'] + [l.strip() for l in fc if l.strip()]
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    files = []
    if args.testset == None:
        files = os.listdir(args.imdir)
    else:
        with open(args.testset) as fl:
            files = [l.strip().split('.')[0]+'.jpg' for l in fl if l.strip()]
    for i in files:
        # Load the demo image
        splitline = i.rsplit('/',1)
        if len(splitline) == 1:
            subdir = ''
            fname = splitline[0]
        else:
            subdir = splitline[0]
            fname = splitline[1]
        impath = os.path.join(args.imdir, i)
        im = cv2.imread(impath)
        if im == None:
            print "Can't open {}".format(impath)
            continue
        res = detect(net, im)   
        if res[1] == 0:
            print 'No target class in ' + i
            continue
        outdir = os.path.join(args.outdir,subdir)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        outpath = os.path.join(outdir, fname+'_out.jpg')
        cv2.imwrite(outpath, res[0])

