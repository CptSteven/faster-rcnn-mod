#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Faster R-CNN network using alternating optimization.
This tool implements the alternating optimization algorithm described in our
NIPS 2015 paper ("Faster R-CNN: Towards Real-time Object Detection with Region
Proposal Networks." Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun.)
"""

import _init_paths
from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
from rpn.generate import imdb_proposals
import datasets.imdb
import argparse
import pprint
import numpy as np
import sys, os
import multiprocessing as mp
import cPickle
import shutil

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Faster R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--net_name', dest='net_name',
                        help='network name (e.g., "ZF")',
                        default=None, type=str)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='voc_2007_trainval', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

class TrainingInstance(object):
    def __init__(self):
        self.imdb_name = ''
        self.devkit = ''
        self.pretrained_model = 'data/imagenet_models/VGG16.v2.caffemodel'
        self.cfg_file = 'experiments/cfgs/faster_rcnn_alt_opt.yml'
        self.net = ''
        self.net_name = ''
        self.rpn_iter1 = 0
        self.rcnn_iter1 = 0
        self.rpn_iter2 = 0
        self.rcnn_iter2 = 0
        self.lr = 0
        self.step = 0
        self.gpu_id = 0
        self.set_cfgs = None

    def getClassNum(self):
        clspath = os.path.join(self.devkit, 'classes.lst')
        assert os.path.exists(clspath)
        with open(clspath) as fcls:
            classes = [l for l in fcls if l.strip() != '']
        self.cls_num = len(classes)

    def rename(newname):
        self.imdb_name = newname

    def validate(self):
        return NotImplemented

def get_roidb(imdb_name, rpn_file=None):
    imdb = get_imdb(imdb_name)
    print 'Loaded dataset `{:s}` for training'.format(imdb.name)
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
    print 'Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD)
    if rpn_file is not None:
        imdb.config['rpn_file'] = rpn_file
    roidb = get_training_roidb(imdb)
    return roidb, imdb

def get_solvers(instance):
    # Faster R-CNN Alternating Optimization
    n = 'faster_rcnn_alt_opt'
    # Solver for each training stage
    solvers = [[n, 'stage1_rpn_solver60k80k.pt'],
               [n, 'stage1_fast_rcnn_solver30k40k.pt'],
               [n, 'stage2_rpn_solver60k80k.pt'],
               [n, 'stage2_fast_rcnn_solver30k40k.pt']]
    solvers = [os.path.join(cfg.ROOT_DIR, instance.net, *s) for s in solvers]
    # Iterations for each training stage
    max_iters = [instance.rpn_iter1, instance.rcnn_iter1, \
                 instance.rpn_iter2, instance.rcnn_iter2]
    # max_iters = [100, 100, 100, 100]
    # Test prototxt for the RPN
    rpn_test_prototxt = os.path.join(
        cfg.ROOT_DIR, instance.net, n, 'rpn_test.pt')
    return solvers, max_iters, rpn_test_prototxt

# ------------------------------------------------------------------------------
# Pycaffe doesn't reliably free GPU memory when instantiated nets are discarded
# (e.g. "del net" in Python code). To work around this issue, each training
# stage is executed in a separate process using multiprocessing.Process.
# ------------------------------------------------------------------------------

def _init_caffe(cfg):
    """Initialize pycaffe in a training process.
    """

    import caffe
    # fix the random seeds (numpy and caffe) for reproducibility
    np.random.seed(cfg.RNG_SEED)
    caffe.set_random_seed(cfg.RNG_SEED)
    # set up caffe
    caffe.set_mode_gpu()
    caffe.set_device(cfg.GPU_ID)

def train_rpn(queue=None, imdb_name=None, init_model=None, solver=None,
              max_iters=None, cfg=None):
    """Train a Region Proposal Network in a separate training process.
    """

    # Not using any proposals, just ground-truth boxes
    cfg.TRAIN.HAS_RPN = True
    cfg.TRAIN.BBOX_REG = False  # applies only to Fast R-CNN bbox regression
    cfg.TRAIN.PROPOSAL_METHOD = 'gt'
    cfg.TRAIN.IMS_PER_BATCH = 1
    print 'Init model: {}'.format(init_model)
    print('Using config:')
    pprint.pprint(cfg)

    import caffe
    _init_caffe(cfg)

    roidb, imdb = get_roidb(imdb_name)
    print 'roidb len: {}'.format(len(roidb))
    output_dir = get_output_dir(imdb, None)
    print 'Output will be saved to `{:s}`'.format(output_dir)

    model_paths = train_net(solver, roidb, output_dir,
                            pretrained_model=init_model,
                            max_iters=max_iters)
    # Cleanup all but the final model
    for i in model_paths[:-1]:
        os.remove(i)
    rpn_model_path = model_paths[-1]
    # Send final model path through the multiprocessing queue
    queue.put({'model_path': rpn_model_path})

def rpn_generate(queue=None, imdb_name=None, rpn_model_path=None, cfg=None,
                 rpn_test_prototxt=None):
    """Use a trained RPN to generate proposals.
    """

    cfg.TEST.RPN_PRE_NMS_TOP_N = -1     # no pre NMS filtering
    cfg.TEST.RPN_POST_NMS_TOP_N = 2000  # limit top boxes after NMS
    print 'RPN model: {}'.format(rpn_model_path)
    print('Using config:')
    pprint.pprint(cfg)

    import caffe
    _init_caffe(cfg)

    # NOTE: the matlab implementation computes proposals on flipped images, too.
    # We compute them on the image once and then flip the already computed
    # proposals. This might cause a minor loss in mAP (less proposal jittering).
    imdb = get_imdb(imdb_name)
    print 'Loaded dataset `{:s}` for proposal generation'.format(imdb.name)

    # Load RPN and configure output directory
    rpn_net = caffe.Net(rpn_test_prototxt, rpn_model_path, caffe.TEST)
    output_dir = get_output_dir(imdb, None)
    print 'Output will be saved to `{:s}`'.format(output_dir)
    # Generate proposals on the imdb
    rpn_proposals = imdb_proposals(rpn_net, imdb)
    # Write proposals to disk and send the proposal file path through the
    # multiprocessing queue
    rpn_net_name = os.path.splitext(os.path.basename(rpn_model_path))[0]
    rpn_proposals_path = os.path.join(
        output_dir, rpn_net_name + '_proposals.pkl')
    with open(rpn_proposals_path, 'wb') as f:
        cPickle.dump(rpn_proposals, f, cPickle.HIGHEST_PROTOCOL)
    print 'Wrote RPN proposals to {}'.format(rpn_proposals_path)
    queue.put({'proposal_path': rpn_proposals_path})

def train_fast_rcnn(queue=None, imdb_name=None, init_model=None, solver=None,
                    max_iters=None, cfg=None, rpn_file=None):
    """Train a Fast R-CNN using proposals generated by an RPN.
    """

    cfg.TRAIN.HAS_RPN = False           # not generating prosals on-the-fly
    cfg.TRAIN.PROPOSAL_METHOD = 'rpn'   # use pre-computed RPN proposals instead
    cfg.TRAIN.IMS_PER_BATCH = 2
    print 'Init model: {}'.format(init_model)
    print 'RPN proposals: {}'.format(rpn_file)
    print('Using config:')
    pprint.pprint(cfg)

    import caffe
    _init_caffe(cfg)

    roidb, imdb = get_roidb(imdb_name, rpn_file=rpn_file)
    output_dir = get_output_dir(imdb, None)
    print 'Output will be saved to `{:s}`'.format(output_dir)
    # Train Fast R-CNN
    model_paths = train_net(solver, roidb, output_dir,
                            pretrained_model=init_model,
                            max_iters=max_iters)
    # Cleanup all but the final model
    for i in model_paths[:-1]:
        os.remove(i)
    fast_rcnn_model_path = model_paths[-1]
    # Send Fast R-CNN model path over the multiprocessing queue
    queue.put({'model_path': fast_rcnn_model_path})

def getTrainingInstance():
    """
    access to the database to check and return a training instance
    """    
    """
    instance = TrainingInstance()
    
    instance.getClsNum
    """
    instance = TrainingInstance()
    instance.imdb_name = 'pascal'
    instance.devkit = os.path.join(cfg.ROOT_DIR,'data/pascal')
    instance.pretrained_model = os.path.join(cfg.ROOT_DIR, 'data/imagenet_models/VGG_CNN_M_1024.v2.caffemodel')
# cfg file will not be changed in the foreseeable future
    instance.cfg_file = os.path.join(cfg.ROOT_DIR, 'experiments/cfgs/faster_rcnn_alt_opt.yml')
    instance.net = os.path.join(cfg.ROOT_DIR, 'models21/VGG_CNN_M_1024')
    instance.net_name = instance.net.rsplit('/',1)[-1]
    instance.rpn_iter1 = 100
    instance.rcnn_iter1 = 100
    instance.rpn_iter2 = 100
    instance.rcnn_iter2 = 100
    instance.lr = 0.001
    instance.step = 20
    instance.gpu_id = 0
    instance.set_cfgs = None
    return instance

def generateFactory(imdb,devkit):
#    cmt_tag = {'##devkit':devkit,'##imdb':imdb,'##imdbtest':imdb+'t'}
    infile = os.path.join(cfg.ROOT_DIR, 'factory.template')
    outfile = os.path.join(cfg.ROOT_DIR, 'factory.py')
    with open(infile) as fin:
        with open(outfile,'w') as fout:
            lines = fin.readlines()
    item = '__sets[\'{}\'] = (lambda data=\'{}\',path=devkit:\n \
                  datasets.pascal_voc(data,path))'
    k = 0
    while k < len(lines):
        tag = lines[k].strip()
        k += 1
        if tag[0,2] != '##'
            fout.write(lines[k])
        else:
            fout.write('devkit = {}\n'.format(devkit))
            fout.write(item.format(instance.imdb_name), instance.imdb_name)
            fout.write(item.format(instance.imdb_name+'test', 'test'))

if __name__ == '__main__':
#    args = parse_args()

#    print('Called with args:')
#    print(args)
    instance = getTrainingInstance()
    imdb_name = instance.imdb_name
    pretrained_model = instance.pretrained_model
    cfg_file = instance.cfg_file
    netdir,netname = instance.net.rsplit('/',1)
    init_lr = instance.lr
    step = instance.step

    if instance.cfg_file is not None:
        cfg_from_file(instance.cfg_file)
    if instance.set_cfgs is not None:
        cfg_from_list(instance.set_cfgs)
    cfg.GPU_ID = instance.gpu_id
    if not os.path.exists(instance.net):
        import sys
        from generate_custom_net import main
        if not os.path.exists(os.path.join(cfg.ROOT_DIR,'models')):
            os.popen('rm -rf ../models')
            main(instance.cls_num,'models')
            instance.net = 'models/VGG16'
            instance.net_name = 'VGG16'

    # --------------------------------------------------------------------------
    # Pycaffe doesn't reliably free GPU memory when instantiated nets are
    # discarded (e.g. "del net" in Python code). To work around this issue, each
    # training stage is executed in a separate process using
    # multiprocessing.Process.
    # --------------------------------------------------------------------------

    # queue for communicated results between processes
    mp_queue = mp.Queue()
    # solves, iters, etc. for each training stage
    solvers, max_iters, rpn_test_prototxt = get_solvers(instance)

    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'Stage 1 RPN, init from ImageNet model'
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

    cfg.TRAIN.SNAPSHOT_INFIX = 'stage1'
    mp_kwargs = dict(
            queue=mp_queue,
            imdb_name=instance.imdb_name,
            init_model=instance.pretrained_model,
            solver=solvers[0],
            max_iters=max_iters[0],
            cfg=cfg)
    p = mp.Process(target=train_rpn, kwargs=mp_kwargs)
    p.start()
    rpn_stage1_out = mp_queue.get()
    p.join()

    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'Stage 1 RPN, generate proposals'
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

    mp_kwargs = dict(
            queue=mp_queue,
            imdb_name=instance.imdb_name,
            rpn_model_path=str(rpn_stage1_out['model_path']),
            cfg=cfg,
            rpn_test_prototxt=rpn_test_prototxt)
    p = mp.Process(target=rpn_generate, kwargs=mp_kwargs)
    p.start()
    rpn_stage1_out['proposal_path'] = mp_queue.get()['proposal_path']
    p.join()

    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'Stage 1 Fast R-CNN using RPN proposals, init from ImageNet model'
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

    cfg.TRAIN.SNAPSHOT_INFIX = 'stage1'
    mp_kwargs = dict(
            queue=mp_queue,
            imdb_name=instance.imdb_name,
            init_model=instance.pretrained_model,
            solver=solvers[1],
            max_iters=max_iters[1],
            cfg=cfg,
            rpn_file=rpn_stage1_out['proposal_path'])
    p = mp.Process(target=train_fast_rcnn, kwargs=mp_kwargs)
    p.start()
    fast_rcnn_stage1_out = mp_queue.get()
    p.join()

    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'Stage 2 RPN, init from stage 1 Fast R-CNN model'
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

    cfg.TRAIN.SNAPSHOT_INFIX = 'stage2'
    mp_kwargs = dict(
            queue=mp_queue,
            imdb_name=instance.imdb_name,
            init_model=str(fast_rcnn_stage1_out['model_path']),
            solver=solvers[2],
            max_iters=max_iters[2],
            cfg=cfg)
    p = mp.Process(target=train_rpn, kwargs=mp_kwargs)
    p.start()
    rpn_stage2_out = mp_queue.get()
    p.join()

    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'Stage 2 RPN, generate proposals'
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

    mp_kwargs = dict(
            queue=mp_queue,
            imdb_name=instance.imdb_name,
            rpn_model_path=str(rpn_stage2_out['model_path']),
            cfg=cfg,
            rpn_test_prototxt=rpn_test_prototxt)
    p = mp.Process(target=rpn_generate, kwargs=mp_kwargs)
    p.start()
    rpn_stage2_out['proposal_path'] = mp_queue.get()['proposal_path']
    p.join()

    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'Stage 2 Fast R-CNN, init from stage 2 RPN R-CNN model'
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

    cfg.TRAIN.SNAPSHOT_INFIX = 'stage2'
    mp_kwargs = dict(
            queue=mp_queue,
            imdb_name=instance.imdb_name,
            init_model=str(rpn_stage2_out['model_path']),
            solver=solvers[3],
            max_iters=max_iters[3],
            cfg=cfg,
            rpn_file=rpn_stage2_out['proposal_path'])
    p = mp.Process(target=train_fast_rcnn, kwargs=mp_kwargs)
    p.start()
    fast_rcnn_stage2_out = mp_queue.get()
    p.join()

    # Create final model (just a copy of the last stage)
    final_path = os.path.join(
            os.path.dirname(fast_rcnn_stage2_out['model_path']),
            instance.net_name + '_faster_rcnn_final.caffemodel')
    print 'cp {} -> {}'.format(
            fast_rcnn_stage2_out['model_path'], final_path)
    shutil.copy(fast_rcnn_stage2_out['model_path'], final_path)
    print 'Final model: {}'.format(final_path)
