#!/usr/bin/env python
"""
Description: This script works as such:
  Firstly get training instance via network( by either network commu or remote database query). 
  Secondly create factory.py and models(if necessary). Lastly, train net.
Author: Steven
Date: Feb. 23, 2016
"""
import _init_paths
import sys,os,json
import argparse
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, SHARED_DIR
import generate_custom_net
import sql

#SHARED_DIR = '/data/disk3/faster_rcnn_train'

class TrainingInstance(object):
    def __init__(self):
        self.tid=None
        self.imdb_name = ''
        self.devkit = ''
        self.imdir = ''
        self.pretrained_model = 'data/imagenet_models/VGG16.v2.caffemodel'
        self.cfg_file = ''
        self.net = ''
        self.netdir = ''
        self.net_name = ''
        self.rpn_iter1 = 0
        self.rcnn_iter1 = 0
        self.rpn_iter2 = 0
        self.rcnn_iter2 = 0
        self.lr = 0
        self.steps = []
        self.rpn_step1 = 0
        self.rcnn_step1 = 0
        self.rpn_step2 = 0
        self.rcnn_step2 = 0
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
        if not os.path.exists(self.devkit):
            return False
        if not os.path.exists(self.pretrained_model):
            return False
        self.net = self.net.rstrip('/')
        self.netdir,self.net_name = os.path.split(self.net)
        self.getClassNum()
        return True
   
    def setDefaultNet(self):
        self.net = 'models/VGG_CNN_M_1024'
        self.net_name = 'VGG_CNN_M_1024'
        self.pretrained_model = os.path.join(cfg.ROOT_DIR, 'data/imagenet_models/VGG_CNN_M_1024.v2.caffemodel')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id',type=int)
    args = parser.parse_args()
    return args

def getTrainingInstance(tid=None,ip=None):
    """
    access to the database to check and return a training instance

    """
#    dbconn = sql.MySQLConnection('192.168.1.90','test','test','zb_label')
#    dbconn.connect()
#    sqlstr = 'select * from zb_train where id = {}'.format(id)
#    #sqlstr = 'select * from zb_train where id = {} and ip={}'.format(id,ip)
#    dbconn.query(sqlstr)
#    data = dbconn.fetchAll()
#    dbconn.close()
#    if len(data) != 1:
#        return None
#    data = data[0]
#
#    status = data[15]
#    assert status == 1
#
#    instance = TrainingInstance()
#    instance.tid = data[0]
#    instance.imdb_name = data[1]    
#    instance.devkit = os.path.join(SHARED_DIR, data[2])
#    instance.rpn_iter1 = data[3]
#    instance.rcnn_iter1 = data[4]
#    instance.rpn_iter2 = data[5]
#    instance.rcnn_iter2 = data[6]
#    instance.lr = data[7]
#    instance.rpn_step1 = data[8]
#    instance.rcnn_step1 = data[9]
#    instance.rpn_step2 = data[10]
#    instance.rcnn_step2 = data[11]
#    instance.gpu_id = data[12]
#    instance.net = data[13]
#    instance.pretrained_model = data[14]
#
#    instance.imdir = os.path.join(SHARED_DIR,'images')
#
#    instance.steps = [instance.rpn_step1,instance.rcnn_step1,instance.rpn_step2,instance.rcnn_step2]
#    instance.cfg_file = os.path.join(instance.devkit, 'faster_rcnn_alt_opt.yml')
#
#    instance.getClassNum()

#"""
#Test mode, set training instance manully
#"""
    instance = TrainingInstance()
    instance.tid = tid
    instance.imdb_name = 'pacoco'
    instance.devkit = os.path.join(cfg.ROOT_DIR,'data/pacoco')
    #if use relative image path , uncomment
    instance.imdir = instance.devkit
    instance.pretrained_model = os.path.join(cfg.ROOT_DIR, 'data/imagenet_models/VGG_CNN_M_1024.v2.caffemodel')
# cfg file will not be changed in the foreseeable future
    #instance.net = os.path.join(cfg.ROOT_DIR, 'models21/VGG_CNN_M_1024')
    #instance.net_name = instance.net.rsplit('/',1)[-1]
    instance.rpn_iter1 = 80000
    instance.rcnn_iter1 = 40000
    instance.rpn_iter2 = 80000
    instance.rcnn_iter2 = 40000
    instance.lr = 0.001
    instance.gpu_id = 0
    instance.set_cfgs = None

    instance.rpn_step1 = 60000
    instance.rcnn_step1 = 30000
    instance.rpn_step2 = 60000
    instance.rcnn_step2 = 30000
    instance.steps = [instance.rpn_step1,instance.rcnn_step1,instance.rpn_step2,instance.rcnn_step2]
    instance.getClassNum()
    instance.cfg_file = os.path.join(instance.devkit, 'faster_rcnn_alt_opt.yml')

    return instance

def generateFactory(instance):
#    cmt_tag = {'##devkit':devkit,'##imdb':imdb,'##imdbtest':imdb+'t'}
    imdb = instance.imdb_name
    devkit = instance.devkit

    infile = os.path.join(cfg.ROOT_DIR, 'lib','datasets','factory.template')
    outfile = os.path.join(cfg.ROOT_DIR, 'lib','datasets','factory.py')
    item = '__sets[\'{}\'] = (lambda data=\'{}\',path=devkit:\n \
                  datasets.pascal_voc(data,path{}))\n'
    imdir = ',\'' + instance.imdir + '\'' if instance.imdir != '' else ''

    with open(infile) as fin:
        with open(outfile,'w') as fout:
            lines = fin.readlines()
            k = 0
            while k < len(lines):
                tag = lines[k].strip()
                if tag[0:2] != '##':
                    fout.write(lines[k])
                else:
                    fout.write('devkit = \'{}\'\n'.format(devkit))
                    fout.write(item.format(imdb, imdb,imdir))
                    fout.write(item.format(imdb+'test', 'test',imdir))
                k += 1

def generateCFG(instance):
    with open(instance.cfg_file,'w') as fp:
        fp.write('EXP_DIR: faster_rcnn_alt_opt\n')
        fp.write('GPU_ID: {}\n'.format(instance.gpu_id))
        fp.write('TRAIN:\n')
        fp.write('  BBOX_THRESH: 0.5\n')
        fp.write('  RPN_ITER1: {}\n'.format(instance.rpn_iter1))
        fp.write('  RCNN_ITER1: {}\n'.format(instance.rcnn_iter1))
        fp.write('  RPN_ITER2: {}\n'.format(instance.rpn_iter2))
        fp.write('  RCNN_ITER2: {}\n'.format(instance.rcnn_iter2))
        fp.write('TEST:\n')
        fp.write('  HAS_RPN: True')

def generateVOCCode(devkit):
    target_path = os.path.join(cfg.ROOT_DIR, 'VOCcode')
    #link_path = os.path.abspath
    if not os.path.exists(os.path.join(devkit,'VOCcode')):
        os.system('ln -s -t {} {}'.format(devkit,target_path))
    
def getAccuracy(tid):
    acc = dict()
    logpath = os.path.join(SHARED_DIR, 'train_logs', tid+'.log')
    with open(logpath) as fl:
	for l in fl:
	    if l[:3] == '!!!':
	        name,rate = l[3:].split(':') 
	        name = name.strip()
                rate = rate.split()[0].strip()
                acc[name] = rate
    return acc

def main():
    args = parse_args()
    ins_id = args.id

    instance = getTrainingInstance(ins_id)
    assert instance != None

    generateCFG(instance)
    cfg_from_file(instance.cfg_file)
    if instance.set_cfgs is not None:
        cfg_from_list(instance.set_cfgs)

    #generate model if it not exists yet

    if instance.net == '': 
        instance.setDefaultNet()
        models_path = os.path.join(cfg.ROOT_DIR, 'models')
        if True or os.path.exists(models_path):
            os.popen('rm -rf {}'.format(models_path))
        generate_custom_net.main(instance.cls_num+1,'models',instance.steps,instance.lr)

    if not os.path.exists(instance.net):
        generate_custom_net.main(instance.cls_num+1,instance.net,instance.steps,instance.lr)

    #generate factory.py
    generateFactory(instance)

    #generate train.sh
    import generate_train_sh
    generate_train_sh.main(instance)

    #make symbolic link to VOCCode
    generateVOCCode(instance.devkit)

    if instance.validate() == False:
        print 'Error in training instance.'
        exit(1)

    #dbconn = sql.MySQLConnection('192.168.1.90','test','test','zb_label')
    #dbconn.connect()
    #sqlstr = 'update zb_train set status = 2 where id = {}'.format(ins_id)
    #dbconn.query(sqlstr)
    #dbconn.commit()

    #start training
    os.system('experiments/scripts/train.sh')
    acc_rate = getAccuracy(ins_id)
    #sqlstr = 'update zb_train set status = 3 , accuracy = {} where id = {}'.format(json.dumps(acc_rate), ins_id)
    #dbconn.query(sqlstr)
    #dbconn.commit()
    #dbconn.close()

    print 'ok'


if __name__ == '__main__':
    main()

    #os.system('experiments/scripts/train_pascal.sh 0 VGG_CNN_M_1024')
    #os.system(r'time ./tools/train_faster_rcnn_alt_opt.py --gpu 0 --net_name VGG_CNN_M_1024 --weights data/imagenet_models/VGG_CNN_M_1024.v2.caffemodel --imdb pascal --cfg experiments/cfgs/faster_rcnn_alt_opt.yml')

    #import train_faster_rcnn_alt_opt
    #train_faster_rcnn_alt_opt.main()
    print 'done\n\n'
