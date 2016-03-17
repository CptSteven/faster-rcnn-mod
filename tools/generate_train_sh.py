#!/usr/bin/env python
import _init_paths
import sys,os
from fast_rcnn.config import cfg, SHARED_DIR
import stat

def main(instance):
    fpath = os.path.join(cfg.ROOT_DIR,'experiments','scripts','train.sh') 

    #params
    tid = instance.tid
    imdb = instance.imdb_name
    net = instance.net
    net_name = instance.net_name
    gpu = instance.gpu_id
    pretrained_model = instance.pretrained_model
    cfgfile = instance.cfg_file
    test_proto = os.path.join(instance.net,'faster_rcnn_alt_opt/faster_rcnn_test.pt')

    #strings to write
    desc = '#\n#This is a machine-generated script for automatic training. \
    Do not edit it unless you are very confident and are familiar of faster-rcnn\n#\n'
    #logname = '{}_{}.txt.`date +\'%Y-%m-%d_%H-%M-%S\'`'.format(imdb, net_name)
    logname = '{}.log'.format(tid)
    #logpath = os.path.join('experiments','logs',logname)
    logpath = os.path.join(SHARED_DIR, 'train_logs', logname)
    train_cmd = './tools/train_faster_rcnn_alt_opt.py --gpu {} \\\n \
  --net {} \\\n \
  --weights {} \\\n \
  --imdb {} \\\n \
  --cfg {}'.format(gpu,net,pretrained_model,imdb,cfgfile)
    
    net_final = 'NET_FINAL=`grep "Final model:" ${LOG} | awk \'{print $3}\'`'

    test_cmd = './tools/test_net.py --gpu {} \\\n \
  --def {} \\\n \
  --net {} \\\n \
  --imdb {} \\\n \
  --cfg {}'.format(gpu, test_proto, '${NET_FINAL}',imdb+'test',cfgfile)



    with open(fpath,'w') as fp:
        fp.write('#!/bin/bash\n')
        fp.write(desc+'\n\n')
	fp.write('set -x\n')
        fp.write('set -e\n\n')
        fp.write('export PYTHONUNBUFFERED="True"\n\n')
        #fp.write('GPU_ID={}'.format(instance.gpu_id))

        fp.write('LOG="{}"\n'.format(logpath))
        fp.write('exec &> >(tee -a "$LOG")\n')
        fp.write('echo Logging output to "$LOG"\n\n')
        
        fp.write('time '+train_cmd+'\n\n')
        fp.write('set +x\n')
        fp.write(net_final+'\n')
        fp.write('set -x\n\n')
        fp.write('time '+test_cmd+'\n')

    os.chmod(fpath,stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
