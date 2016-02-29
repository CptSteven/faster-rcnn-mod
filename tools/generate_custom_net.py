#!/usr/bin/env python
import os
import argparse



def parseArgs():
    myparser = argparse.ArgumentParser(description='generate models of specified class(include _background)')
    myparser.add_argument('--cls',type=int)
    myparser.add_argument('--out','-o')
    return myparser.parse_args()

def main(ncls,outdir,steps,lr):

    nbox = ncls * 4
    cmt_tag = {"##ncls":ncls,"##nbboxpred":nbox,"##outdir":outdir,'##lr':lr}
    for c,i in enumerate(steps):
        key = '##step{}'.format(c+1)
        cmt_tag[key] = i
#    if os.path.exists('models'):
    if os.path.exists(outdir):
        print 'Outdir already exists, Please backup it.'
        exit(1)
    subdir_vgg16 = 'VGG16/faster_rcnn_alt_opt'
    subdir_vgg_m = 'VGG_CNN_M_1024/faster_rcnn_alt_opt'
    src = [subdir_vgg16, subdir_vgg_m]
    os.makedirs(os.path.join(outdir, subdir_vgg16))  
    os.makedirs(os.path.join(outdir, subdir_vgg_m))

    for i in src:
        files = os.listdir(os.path.join('models-bak',i))
        for j in files:
            src_path = os.path.join('models-bak',i,j)
            dst_path = os.path.join(outdir,i,j)
            fin = open(src_path)
            fout = open(dst_path,'w')
            lines = fin.readlines()
            k = 0
            while k < len(lines):
                tag = lines[k].strip()
                if tag not in cmt_tag:
                    fout.write(lines[k])
                    k += 1
                else:
                    n = cmt_tag[tag]
                    #import pdb
                    #pdb.set_trace()
                    l = lines[k+1].split('$',1)
                    fout.write('{}{}{}'.format(l[0],n,l[1]))
                    k += 2

if __name__ == '__main__':
    args = parseArgs()
    ncls = args.cls
    outdir = args.out
    main(ncls,outdir)
