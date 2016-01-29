#!/usr/bin/env python

import datasets.pascal_voc#imdb#, pascal_voc, factory
import datasets.imdb
import datasets.factory
_DEBUG_ON = False
if _DEBUG_ON:
    import pdb
    pdb.set_trace()
idb = datasets.factory.get_imdb("rivet_test")
idb.set_proposal_method("rpn")
idb.config["rpn_file"] = "/root/tools/py-faster-rcnn/output/faster_rcnn_alt_opt/rivet_test/vgg_cnn_m_1024_rpn_stage1_iter_80000_proposals.pkl"
idb.append_flipped_images()
