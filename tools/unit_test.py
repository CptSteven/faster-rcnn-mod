#!/usr/bin/env python

from datasets import imdb, pascal_voc, factory
#import datasets.imdb
#import datasets.factory

idb = datasets.factory.get_imdb("rivet_test")
idb.append_flipped_images()
