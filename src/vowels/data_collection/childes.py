#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------
# author:
# Maarten Versteegh
# http://lands.let.ru.nl/~versteegh/
# maartenversteegh AT gmail DOT com
# Centre for Language Studies
# Radboud University Nijmegen
# 
# Licensed under GPLv3
# ------------------------------------

'''
data_collection.childes:

'''

from __future__ import division

__author__ = 'Maarten Versteegh'
__date__ = 'Apr 6, 2012'

import re
import os
from glob import iglob

from itertools import chain

from ..config.paths import cfg_childesdir



class ChildesCorpus(object):
    def __init__(self, 
                 corpora=None, # list of corpora to include
                 exclude_target_child=True, # do not yield utterances of the target child
                 exclude_identifiers=True, # yield just the utterances, not who speaks them
                 yield_words=True, # yield words instead of lines
                 exclude_eol=True): # exclude eol markers
        # assume only corpus directories in cfg_childesdir
        valid_corpora = [d for d in os.listdir(cfg_childesdir)
                         if os.path.isdir(os.path.join(cfg_childesdir, d))]
        if corpora is None:
            self.corpora = valid_corpora
        elif all(x in valid_corpora for x in corpora):
            self.corpora = corpora
        else:
            raise ValueError, 'Invalid corpus choice. Valid choices are %s' % ', '.join(valid_corpora)
        
        self.exclude_target_child = exclude_target_child
        self.exclude_identifiers = exclude_identifiers
        self.yield_words = yield_words
        
    def list_directories(self, abspath=True):
        if abspath:
            return [os.path.join(cfg_childesdir, x)
                    for x in self.corpora]
        else:
            return self.corpora
        
    def __iter__(self):
        """iterate over the utterances of caregivers"""
        for fname in chain.from_iterable(iglob(os.path.join(cfg_childesdir, d, '*.cha'))
                                         for d in self.corpora):
            for line in get_lines_from_interactors(fname, get_caregivers(fname, exclude_target_child = self.exclude_target_child)):
                if self.exclude_identifiers:
                    if self.yield_words:
                        for word in line[1]:
                            yield word
                    else:
                        yield line[1]
                else:
                    yield line
        
def get_caregivers(childes_file, exclude_target_child=True):
    tabwhitepat = re.compile('\t|\s')
    id_found = False
    interactors=[]
    for line in open(childes_file, 'r'):
        line = re.split(tabwhitepat,line.strip())
        if line[0] == '@ID:':
            id_found = True
            id_info = line[1].split('|')
            if exclude_target_child:
                if id_info[-4] != 'Target_Child':
                    interactors.append(id_info[2])
            else:
                interactors.append(id_info[2])
        elif id_found:
            break
        else:
            continue
    return interactors

def get_lines_from_interactors(childes_file, interactors):
    p = re.compile(r'^' + '|'.join(map(lambda x:r'\*'+x, interactors)))
    for line in open(childes_file,'r'):
        m = p.match(line)
        if m:
            line = line.split('\t')
            try:
                yield line[0],line[1].strip()
            except:
                continue
