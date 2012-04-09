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
vowels.data_collection.ifa:

'''
from glob import iglob
import os
import re

from itertools import chain

from ..config.paths import cfg_ifadir
from .textgrid import TextGrid

ifa_tgdir = os.path.join(cfg_ifadir, 'textgrids')

class IFA_TextGrids(object):
    """ interface to ifa's textgrids"""
    def __init__(self,
                 corpora=None):
        valid_corpora = [d for d in os.listdir(ifa_tgdir)
                         if os.path.isdir(os.path.join(ifa_tgdir, d))]
        
        if corpora is None:
            self.corpora = valid_corpora
        elif all(x in valid_corpora for x in corpora):
            self.corpora = corpora
        else:
            raise ValueError, 'Invalid corpus choice. Valid choices are %s' % ', '.join(valid_corpora)
    
    def iter_textgrids(self):
        """Generator object for TextGrid objects in specified corpora"""
        for fname in chain.from_iterable(iglob(os.path.join(ifa_tgdir, d, '*.ltg'))
                                         for d in self.corpora):
            name = re.search(r'([FM]\d\d\w)%s(?P<name>\w*?).ltg$' % os.sep,
                             fname).group('name')
            tg = TextGrid(name=name)
            fid = open(fname,'r')
            tg.read(fid)
            fid.close()
            yield tg
                                             
        
        
   

