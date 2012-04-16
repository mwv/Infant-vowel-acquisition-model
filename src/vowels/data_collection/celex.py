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
data_collection.celex:
interface to dutch phonetic transcriptions of wordforms in celex corpus

sampa specification: http://www.phon.ucl.ac.uk/home/sampa/
'''

from __future__ import division

__author__ = 'Maarten Versteegh'
__date__ = 'Apr 6, 2012'

import re
import os

from bidict import bidict
from ..config.paths import cfg_celexdir
from ..util.transcript_formats import disc_to_sampa

CELEX_FILE = os.path.join(cfg_celexdir, 'DUTCH/DPW/DPW.CD')

class PhonLookup(object):
    def __init__(self,
                 dict_fid=CELEX_FILE,
                 cleanup=True,
                 toSampa=True):
        self.dikt={}
        for line in open(dict_fid,'r'):
            line=line.strip().split('\\')
            trans = line[4]
            if cleanup:
                trans = stripSyllableMarkers(stripStressMarkers(trans))
                if toSampa:
                    trans = transToSampa(trans)
            self.dikt[line[1].lower()] = trans

    def __getitem__(self, item):
        try:
            return self.dikt[item]
        except:
            return None

    def __contains__(self, item):
        return (item in self.dikt)

    def __iter__(self):
        return iter(self.dikt)

def stripStressMarkers(trans):
    return re.sub("\'","", trans)

def transToSampa(trans):
    return map(lambda x:disc_to_sampa(x), list(trans))

def stripSyllableMarkers(trans):
    return re.sub("-","",trans)



