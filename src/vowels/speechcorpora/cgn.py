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
vowels.data_collection.cgn:


interface to force-aligned part of cgn corpus
components f. (interviews, debates, meetings), and o. (read text)

'''

from glob import iglob
import os
import re
from copy import deepcopy
from ..config.paths import cfg_cgntgdir, cfg_cgnwavdir
from .textgrid import TextGrid, Interval, IntervalTier

from ..util.transcript_formats import htk_to_sampa, sampa_merge
from ..util.functions import compose

class CGN(object):
    def __init__(self, convert_phones=True, merge_vowels=True):
        self.convert_phones = convert_phones
        self.merge_vowels = merge_vowels
        self.name = 'cgn'
        if convert_phones:
            if merge_vowels:
                self.phone_convert_func = compose(sampa_merge, htk_to_sampa)
            else:
                self.phone_convert_func = htk_to_sampa
        else:
            self.convert_func = lambda x:x
    
    def __repr__(self):
        return self.name
    
    def _textgrids(self):
        for fname in iglob(os.path.join(cfg_cgntgdir, '*.ltg')):
            basename = fname.split(os.sep)[-1][:-4]
            namep = re.compile(r'(?P<name>[\w\d]+)__\d+-\d+$')
            tg = TextGrid(name=basename)
            fid = open(fname, 'r')
            tg.read(fid)
            fid.close()            
            if self.convert_phones:
                it = tg['phone alignment']
                for i in range(len(it)):
                    it[i].mark = self.phone_convert_func(it[i].mark)
            yield tg
            
    def _wavfile(self, basename):
        return os.path.join(cfg_cgnwavdir, basename + '.wav')
    
    def utterances(self):
        for tg in self._textgrids():
            yield (self._wavfile(tg.name), tg)
    
