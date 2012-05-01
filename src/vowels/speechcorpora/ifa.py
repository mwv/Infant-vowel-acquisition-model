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

interface to ifa corpus

only using FR (fixed text retold), VI (variable informal story), VR (variable story retold)


'''
from glob import iglob
import os
import re

from itertools import chain, imap
from functools import partial

from ..config.paths import cfg_ifadir
from .textgrid import TextGrid
from ..util.transcript_formats import cgn_to_sampa, sampa_merge
from ..util.functions import compose

ifa_tgdir = os.path.join(cfg_ifadir, 'textgrids')
ifa_wavdir = os.path.join(cfg_ifadir, 'wavs')

def valid_speakers():
    return [d for d in os.listdir(ifa_tgdir) if os.path.isdir(os.path.join(ifa_tgdir, d))]
    
def female_speakers():
    return filter(lambda x:x.startswith('F'), valid_speakers())
    
def male_speakers():
    return filter(lambda x:x.startswith('M'), valid_speakers())

class IFA(object):
    """ interface to ifa's textgrids"""
    def __init__(self,
                 convert_phones=True,
                 merge_vowels=True,
                 speakers=None, 
                 gender=None):
        self.name = 'ifa'
        self.convert_phones = convert_phones
        self.merge_vowels=merge_vowels
        if gender == 'male':
            self.speakers = male_speakers()
        elif gender == 'female':
            self.speakers = female_speakers()
        if speakers is None:
            self.speakers = valid_speakers()
        elif all(x in valid_speakers() for x in speakers):
            self.speakers = speakers
        else:
            raise ValueError, 'Invalid speaker choice. Valid choices are %s' % ', '.join(self.valid_speakers())
        
        if convert_phones:
            if merge_vowels:
                self.phone_convert_func = partial(self._phone_convert_func, merge=True)
            else:
                self.phone_convert_func = self._phone_convert_func
        else:
            self.convert_func = lambda x:x
            
    def _phone_convert_func(self, p, merge=False):
        p = re.sub(r'[\^\d]+$','', p)
        if merge:
            return sampa_merge(cgn_to_sampa(p))
        else:
            return cgn_to_sampa(p)        
        
    def __repr__(self):
        return self.name + ''.join(self.speakers)

    def _wavfile(self, basename):
        return os.path.join(ifa_wavdir, basename[:4], basename + '.wav')
    
    def utterances(self):
        """yields pairs of textgrids and wavfilenames"""
        for tg in self._textgrids():
            yield (self._wavfile(tg.name), tg)
    
    def _textgrids(self):
        """Generator object for TextGrid objects in specified speakers"""
        for fname in chain.from_iterable(iglob(os.path.join(ifa_tgdir, d, '*.ltg'))
                                         for d in self.speakers):
            namep = re.compile(r'([FM]\d\d\w)%s(?P<name>[\w-]*?).ltg$' % os.sep)
            name = namep.search(fname).group('name')
            tg = TextGrid(name=name)
            fid = open(fname,'r')
            tg.read(fid)
            fid.close()
            if self.convert_phones:
                it = tg['phone alignment']
                for i in range(len(it)):
                    it[i].mark = self.phone_convert_func(it[i].mark)
            #print tg.name
            yield tg
    
    def iter_phones(self, merge=False):
        for it in imap(lambda x:x['phone alignment'],
                       self._textgrids()):
            for interval in it:
                mark = re.sub(r'[\^\d]+$','',interval.mark)
                # convert mark
                try:
                    mark = cgn_to_sampa(mark)
                    if merge:
                        mark = sampa_merge(mark)
                except:
                    # not a phonemic marker
                    continue
                yield mark