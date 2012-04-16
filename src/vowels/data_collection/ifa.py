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
from bidict import bidict

from ..config.paths import cfg_ifadir
from .textgrid import TextGrid
from ..util.transcript_formats import cgn_to_sampa

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
                 speakers=None):
        if speakers is None:
            self.speakers = valid_speakers()
        elif all(x in valid_speakers() for x in speakers):
            self.speakers = speakers
        else:
            raise ValueError, 'Invalid speaker choice. Valid choices are %s' % ', '.join(self.valid_speakers())
        

    
    def iter_textgrids(self):
        """Generator object for TextGrid objects in specified speakers"""
        for fname in chain.from_iterable(iglob(os.path.join(ifa_tgdir, d, '*.ltg'))
                                         for d in self.speakers):
            namep = re.compile(r'([FM]\d\d\w)%s(?P<name>[\w-]*?).ltg$' % os.sep)
            name = namep.search(fname).group('name')
            tg = TextGrid(name=name)
            fid = open(fname,'r')
            tg.read(fid)
            fid.close()
            yield tg
    
    def iter_phones(self):
        for it in imap(lambda x:x['phone alignment'],
                       self.iter_textgrids()):
            for interval in it:
                mark = re.sub(r'[\^\d]+$','',interval.mark)
                # convert mark
                try:
                    mark = cgn_to_sampa(mark)
                except:
                    # not a phonemic marker
                    continue
                yield mark