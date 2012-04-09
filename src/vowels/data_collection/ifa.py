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

interface to ifa textgrid files.

only using FR (fixed text retold), VI (variable informal story), VR (variable story retold)


'''
from glob import iglob
import os
import re

from itertools import chain, imap
from bidict import bidict

from ..config.paths import cfg_ifadir
from .textgrid import TextGrid

ifa_tgdir = os.path.join(cfg_ifadir, 'textgrids')

class IFA_TextGrids(object):
    """ interface to ifa's textgrids"""
    def __init__(self,
                 speakers=None):
        valid_speakers = [d for d in os.listdir(ifa_tgdir)
                         if os.path.isdir(os.path.join(ifa_tgdir, d))]
        
        if speakers is None:
            self.speakers = valid_speakers
        elif all(x in valid_speakers for x in speakers):
            self.speakers = speakers
        else:
            raise ValueError, 'Invalid speaker choice. Valid choices are %s' % ', '.join(valid_speakers)
    
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
                    mark = sampa_to_cgn[:mark]
                except:
                    # not a phonemic marker
                    continue
                yield mark

sampa_to_cgn = bidict({
' ' : ' ',                       
'p' : 'p',
'b' : 'b',
't' : 't',
'd' : 'd',
'k' : 'k',
'g' : 'g',
'f' : 'f',
'v' : 'v',
's' : 's',
'z' : 'z',
'S' : 'S',
'Z' : 'Z',
'x' : 'x',
'G' : 'G',
'h' : 'h',
'N' : 'N',
'm' : 'm',
'n' : 'n',
'nj' : 'J',
'l' : 'l',
'r' : 'r',
'w' : 'w',
'j' : 'j',
'I' : 'I',
'E' : 'E',
'A' : 'A',
'O' : 'O',
'}' : 'Y',
'i:' : 'i',
'y:' : 'y',
'e:' : 'e',
'|:' : '2',
'a:' : 'a',
'o:' : 'o',
'u:' : 'u',
'@' : '@',
'EI' : 'E+',
'/I' : '9+',
'Au' : 'O+',
'E:' : 'E:',
'/:' : '9:',
'Q:' : 'O:',
'E~' : 'E~',
'A~' : 'A~',
'O~' : 'O~',
'Y~' : 'Y~'
})















