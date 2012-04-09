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
    return map(lambda x:disc_to_sampa[x], list(trans))

def stripSyllableMarkers(trans):
    return re.sub("-","",trans)

vowels_sampa = set(
                ['i:', # liep
                'y:', # buut
                'e:', # leeg
                '|:', # deuk
                'a:', # laat
                'o:', # boom
                'u:', # boek
                'I',  # lip
                'E',  # leg
                'A',  # lat
                'O',  # bom
                '}',  # put
                '@',  # schwa
                'i::', # analyse
                'y::', # centrifuge
                'E:',  # scene
                '/:',  # freule
                'Q:',  # zone
                'EI',  # wijs
                '/I',  # huis
                'Au',  # koud
                '{', # trap (eng)
                '6', # open schwa
                '3', # nurse (eng)
                '2', # deux (fr)
                '9', # neuf (fr)
                '&', # open front rounded
                'U', # foot (eng)
                'V', # strut (eng)
                'Y']) # huebsch (de)

disc_to_sampa = bidict({
    ' ':' ',
    'p':'p',
    'b':'b',
    't':'t',
    'd':'d',
    'k':'k',
    'g':'g',
    'N':'N',
    'm':'m',
    'n':'n',
    'l':'l',
    'r':'r',
    'f':'f',
    'v':'v',
    'T':'T',
    'D':'D',
    's':'s',
    'z':'z',
    'S':'S',
    'Z':'Z',
    'j':'j',
    'x':'x',
    'G':'G',
    'h':'h',
    'w':'w',
    '+':'pf',
    '=':'ts',
    'J':'tS',
    '_':'dZ',
    'C':'N,',
    'F':'m,',
    'H':'n,',
    'P':'l,',
    'R':'r*',
    'i':'i:',
    '!':'i::',
    '#':'A:',
    'a':'a:',
    '$':'0:',
    'u':'u:',
    '3':'3:',
    'y':'y:',
    '(':'y::',
    ')':'E:',
    '*':'/:',
    '<':'Q:',
    'e':'e:',
    '|':'|:',
    'o':'o:',
    '1':'eI',
    '2':'aI',
    '4':'OI',
    '5':'@U',
    '6':'aU',
    '7':'I@',
    '8':'E@',
    '9':'U@',
    'K':'EI',
    'L':'/I',
    'M':'Au',
    'W':'ai',
    'B':'au',
    'X':'Oy',
    'I':'I',
    'Y':'Y',
    'E':'E',
    '/':'/',
    '{':'{',
    '&':'{',
    'A':'A',
    'Q':'Q',
    'V':'V',
    'O':'O',
    'U':'U',
    '}':'}',
    '@':'@',
    '^':'/~:',
    'c':'{~',
    'q':'A~:',
    '0':'{~:',
    '-':'-',
    "'":"\"",
    "\"":'%',
    '.':'.'})