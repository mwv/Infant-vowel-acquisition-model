#!/usr/bin/python
# -*- coding: utf-8 -*-
""" 
"""

from __future__ import division

__author__ = 'Maarten Versteegh'
__date__ = 'Fri Apr  6 01:06:21 2012'

import re
from transcript_formats import disc_to_sampa

pattern1 = re.compile('\t|\s')

def get_caregivers(childes_file, exclude_target_child=True):
    id_found = False
    interactors=[]
    for line in open(childes_file, 'r'):
        line = re.split(pattern1,line.strip())
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
            yield line[0],line[1].strip()

class PhonLookup(object):
    def __init__(self,
                 dict_fid='/Users/maartenversteegh/projects/vowel_model/data/CELEX/DUTCH/DPW/DPW.CD',
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