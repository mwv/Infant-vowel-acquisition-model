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

from __future__ import division

'''
vowels.audio_measurements.formants:

extract formants from sound files
'''
from bisect import bisect_left

import numpy as np
import os
import re

from .praat_interact import run_praat
from ..util.decorators import memoize
from ..data_collection import ifa
from ..config.paths import cfg_ifadir
from ..data_collection.freq_counter import vowels_sampa

class FormantError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return self.value

def _float(s):
    """wrapper for float, but returns np.nan if s cannot be converted
    """
    try:
        return float(s)
    except ValueError:
        return np.nan

@memoize
def _get_formants_from_file(filename, maxformant=5500, winlen=0.025, preemph=50):
    res = run_praat('vowels/audio_measurements/extract_formants_single.praat', filename, maxformant, winlen, preemph)
    return np.array(map(lambda x:map(_float,x.rstrip().split('\t')[:4]), res.split('\n')[1:-1]))

    
def _extract_formants(filename, time, maxformant=5500, winlen=0.025, preemph=50):
    """ extract f1, f2, f3 formants from speech file"""
    formants_array = _get_formants_from_file(filename, maxformant=maxformant, winlen=winlen, preemph=preemph)
    try:
        res = formants_array[bisect_left(formants_array[:,0],time),1:]
    except IndexError:
        raise ValueError, 'time out of range of filelength'
    if any(np.isnan(res)):
        raise FormantError, 'undefined formant found'
    return res
    
def measure_ifa_formants(percentile=0.05):
    """returns a dict from vowels to list of measurements
    
    measurements are laid out in an 10-dimensional array as follows:
    0 : F1_begin 
    1 : F2_begin
    2 : F3_begin
    3 : F1_middle
    4 : F2_middle
    5 : F3_middle
    6 : F1_end
    7 : F2_end
    8 : F3_end
    9 : duration
    
    """
    corpus = ifa.IFA()
    result = {} # dict from vowels to list of measurements
    undefined_formants_found = 0
    for tg in corpus.iter_textgrids():
        basename = tg.name
        female = basename[0] == 'F'
        wavname = os.path.join(cfg_ifadir, 'wavs', basename[:4], basename+'.wav')
        # find the vowel intervals 
        for phone_interval in tg['phone alignment']:
            mark = re.sub(r'[\^\d]+$','', phone_interval.mark)
            try:
                mark = ifa.sampa_to_cgn[:mark]
            except:
                continue
            if mark in vowels_sampa:
                xmin = phone_interval.xmin
                xmax = phone_interval.xmax
                # pick 3 points
                delta = (xmax-xmin)
                begin = xmin + delta*percentile
                middle = xmin + delta/2
                end = xmax - delta*percentile
                #print '%s - %s (%.3f, %.3f)' % (basename, mark, xmin, xmax)
                print '%s - %s (%.3f,%.3f,%.3f)' % (basename, mark, begin, middle, end) 
                try:    
                    begin = _extract_formants(wavname, begin, maxformant=(5500 if female else 5000))
                    middle = _extract_formants(wavname, middle, maxformant=(5500 if female else 5000))
                    end = _extract_formants(wavname, end, maxformant=(5500 if female else 5000))
                except FormantError:
                    # just skip vowels with undefined formants
                    undefined_formants_found += 1
                    continue

                vector = np.hstack((begin, middle, end, np.array(delta)))
                try:
                    result[mark].append(vector)
                except:
                    result[mark] = [np.hstack(vector)]
                    
    print 'undefined formants found: %d' % undefined_formants_found
    return result
                
                
                
    
    
    

