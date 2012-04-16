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
import cPickle

from .praat_interact import run_praat
from ..util.decorators import memoize
from ..data_collection import ifa
from ..config.paths import cfg_ifadir, cfg_dumpdir
from ..data_collection.freq_counter import vowels_sampa
from ..util.standards import cgn_to_sampa

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
def _forms_from(filename, maxformant=5500, winlen=0.025, preemph=50):
    """extract formant table from file using praat
    returns and ndarray laid out as:
    [time f1 f2 f3
    ...]
    
    formants that praat returns as undefined are represented as nans
    
    this function is memoized to minimize the number of calls to praat
    """
    res = run_praat('vowels/audio_measurements/extract_formants_single.praat', filename, maxformant, winlen, preemph)
    return np.array(map(lambda x:map(_float,x.rstrip().split('\t')[:4]), res.split('\n')[1:-1]))

    
def _extr_forms_at(filename, time, maxformant=5500, winlen=0.025, preemph=50):
    """ extract f1, f2, f3 formants from speech file at time t"""
    formants_array = _forms_from(filename, maxformant=maxformant, winlen=winlen, preemph=preemph)
    try:
        res = formants_array[bisect_left(formants_array[:,0],time),1:]
    except IndexError:
        raise ValueError, 'time out of range of filelength'
    if any(np.isnan(res)):
        raise FormantError, 'undefined formant found'
    return res
    
def measure_ifa_formants(edge_margin=0.05, init_alloc=10000, verbose=True, dump=True, speakers=None):
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
    
    Arguments:
    - edge_margin : percentage of total duration of vowel to take boundary measurements
                   'begin' will be taken at xmin + duration*edge_margin
                   'end' will be taken at xmax - duration*edge_margin
    - init_alloc : initial allocation for number of vowels to be expected. 
                   make sure this number is set high enough (depends on corpus)
    
    """
    dumpfile = os.path.join(cfg_dumpdir, 'ifa_formants_%.3f_%s.pkl' % (edge_margin, '_'.join(sorted(speakers)) if speakers else ''))
    if os.path.exists(dumpfile):
        if verbose:
            print 'Loading formants from file.'
        fid = open(dumpfile, 'rb')
        result = cPickle.load(fid)
        fid.close()
        return result
    
    corpus = ifa.IFA(speakers=speakers)
    result = dict((x, np.empty((init_alloc, 10))) for x in vowels_sampa)
    nobs = dict((x, 0) for x in vowels_sampa) # number of observations per vowel
    undefined = 0
    for tg in corpus.iter_textgrids():
        basename = tg.name
        female = basename[0] == 'F'
        wavname = os.path.join(cfg_ifadir, 'wavs', basename[:4], basename+'.wav')
        # find the vowel intervals 
        for phone_interval in tg['phone alignment']:
            mark = re.sub(r'[\^\d]+$','', phone_interval.mark)
            try:
                mark = cgn_to_sampa(mark)
                
            except:
                continue
            if mark in vowels_sampa:
                xmin = phone_interval.xmin
                xmax = phone_interval.xmax
                # pick 3 points
                delta = (xmax-xmin)
                begin = xmin + delta*edge_margin
                middle = xmin + delta/2
                end = xmax - delta*edge_margin
                if verbose:
                    print '%s - %s (%.3f,%.3f,%.3f)' % (basename, mark, begin, middle, end) 
                try:    
                    begin = _extr_forms_at(wavname, begin, maxformant=(5500 if female else 5000))
                    middle = _extr_forms_at(wavname, middle, maxformant=(5500 if female else 5000))
                    end = _extr_forms_at(wavname, end, maxformant=(5500 if female else 5000))
                except FormantError:
                    # just skip vowels with undefined formants
                    undefined += 1
                    continue

                vector = np.hstack((begin, middle, end, np.array(delta)))
                result[mark][nobs[mark]] = vector
                nobs[mark] += 1
    if verbose:                    
        print 'undefined formants found: %d' % undefined
    for vowel in result:
        result[vowel].resize((nobs[vowel], 10))
    if verbose:
        print 'Observed phones:'
        for p in nobs:
            print '%s\t%d' % (p, nobs[p])
    
    if dump:
        fid = open(dumpfile, 'wb')
        cPickle.dump(result, fid)
        fid.close()
    
    return result
                
                
                
    
    
    

