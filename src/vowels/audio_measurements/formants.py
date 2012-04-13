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
vowels.audio_measurements.formants:

extract formants from sound files
'''
from bisect import bisect_left

import numpy as np

from .praat_interact import run_praat
from ..util.decorators import memoize

@memoize
def get_formants_from_file(filename, maxformant=5500, winlen=0.025, preemph=50):
    res = run_praat('vowels/audio_measurements/extract_formants_single.praat', filename, maxformant, winlen, preemph)
    res = np.array(map(lambda x:map(float,x.rstrip().split('\t')[:4]),res[1:]))
    return res
    
def extract_formants(filename, time, maxformant=5500, winlen=0.025, preemph=50):
    """ extract f1, f2, f3 formants from speech file"""
    formants_array = get_formants_from_file(filename, maxformant=maxformant, winlen=winlen, preemph=preemph)
    # TODO check if this is correct
    return formants_array[bisect_left(formants_array[:,0],time),1:]

