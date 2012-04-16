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

import numpy as np

'''
vowels.util.functions:

utility functions

'''

# frequency conversion

def hertz_to_mel(f):
    return 2595 * np.log10(1+f/700)

def mel_to_hertz(m):
    return 700 * (10**(m/2595)-1)

def hertz_to_bark(f):
    return 13 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f/7500)**2)

def _float(s):
    """wrapper for float, but returns np.nan if s cannot be converted
    """
    try:
        return float(s)
    except ValueError:
        return np.nan
