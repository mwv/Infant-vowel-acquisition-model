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
childes_freq_counter:

'''
from __future__ import division

import vowels.data_collection.freq_counter as freq_counter
from vowels.data_collection.celex import PhonLookup

def run_childes():
    pl = PhonLookup()
    result = freq_counter.count_childes_vowels(pl)
    print '-'*35
    print 'CHILDES:'
    print '-'*35    
    print 'Number of processed word tokens: %d (%d unanalyzed)' % (result['n_word_tokens'], result['n_unrecognized_word_tokens'])
    print 'Total number of phone tokens: %d' % (result['n_phon_tokens_total'])
    print 'Total number of vowel tokens: %d' % (result['n_vowel_tokens_total'])
    print '-'*35
    print 'Vowel\tAbs\tRel'
    for vwl in result['vowel_freqs']:
        count = result['vowel_freqs'][vwl]
        if count > 0:
            print '%s\t%7d\t%.5f' % (vwl, count, count/result['n_vowel_tokens_total'])
    print '-'*35
    
def run_ifa():
    result = freq_counter.count_ifa_vowels()
    print '-'*35
    print 'IFA Corpus:'
    print '-'*35    
    print 'Total number of phone tokens: %d' % (result['n_phon_tokens_total'])
    print 'Total number of vowel tokens: %d' % (result['n_vowel_tokens_total'])
    print '-'*35
    print 'Vowel\tAbs\tRel'
    for vwl in result['vowel_freqs']:
        count = result['vowel_freqs'][vwl]
        if count > 0:
            print '%s\t%7d\t%.5f' % (vwl, count, count/result['n_vowel_tokens_total'])
    print '-'*35
    
if __name__ == '__main__':
    run_childes()
    run_ifa()