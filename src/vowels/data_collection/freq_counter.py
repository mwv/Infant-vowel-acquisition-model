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

""" 
"""

from __future__ import division

__author__ = 'Maarten Versteegh'
__date__ = 'Fri Apr  6 01:06:21 2012'

from collections import Counter

from .childes import ChildesCorpus
from .ifa import IFA_TextGrids

def count_childes_vowels(phone_dict, corpora=None):
    """count phones in specified childes corpora"""
    pl = phone_dict
    cc = ChildesCorpus(corpora=corpora)
    
    word_token_cntr = 0
    word_unrecognized_cntr = 0
    phon_freqs = {}
    for word in cc:
        word_token_cntr += 1
        phon_list = pl[word]
        if phon_list:
            for phon in phon_list:
                try:
                    phon_freqs[phon] += 1
                except:
                    phon_freqs[phon] = 1
        else:
            word_unrecognized_cntr += 1
    n_phon_tokens_total = sum(phon_freqs.values())
    vowel_freqs = {}
    for vowel in vowels_sampa:
        try:
            vowel_freqs[vowel] = phon_freqs[vowel]
        except:
            vowel_freqs[vowel] = 0
    n_vowel_tokens_total = sum(vowel_freqs.values())
    return dict(n_word_tokens = word_token_cntr,
                n_unrecognized_word_tokens = word_unrecognized_cntr,
                n_phon_tokens_total = n_phon_tokens_total,
                n_vowel_tokens_total = n_vowel_tokens_total,
                vowel_freqs = vowel_freqs)

def count_ifa_vowels(speakers=None):
    ifatg = IFA_TextGrids(speakers=speakers)
    phon_freqs = Counter(ifatg.iter_phones())
    n_phon_tokens_total = sum(phon_freqs.values())
    vowel_freqs = {}
    for vowel in vowels_sampa:
        try:
            vowel_freqs[vowel] = phon_freqs[vowel]
        except:
            vowel_freqs[vowel] = 0
    n_vowel_tokens_total = sum(vowel_freqs.values())
    return dict(n_phon_tokens_total = n_phon_tokens_total,
                n_vowel_tokens_total = n_vowel_tokens_total,
                vowel_freqs = vowel_freqs)
    
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
                'Y', # huebsch (de)
                'E~', # vacc_in_ (fr)
                'A~', # croiss_ant_ (fr)
                'O~', # c_o_nge (fr)
                'Y~']) # parf_um_ (fr)

 

