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
vowels.speechcorpora.corpus:

abstract base class for speech corpora
# TODO implement

'''

from itertools import chain

class MergedCorpus(object):
    def __init__(self, corpora):
        self.corpora = corpora
        self.name = '-'.join(c.name for c in self.corpora)
        
    def __repr__(self):
        return 'merged_corpus' + '-'.join(str(c) for c in self.corpora)
        
    def utterances(self):
        return chain.from_iterable(c.utterances() 
                                   for c in self.corpora)

