#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------
# author:
# Christina Bergmann
# 
# Licensed under GPLv3
# ------------------------------------

from __future__ import division

'''
vowels.audio_measurements.lda:

LDA for dimensionality reduction

'''
import numpy as np

from bidict import bidict
import sklearn.lda as sklda

class LDA(object):
    def __init__(self):
        self.lda = sklda.LDA()
    
    def train(self, samples, labels):
        """
        Train LDA model.
        Parameters
        ----------
        X : ndarray, shape = [n_vowelsamples, n_MFCCsamples, n_MFCCcoefficients]
          Training ndarray, where n_vowelsamples is the number of vowels sampled,
          n_MFCCsamples is the number of MFCCs extracted from one vowel and
          n_MFCCcoefficients is the number of coefficients in each MFCC vector
        y : ndarray, shape = [n_vowelsamples]
          Vowel labels (string or integer)
        Returns
        ----------
        LDAobject
    
        Leaving out check for sample size match between X and y, since LDA does that for us.
        """
        nsamples, nframes, ncepstra = samples.shape
        samples = np.resize(samples,(nsamples, nframes*ncepstra))
        
        # build label map
        label_set = sorted(list(set(labels)))
        self._label_map = bidict(zip(range(len(label_set)), label_set))

        self.lda.fit(samples, map(lambda x:self._label_map[:x], labels))
        return self

    def transform(self, samples):
        nsamples, nframes, ncepstra = samples.shape
        samples = np.resize(nsamples, nframes*ncepstra)
        
        return self.lda.transform(samples)
    
            

