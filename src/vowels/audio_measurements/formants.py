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
import hashlib

import numpy as np
import os
import re
import cPickle
import shelve

from .praat_interact import run_praat
from ..util.decorators import instance_memoize
from ..config.paths import cfg_dumpdir
from ..util.transcript_formats import cgn_to_sampa, vowels_sampa, vowels_sampa_merged, sampa_merge
from ..util.functions import _float, hertz_to_bark, hertz_to_mel

class FormantError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return self.value
    
class FormantsMeasure(object):
    def __init__(self, 
                 corpus,
                 merge_vowels=True,
                 edge_margin=0.05, 
                 init_alloc=10000, 
                 winlen=0.025, 
                 preemph=50, 
                 db_name=None,
                 force_rebuild=False,
                 verbose=True):
        """ Wrapper around formant measurements database 
        
        Arguments:
        - edge_margin: percentage from vowel boundary to measure begin and end slices from
                       default=0.05
        - init_alloc: initial allocation size for measurements. default probably high enough for most
                      default=10000 
        - winlen: window length for formant measurements in seconds
                  default=0.025
        - preemph: preemphasis in hertz
                   default=50
        - db_name: name of custom database to use as backend. if not specified, a file is generated in cfg_dumpdir
        - force_rebuild: forces a rebuild of the database, regardless of whether the database exists
        """
        self.corpus = corpus
        self.merge_vowels = merge_vowels
        self._edge_margin = edge_margin
        self._init_alloc = init_alloc
        self._winlen = winlen
        self._preemph = preemph
        self.vowels = vowels_sampa
        self.verbose=verbose
        
        if db_name is None:
            hex = hashlib.sha1(str(self.corpus) + 
                               str(self.merge_vowels) +
                               str(self._edge_margin) + 
                               str(self._winlen) + 
                               str(self._preemph)).hexdigest()
            self._db_name = os.path.join(cfg_dumpdir, 'formant_db_%s' % hex)
        else:
            self._db_name = db_name
        if force_rebuild:
            if os.path.exists(self._db_name):
                os.remove(self._db_name)
            self._build_db()        
        if not os.path.exists(self._db_name):
            self._build_db()
        
    def _build_db(self):
        if self.verbose:
            print 'building database...'
        corpus = self.corpus

        result = dict((v, 
                       np.empty((self._init_alloc,10))) 
                       for v in self.vowels) 
        nobs = dict((v, 0) 
                    for v in self.vowels)
        for (wavname, tg) in corpus.utterances():
            basename = tg.name
            female = basename[0] == 'F'
            # find the vowel intervals
            for phone_interval in tg['phone alignment']:
                mark = re.sub(r'[\^\d]+$','', phone_interval.mark)
                
                try:
                    mark = cgn_to_sampa(mark)
                    if self.merge_vowels:
                        mark = sampa_merge(mark)
                except:
                    continue
                if mark in self.vowels:
                    xmin = phone_interval.xmin
                    xmax = phone_interval.xmax
                    # pick 3 points
                    delta = (xmax-xmin)
                    begin = xmin + delta*self._edge_margin
                    middle = xmin + delta/2
                    end = xmax - delta*self._edge_margin
                    if self.verbose:
                        print '%s - %s' % (basename, mark)
                    try:    
                        begin = self._extr_forms_at(wavname, begin, maxformant=(5500 if female else 5000))
                        middle = self._extr_forms_at(wavname, middle, maxformant=(5500 if female else 5000))
                        end = self._extr_forms_at(wavname, end, maxformant=(5500 if female else 5000))
                    except FormantError:
                        # just skip vowels with undefined formants
                        continue
    
                    vector = np.hstack((begin, middle, end, np.array(delta)))
                    result[mark][nobs[mark]] = vector
                    nobs[mark] += 1
        # resize the matrices
        for vowel in result:
            result[vowel].resize((nobs[vowel],10))
        # save the results in the database
        db =  shelve.open(self._db_name)                
        for vowel in result:
            self._db[vowel] = result[vowel]
        db.close()
        self._instance_memoize__cache = {}
        
                     
    def population_size(self,
                        vowel):
        return self._db[vowel].shape[0]
    
    def formants(self,
                 vowel,
                 ):
        """returns all measurements for specified vowel
        
        measurements are laid out in a 10-dimensional array as follows:
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
        db = shelve.open(self._db_name)
        res = db[vowel]
        db.close()
        return db
    
    @instance_memoize
    def _forms_from(self, filename, maxformant=5500):
        """extract formant table from file using praat
        returns and ndarray laid out as:
        [time f1 f2 f3
        ...]
        
        formants that praat returns as undefined are represented as nans
        
        this function is memoized to minimize the number of calls to praat
        """
        res = run_praat('vowels/audio_measurements/extract_formants_single.praat', 
                        filename, 
                        maxformant, 
                        self._winlen, 
                        self._preemph)
        return np.array(map(lambda x:map(_float,x.rstrip().split('\t')[:4]), res.split('\n')[1:-1]))
    
    def _extr_forms_at(self, filename, time, maxformant=5500):
        """ extract f1, f2, f3 formants from speech file at time t"""
        formants_array = self._forms_from(filename, maxformant=maxformant)
        try:
            res = formants_array[bisect_left(formants_array[:,0],time),1:]
        except IndexError:
            raise ValueError, 'time out of range of filelength'
        if any(np.isnan(res)):
            raise FormantError, 'undefined formant found'
        return res
        
    def sample(self, 
               vowels, 
               features=None, # not implemented yet
               k=None,
               scale='hertz', 
               exclude_outliers=False, # not implemented yet
               percentile=99.9): # not implemented yet
        """Chooses k unique measurements for each vowel in vowels for the specified speakers.
        Returns a dict from vowels to measurements
        if k is not specified, the maximum available number of samples is chosen.
        
        Arguments:
        - vowels: list of vowels
        - [k]: sample size. if not specified, maximum sample size is chosen
        - [speakers]: list of speakers, if gender is also specified, this overrrides that
        - [gender]: ('male', 'female'), restriction on speakers. specification of speakers overrides this
        - [scale]: ('bark','mel','hertz') frequency scale used. defaults to 'hertz'
        - [exclude_outliers]: returns only samples in 'percentile' quantile of variance. default False
        - [percentile]: quantile to use in excluding outliers. default 99.9
        """
        if not all(v in self.vowels for v in vowels):
            raise ValueError, 'vowels must be subset of [%s]' % ', '.join(self.vowels)

        
        # figure out how many samples we're gonna see per vowel
        nsamples = dict((v,0) for v in vowels)   
        for v in vowels:
            nsamples[v] += self.population_size(v)
        result = dict((v, np.empty((nsamples[v],10))) for v in vowels)
        filled = dict((v,0) for v in vowels)

        db = shelve.open(self._db_name)
        for n in range(len(vowels)):
            if self.population_size(vowels[n]) == 0:
                continue
            data = db[vowels[n]]
            if scale == 'bark':
                data[:,:10] = hertz_to_bark(data[:,:10])
            elif scale == 'mel':
                data[:,:10] = hertz_to_mel(data[:,:10])
            start_idx = filled[vowels[n]]
            end_idx = start_idx + data.shape[0]
            result[vowels[n]][start_idx:end_idx, :] = data
            filled[vowels[n]] += data.shape[0]
        db.close()
        return result