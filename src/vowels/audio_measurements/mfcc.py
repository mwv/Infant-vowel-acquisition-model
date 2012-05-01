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
vowels.audio_measurements.mfcc:

'''

import re
import os
import shelve
import hashlib
import random
import numpy as np

import scikits.audiolab
import scikits.samplerate

from ..config.paths import cfg_dumpdir

from ..util.transcript_formats import vowels_sampa, vowels_sampa_merged
from ..util.decorators import instance_memoize
from .spectral import MFCC

class MFCCError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return self.value

class MFCCMeasure(object):
    def __init__(self,
                 corpus,
                 merge_vowels=True,
                 nframes=23,
                 spectral_front_end=None,
                 init_alloc=200000,
                 db_name=None,
                 force_rebuild=False,
                 verbose=False
                 ):
        if spectral_front_end is None:
            self._spectral_front_end = MFCC()
        else:
            self._spectral_front_end = spectral_front_end
        self.corpus = corpus
        self.merge_vowels=merge_vowels
        self._nframes=nframes
        self._init_alloc = init_alloc
        self.verbose=verbose
        if merge_vowels:
            self.vowels = vowels_sampa_merged
        else:
            self.vowels = vowels_sampa
        if db_name is None:
            hex = hashlib.sha1(str(self._nframes) + 
                               str(self.merge_vowels) +
                               str(self.corpus) +  
                               '_'.join(map(str, self._spectral_front_end.config().values()))).hexdigest()
            self._db_name = os.path.join(cfg_dumpdir, 'mfcc_db_%s' % (hex))
            self._nobs_name = os.path.join(cfg_dumpdir, 'mfcc_nobs_%s' % (hex))
        else:
            self._db_name = db_name
        if force_rebuild:
            if os.path.exists(self._db_name):
                os.remove(self._db_name)
            self._build_db()
        if not os.path.exists(self._db_name):
            self._build_db()
            
        self._make_nobs()
        
    def _make_nobs(self):
        if self.verbose:
            print 'gathering population statistics...',
        # gather population statistics        
        db = shelve.open(self._nobs_name)
        self._nobs = {}
        for vowel in db:
            self._nobs[vowel] = db[vowel]
        db.close()

#        self._nobs = dict((s, 
#                           dict((v, 0)
#                                for v in self.vowels))
#                            for s in self.speakers)
#        db = shelve.open(self._db_name)
#        for speaker in self.speakers:
#            for vowel in self.vowels:
#                self._nobs[speaker][vowel] = db[speaker][vowel].shape[0]
#        db.close()        
        if self.verbose:
            print 'done.'
        
    def _read_wav(self, filename):
        """returns mfcc matrix of the specified audio file
        """
        wave, filefs, _ = scikits.audiolab.wavread(filename)
        if self._spectral_front_end.fs != filefs:
            wave = scikits.samplerate.resample(wave, self._spectral_front_end.fs / filefs, 'sinc_best')
        return wave

    @instance_memoize
    def _get_mfcc(self, filename):
        wave = self._read_wav(filename)
        return self._spectral_front_end.mfcc(wave)
        
    @instance_memoize
    def _get_mel(self, filename):
        wave = self._read_wav(filename)
        return self._spectral_front_end.sig2logspec(wave)
    
    def _get_stack_at_interval(self, filename, interval):
        start, end = interval
        middle = start + (end-start)/2
        mfcc_matrix = self._get_mfcc(filename)
        middle_frame_idx = middle * self._spectral_front_end.frate
        start_idx = middle_frame_idx - (self._nframes//2)
        end_idx = middle_frame_idx + (self._nframes//2)
        if start_idx < 0 or end_idx > mfcc_matrix.shape[0]:
            raise MFCCError, 'frame selection out of bounds'
        return np.resize(mfcc_matrix[start_idx:end_idx,:], (self._spectral_front_end.ncep * self._nframes,))        

    def population_size(self, vowel):
        return self._nobs[vowel]
    
    def _build_db(self):
        if self.verbose:
            print 'building database...'
        corpus = self.corpus
        
        result = dict((v,
                       np.empty((self._init_alloc, self._spectral_front_end.ncep * self._nframes)))
                              for v in self.vowels)

        nobs = dict((v, 0)
                    for v in self.vowels)
        for (wavname, tg) in corpus.utterances():
        #for tg in corpus.iter_textgrids():
            basename = tg.name
#            speaker = basename[:4]
#            wavname = corpus.wavfile(basename)
            #wavname = os.path.join(cfg_ifadir, 'wavs', speaker, basename+'.wav')
            for phone_interval in tg['phone alignment']:
                mark = phone_interval.mark
                #mark = re.sub(r'[\^\d]+$','',phone_interval.mark)
#                try:
#                    mark = cgn_to_sampa(mark)
#                    if self.merge_vowels:
#                        mark = sampa_merge(mark)
#                except:
#                    continue
                if mark in self.vowels:
                    interval = (phone_interval.xmin, phone_interval.xmax)
                    if self.verbose:
                        print '%s - %s' % (basename, mark)
                        try:
                            vector = self._get_stack_at_interval(wavname, interval)
                        except MFCCError:
                            continue
                        
                    result[mark][nobs[mark]] = vector
                    nobs[mark] += 1
                    
        if self.verbose:
            print 'building database...done.'
        # resize the matrices
        if self.verbose:
            print 'resizing mfcc arrays...'
        for vowel in result:
                result[vowel].resize((nobs[vowel], self._spectral_front_end.ncep * self._nframes))
        if self.verbose:
            print 'done.'
                
        # save the result in the database
        if self.verbose:
            print 'saving results...'
        db = shelve.open(self._db_name)
        for vowel in result:
            db[vowel] = result[vowel]
        db.close()
        
        db = shelve.open(self._nobs_name)
        for vowel in nobs:
            db[vowel] = nobs[vowel]
        db.close()
        if self.verbose:
            print 'done.'
            
        # clear the memo cache
        if self.verbose:
            print 'clearing cache...'
        self._instance_memoize__cache = {}
        if self.verbose:
            print 'done.'
        
    def get_mfcc_frames(self, vowel):
        db = shelve.open(self._db_name)
        res = db[vowel]
        db.close()
        return res
    
    def sample(self,
               vowels=None,
               k=None):
        if vowels is None:
            vowels = self._nobs.keys()
        elif not all(v in self.vowels for v in vowels):
            raise ValueError, 'vowels must be subset of [%s]' % ', '.join(self.vowels)

        
        nsamples = dict((v,0) for v in vowels)
        for v in vowels:
            nsamples[v] += self.population_size(v)
            
        min_nsamples = min(nsamples.values())
        if k is None or k > min_nsamples:
            k = min_nsamples
        result = dict((v, np.empty((k, 
                                    self._spectral_front_end.ncep * self._nframes)))
                        for v in vowels)
        filled = dict((v, 0) for v in vowels)
        db = shelve.open(self._db_name)
        for v in vowels:
            if self.population_size(v) == 0:
                continue
            data = db[v]
            sample_idcs = random.sample(range(data.shape[0]), k)
            data = data[sample_idcs,:]
            start_idx = filled[v]
            end_idx = start_idx + data.shape[0]
            result[v][start_idx:end_idx, :] = data
            filled[v] += data.shape[0]
        db.close()
        return result
                
                
                
                
                
                
                
                
                
                
            