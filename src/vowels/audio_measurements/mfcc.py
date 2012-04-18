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

import numpy as np

import scikits.audiolab
import scikits.samplerate

from ..config.paths import cfg_dumpdir, cfg_ifadir
from ..data_collection import ifa
from ..util.transcript_formats import vowels_sampa, cgn_to_sampa
from ..util.decorators import memoize, instance_memoize
from .spectral import MFCC

class MFCCError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return self.value

class MFCCMeasure(object):
    def __init__(self,
                 nframes=45,
                 spectral_front_end=None,
                 init_alloc=10000,
                 db_name=None,
                 force_rebuild=False,
                 verbose=True
                 ):
        if spectral_front_end is None:
            self._spectral_front_end = MFCC()
        else:
            self._spectral_front_end = spectral_front_end
        self._nframes=nframes
        self._init_alloc = init_alloc
        self.verbose=verbose
        self.speakers = ifa.valid_speakers()
        self.females = ifa.female_speakers()
        self.males = ifa.male_speakers()
        self.vowels = vowels_sampa
        if db_name is None:
            hex = hashlib.sha224(str(self._nframes) + '_'.join(map(str, self._spectral_front_end.config().values()))).hexdigest()
            self._db_name = os.path.join(cfg_dumpdir, 'mfcc_db_%s' % hex)
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
        self._nobs = dict((s, 
                           dict((v, 0)
                                for v in self.vowels))
                            for s in self.speakers)
        db = shelve.open(self._db_name)
        for speaker in self.speakers:
            for vowel in self.vowels:
                self._nobs[speaker][vowel] = db[speaker][vowel].shape[0]
        db.close()        
        if self.verbose:
            print 'done.'
        
    def _read_wav(self, filename):
        """returns mfcc matrix of the specified audio file
        """
        wave, filefs, enc = scikits.audiolab.wavread(filename)
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

    def population_size(self, speaker, vowel):
        return self._nobs[speaker][vowel]
    
    def _build_db(self):
        if self.verbose:
            print 'building database...'
        corpus = ifa.IFA()
        
        result = dict((s,
                       dict((v,
                             np.empty((self._init_alloc, self._spectral_front_end.ncep * self._nframes)))
                             for v in self.vowels))
                        for s in self.speakers)
        nobs = dict((s,
                     dict((v, 0)
                          for v in self.vowels))
                     for s in self.speakers)
        
        for tg in corpus.iter_textgrids():
            basename = tg.name
            speaker = basename[:4]
            wavname = os.path.join(cfg_ifadir, 'wavs', speaker, basename+'.wav')
            for phone_interval in tg['phone alignment']:
                mark = re.sub(r'[\^\d]+$','',phone_interval.mark)
                try:
                    mark = cgn_to_sampa(mark)
                except:
                    continue
                if mark in vowels_sampa:
                    interval = (phone_interval.xmin, phone_interval.xmax)
                    if self.verbose:
                        print '%s - %s' % (basename, mark)
                        try:
                            vector = self._get_stack_at_interval(wavname, interval)
                        except MFCCError:
                            continue
                        
                    result[speaker][mark][nobs[speaker][mark]] = vector
                    nobs[speaker][mark] += 1
                    
        if self.verbose:
            print 'building database...done.'
        # resize the matrices
        if self.verbose:
            print 'resizing mfcc arrays...'
        for speaker in result:
            for vowel in result[speaker]:
                result[speaker][vowel].resize((nobs[speaker][vowel], self._spectral_front_end.ncep * self._nframes))
        if self.verbose:
            print 'done.'
                
        # save the result in the database
        if self.verbose:
            print 'saving results...'
        db = shelve.open(self._db_name)
        for speaker in result:
            db[speaker] = result[speaker]
        db.close()
        if self.verbose:
            print 'done.'
        
        # clear the memo cache
        if self.verbose:
            print 'clearing cache...'
        self._instance_memoize__cache = {}
        if self.verbose:
            print 'done.'
        
    def get_mfcc_frames(self, speaker, vowel):
        db = shelve.open(self._db_name)
        res = db[speaker][vowel]
        db.close()
        return res
    
    def sample(self,
               vowels,
               k=None,
               speakers=None,
               gender=None):
        if not all(v in self.vowels for v in vowels):
            raise ValueError, 'vowels must be subset of [%s]' % ', '.join(self.vowels)
        if speakers is None:
            if gender is None:
                speakers = self.speakers
            elif gender == 'female':
                speakers = self.females
            else:
                speakers = self.males
        elif not all(s in self.speakers for s in speakers):
            raise ValueError, 'speakers must be subset of [%s]' % ', '.join(self.speakers)
        
        nsamples = dict((v,0) for v in vowels)
        for s in speakers:
            for v in vowels:
                nsamples[v] += self.population_size(s, v)
        result = dict((v, np.empty((nsamples[v], 
                                    self._spectral_front_end.ncep * self._nframes)))
                        for v in vowels)
        filled = dict((v, 0) for v in vowels)
        db = shelve.open(self._db_name)
        for s in speakers:
            for v in vowels:
                if self.population_size(s, v) == 0:
                    continue
                data = db[s][v]
                start_idx = filled[v]
                end_idx = start_idx + data.shape[0]
                result[v][start_idx:end_idx, :] = data
                filled[v] += data.shape[0]
        db.close()
        return result
                
                
                
                
                
                
                
                
                
                
            