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
formant_extractor:

'''
import os

# set up matplotlib for tipa
import matplotlib
matplotlib.use('Agg')
#from matplotlib import rc
#rc('font',**{'family':'serif'})
#rc('text',**{'usetex':True, 'latex.preamble':['\usepackage{tipa}']})

import matplotlib.pyplot as plt

import random

import numpy as np
import scipy.stats as stats


import vowels.speechcorpora.ifa as ifa
import vowels.speechcorpora.cgn as cgn
from vowels.speechcorpora.corpus import MergedCorpus
import vowels.config.paths as paths
from vowels.util.transcript_formats import vowels_sampa, sampa_to_unicode
from vowels.config.paths import cfg_figdir
import vowels.audio_measurements.formants as formants

def plot_vowels(outfiletag, 
                #nsamples=1000, 
                vowels=None, 
                scale='log', 
                percentile=99, 
                speakers=None, 
                verbose=True,
                minsamples=1):
    if vowels is None:
        vowels = list(vowels_sampa)
        
    allowed_scales = ['log', 'linear', 'bark','mel']
    if not scale in allowed_scales:
        raise ValueError, 'scale must be one of [%s]' % ', '.join(allowed_scales)

    cgn_corpus = cgn.CGN()
    ifa_corpus = ifa.IFA()
    corpus = MergedCorpus([ifa_corpus, cgn_corpus])
    #corpus = MergedCorpus([ifa_corpus])
    fm = formants.FormantsMeasure(corpus)
    fm.info()
    forms = fm.sample(vowels, equal_samples=True,scale='hertz')
    
    vowels = filter(lambda x:forms[x].shape[0] >= minsamples, vowels)    
    # plot the static F1, F2
    fig = plt.figure()
    colors = ['b','g','r','c','m','y','k']
    xs = []
    ys = []
    min_x=np.inf
    max_x=np.NINF
    min_y=np.inf 
    max_y=np.NINF
    means = {}
    for n in range(len(vowels)):
        f2 = forms[vowels[n]][:,4]
        f1 = forms[vowels[n]][:,3]
        
        # filter out values outside specified percentile score
        f2_bottom_perc = stats.scoreatpercentile(f2, 100-percentile)
        f2_top_perc = stats.scoreatpercentile(f2, percentile)
        f1_bottom_perc = stats.scoreatpercentile(f1, 100-percentile)
        f1_top_perc = stats.scoreatpercentile(f1, percentile)
        
        f2_mask = np.logical_and(f2 > f2_bottom_perc, f2 < f2_top_perc)
        f1_mask = np.logical_and(f1 > f1_bottom_perc, f1 < f1_top_perc)
        
        mask = np.logical_and(f2_mask, f1_mask)
        
        f2 = f2[mask]
        f1 = f1[mask]
      
        xs.append(f2)
        ys.append(f1)
        min_f2 = np.min(f2)
        max_f2 = np.max(f2)
        min_f1 = np.min(f1)
        max_f1 = np.max(f1)
        if min_f2 < min_x:
            min_x = min_f2
        if max_f2 > max_x:
            max_x = max_f2
        if min_f1 < min_y:
            min_y = min_f1
        if max_f1 > max_y:
            max_y = max_f1
            
        f1_mean = np.mean(f1, axis=0)
        f2_mean = np.mean(f2, axis=0)
        means[vowels[n]] = (f1_mean, f2_mean)
        if verbose:
            print 'vowel: %s\tobserved: %d\t sample mean (f1,f2): (%.3f,%.3f)' % (vowels[n], f2.shape[0], f1_mean, f2_mean) 
    
    #nsamples = min(map(lambda x:x.shape[0], xs))
    
    for n in range(len(vowels)):
        #sample_ids = random.sample(range(xs[n].shape[0]), nsamples)
#        xs_loc = xs[n][sample_ids,:]
#        ys_loc = ys[n][sample_ids,:]
        xs_loc = xs[n]
        ys_loc = ys[n]
        #print 'samplesize for %s: %d' % (vowels[n],xs_loc.shape[0])
        plt.scatter(xs_loc, ys_loc,
                    color=colors[n % len(colors)], 
                    label=ur'$\mathrm{%s}$' % sampa_to_unicode(vowels[n]), 
                    alpha=0.2)
    for n in range(len(vowels)):
        
        plt.scatter(means[vowels[n]][1], means[vowels[n]][0], 
                    s=80,
                    color='k',
                    marker=ur'$\mathrm{%s}$' % sampa_to_unicode(vowels[n]))
        
    #print means
        
    plt.xlim(max_x+100, min_x-100)
    plt.ylim(max_y+100, min_y-100)
    plt.xlabel(r'F2')
    plt.ylabel(r'F1')
    if scale == 'log':
        plt.xscale('log')
        plt.yscale('log')
    plt.legend(loc='lower left')
    plt.savefig(os.path.join(cfg_figdir, outfiletag+'.png'))
    
if __name__ == '__main__':
    plot_vowels('vowel_square_formants',
                scale='linear',
                vowels=['I', # lip
                        'e:', # leeg
                        '|:', # deuk 
                        '}'], # put
                percentile=95,
                speakers=None)

                

