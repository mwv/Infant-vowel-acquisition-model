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
manifold_comparison:


comparison of embeddings of the mfcc and formant datasets
'''

import os
from time import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import vowels.speechcorpora.ifa as ifa
import vowels.speechcorpora.cgn as cgn
from vowels.speechcorpora.corpus import MergedCorpus

from sklearn.utils.fixes import qr_economic
from sklearn import manifold, decomposition, lda, preprocessing

import vowels.audio_measurements.formants as formants
import vowels.audio_measurements.mfcc as mfcc

from vowels.config.paths import cfg_figdir

from vowels.util.transcript_formats import vowels_sampa, sampa_to_unicode

vowels = sorted(['e:','I','}','|:'])  
#vowels = sorted(['E','I','}'])

colors = ['b','g','r','c','m','y','k']
#colors=['b','g','r']

def plot_embedding(X, y, outtag, title=None, alpha=0.6):
    xmin, xmax = np.min(X,0), np.max(X,0)
    X = (X-xmin)/(xmax-xmin)
    
    fig = plt.figure()
    for i in range(len(y)):
        plt.text(X[i,0],X[i,1], 
                 ur'$\mathrm{%s}$' % sampa_to_unicode(vowels[int(y[i])]), 
                 color=colors[int(y[i])], 
                 alpha=alpha)
#        plt.scatter(X[i,0],X[i,1], color=colors[int(y[i])], alpha=alpha)   
    plt.xticks([]), plt.yticks([])
    if not title is None:
        plt.title(title)
    plt.xlim(np.min(X,0)[0]-0.01, np.max(X,0)[0]+0.01)
    plt.ylim(np.min(X,0)[1]-0.01, np.max(X,0)[1]+0.01)
    plt.savefig(os.path.join(cfg_figdir, outtag))

def get_formant_data(verbose=True):
    cgn_corpus=cgn.CGN()
    ifa_corpus =ifa.IFA()
    corpus= MergedCorpus([ifa_corpus, cgn_corpus])
    fm = formants.FormantsMeasure(corpus)
    data = fm.sample(vowels, equal_samples=True,scale='hertz')
    if verbose:
        print 'vowel:\tsamples:'
        for v in data:
            print '%6s\t%d' % (v, data[v].shape[0])
    X = np.vstack(data[x] for x in sorted(data.keys()))
    X_scaled = preprocessing.scale(X)
    y = np.hstack(np.ones(data[sorted(data.keys())[x]].shape[0],) * x
                  for x in range(len(data)))
    
    return X_scaled,y
 
def get_mfcc_data():
    mfccm = mfcc.MFCCMeasure()
    data = mfccm.sample(vowels)
    X = np.vstack(data[x] for x in sorted(data.keys()))
    X_scaled = preprocessing.scale(X)
    y = np.hstack(np.ones(data[sorted(data.keys())[x]].shape[0],) * x
                  for x in range(len(data)))
    return X_scaled,y    

def random_projection(X,y, tag):
    _, n_features = X.shape
    rng = np.random.RandomState(13)
    Q, _ = qr_economic(rng.normal(size=(n_features,2)))
    X_proj = np.dot(Q.T, X.T).T
    plot_embedding(X_proj, y, 'random_%s' % tag, title='Random Projection: %s' % tag)
    
def pca(X,y, tag, n_components=2):
    X_pca = decomposition.RandomizedPCA(n_components=n_components).fit_transform(X, y)
    plot_embedding(X_pca, y, 'pca_%s' % tag, title='Principal Components Projection: %s' % tag)
    
def lda_proj(X, y, tag, n_components=2):
    X2 = X.copy()
    X2.flat[::X.shape[1] + 1] += 0.01
    X_lda = lda.LDA(n_components=n_components).fit_transform(X2, y)
    plot_embedding(X_lda, y, 'lda_%s' % tag, title='Linear Discriminant Projection: %s' % tag)
    
def isomap(X, y, tag, n_neighbors=50):    
    #X_iso = manifold.Isomap(n_neighbors=n_neighbors, out_dim=2).fit_transform(X, y)
    X_iso = manifold.Isomap(n_neighbors=n_neighbors).fit_transform(X, y)
    plot_embedding(X_iso, y, 'isomap_%s' % tag, title='Isomap Projection: %s' % tag)
    
def lle(X, y, tag, n_neighbors=50):
    #clf = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, out_dim=2, method='standard')
    clf = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, method='standard')        
    X_lle = clf.fit_transform(X, y)
    plot_embedding(X_lle, y, 'lle_%s' % tag, title='Locally Linear Embedding: %s' % tag)
    
def mlle(X, y, tag, n_neighbors=50):
    #clf = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, out_dim=2, method='modified')
    clf = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, method='modified')
    X_mlle = clf.fit_transform(X, y)
    plot_embedding(X_mlle,y, 'mlle_%s' % tag, title='Modified Locally Linear Embedding: %s' % tag)
    
def hlle(X, y, tag, n_neighbors=50):
    #clf = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, out_dim=2, method='hessian')
    clf = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, method='hessian')
    X_hlle = clf.fit_transform(X, y)
    plot_embedding(X_hlle, y, 'hlle_%s' % tag, title='Hessian Eigenmap Projection: %s' % tag)
    
if __name__ == '__main__':
    globaltag = 'sq'
#    print '-'*35
#    print 'MFCC'
#    print '-'*35
#    print 'gathering mfcc data...',
#    X,y = get_mfcc_data()
#    print 'done.'
    
#    t0 = time()
#    print 'random projection...',
#    random_projection(X,y, 'MFCC_all')
#    print 'done. time %.3fs' % (time() - t0)
    
#    t0 = time()
#    print 'pca projection...',
#    pca(X,y, 'MFCC_all')
#    print 'done. time %.3fs' % (time() - t0)
    
#    t0 =time()
#    print 'lda projection...',
#    lda_proj(X,y, 'MFCC_all')
#    print 'done. time %.3fs' % (time() - t0)
#    
#    t0 = time()
#    print 'isomap...',
#    isomap(X,y, 'MFCC_all')
#    print 'done. time %.3fs' % (time() - t0)
    
#    t0 = time()
#    print 'locally linear embedding...',
#    lle(X,y,'MFCC_all')
#    print 'done. time %.3fs' % (time() - t0)
    
#    t0 = time()
#    print 'modified locally linear embedding...',    
#    mlle(X,y,'MFCC_all')
#    print 'done. time %.3fs' % (time() - t0)
#    
#    t0 = time()
#    print 'hessian eigenmapping...',    
#    try:
#        hlle(X,y, 'MFCC_all')
#        print 'done. time %.3fs' % (time() - t0)
#    except:
#        print 'failed.'
#
    print '-'*35
    print 'FORMANTS'
    print '-'*35        
    print 'gathering formant data...',
    X,y = get_formant_data()
    print 'done.'
    
    t0 = time()
    print 'random projection...',
    random_projection(X,y, 'Formant_%s' % globaltag)
    print 'done. time %.3fs' % (time() - t0)
    
    t0 = time()
    print 'pca projection...',
    pca(X,y, 'Formant_%s' % globaltag)
    print 'done. time %.3fs' % (time() - t0)
    
    t0 =time()
    print 'lda projection...',
    lda_proj(X,y, 'Formant_%s' % globaltag)
    print 'done. time %.3fs' % (time() - t0)
    
    t0 = time()
    print 'isomap...',
    isomap(X,y, 'Formant_%s' % globaltag)
    print 'done. time %.3fs' % (time() - t0)
    
    t0 = time()
    print 'locally linear embedding...',
    lle(X,y,'Formant_%s' % globaltag)
    print 'done. time %.3fs' % (time() - t0)
    
    t0 = time()
    print 'modified locally linear embedding...',    
    mlle(X,y,'Formant_%s' % globaltag)
    print 'done. time %.3fs' % (time() - t0)
    
    t0 = time()
    print 'hessian eigenmapping...',    
    try:
        hlle(X,y, 'Formant_%s' % globaltag)
        print 'done. time %.3fs' % (time() - t0)
    except:
        print 'failed.'        



    
    
