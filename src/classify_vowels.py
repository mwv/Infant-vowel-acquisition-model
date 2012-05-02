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
classify_vowels:

quick classification of vowels as a proxy for overlap
'''
from collections import namedtuple
from pprint import pprint
from time import time
import cPickle
import os

import numpy as np 

from sklearn.grid_search import GridSearchCV
from sklearn import manifold, lda, preprocessing
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.cross_val import StratifiedKFold
from sklearn.linear_model import LogisticRegression

from vowels.config.paths import cfg_dumpdir 
from vowels.audio_measurements.formants import FormantsMeasure
from vowels.audio_measurements.mfcc import MFCCMeasure
from vowels.speechcorpora.ifa import IFA
from vowels.speechcorpora.cgn import CGN
from vowels.speechcorpora.corpus import MergedCorpus
from vowels.clustering.confusion_matrix import ConfusionMatrix

DataSet = namedtuple('DataSet', ['X_train','y_train','X_test','y_test'])

def split_data(X, y, test_proportion=0.25, seed=42):
    """split X,y into training and test sets"""
    n = X.shape[0]
    cut = np.floor(test_proportion * n)
    p = np.random.RandomState(seed).permutation(n)
    train_idx = p[:-cut]
    test_idx = p[-cut:]
    return DataSet(X[train_idx], y[train_idx], X[test_idx], y[test_idx])

def vowel_dict_to_ds(data, 
                     scale=True, 
                     split=True, 
                     test_proportion=0.25, 
                     seed=42):
    X_train = None
    y_train = None
    X_test = None
    y_test = None
    rng = np.random.RandomState(seed)
    for idx, key in enumerate(sorted(data.keys())):
        X = data[key]
        n = X.shape[0]
        y = np.ones(n, dtype=np.int32) * idx
        cut = np.floor(test_proportion * n)
        p = rng.permutation(n)
        train_idx = p[:-cut]
        test_idx = p[-cut:]
        if X_train is None:
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_test = X[test_idx]
            y_test = y[test_idx]
        else:
            X_train = np.vstack((X_train, X[train_idx]))
            y_train = np.hstack((y_train, y[train_idx]))
            X_test = np.vstack((X_test, X[test_idx]))
            y_test = np.hstack((y_test, y[test_idx]))
        
#    X = np.vstack(data[x] for x in sorted(data.keys()))
#    y = np.hstack(np.ones(data[sorted(data.keys())[x]].shape[0],dtype=np.int32) * x
#                  for x in range(len(data)))
    symbol_map = dict(zip(sorted(data.keys()), range(len(data))))
    if scale:
        scaler = preprocessing.Scaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    return DataSet(X_train, y_train, X_test, y_test), symbol_map
    
def dimred(dataset, method='lda', outdim=20, n_neighbors=50):
    """reduce the dimensionality of the dataset"""
    if method == 'lda':
        red = lda.LDA()
    elif method == 'isomap':
        red = manifold.Isomap(n_neighbors=n_neighbors, out_dim=outdim)
    elif method == 'lle':
        red = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, out_dim=outdim, method='standard')
    elif method == 'mlle':
        red = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, out_dim=outdim, method='modified')
    else:
        raise ValueError, "method must be one of ['lda','isomap','lle','mlle']"
    
    X_train = red.fit_transform(dataset.X_train, dataset.y_train)
    X_test = red.transform(dataset.X_test)
    return DataSet(X_train, dataset.y_train, X_test, dataset.y_test)

def classify(vowels=None,features='mfcc',verbose=True):
    time0 = time()
    print 'Gathering data...',
    # prepare corpora    
    ifa_corpus = IFA()
    cgn_corpus = CGN()
    corpus = MergedCorpus([ifa_corpus, cgn_corpus])
    
    if features == 'mfcc':
        mm = MFCCMeasure(corpus)
    elif features == 'formants':
        mm = FormantsMeasure(corpus)
    else:
        raise ValueError, "features must be one of ['mfcc','formants']"
    if vowels is None:
        vowels = mm._nobs.keys()
    elif not all(v in mm._nobs.keys() for v in vowels):
        raise ValueError, "illegal vowel argument"
    

        
    data = mm.sample(vowels)
    ds, symbol_map=vowel_dict_to_ds(data, test_proportion=0.1)
    print 'done. Time: %.3fs' % (time()- time0)
    
    print '-'*20
    print '%10s%10s' % ('Vowel', '#Samples')
    for v in mm._nobs.keys():
        print '%10s%10d' % (v, mm._nobs[v])
    print '-'*20    
    
    time0 = time()
    print 'Reducing dimensions...',
    ds_lda=None
    for i in range(1000):
        try:
            ds_lda = dimred(ds, method='lda')
            break
        except: # problem with SVD convergence
            continue
    if ds_lda is None:
        print 'SVD convergence failed after 1000 times.'
        return
    print 'done. Time: %.3fs' % (time()-time0)
    
    # classify with rbf-SVM
    time0=time()
    print 'GRIDSEARCHING SVM...'
    param_grid = [{'kernel':['rbf'], 'gamma':[0.0,1e-2,1e-3,1e-4], 'C':[1e0,1e1,1e2,1e3,1e4]}]

    clf = GridSearchCV(SVC(), param_grid, score_func=f1_score, verbose=2)
    clf.fit(ds_lda.X_train, ds_lda.y_train,
            cv=StratifiedKFold(ds.y_train,3), verbose=2)
    predicted_test = clf.predict(ds_lda.X_test)
    predicted_train = clf.predict(ds_lda.X_train)
    cm_test = ConfusionMatrix(ds.y_test, predicted_test)
    cm_train = ConfusionMatrix(ds.y_train, predicted_train)
    return cm_test, cm_train, symbol_map

def pprint_cm(cm, symbol_map):
    """pretty print the confusion matrix"""
    print 
    print ' '*4,
    for v in symbol_map:
        print '%4s' % v,
    print '\n'
    inds = [cm.item_map[symbol_map[v]] for v in symbol_map]
    for v in symbol_map:
        arr = cm.matrix[cm.item_map[symbol_map[v]]][inds]
        print '%4s' % v,
        for val in arr:
            print '%4d' % val,
        print

def pprint_cm_scores(cm, symbol_map):
    """pretty print scores"""
    print
    print '+' + '-'*44 + '+'
    print '|%7s |%10s |%10s |%10s |' % ('Vowel', 'Recall', 'Precision', 'F1-score')
    print '+' + '-'*44 + '+'
    for v in symbol_map:
        print '|%7s |%10.3f |%10.3f |%10.3f |' % (v,
                                               cm.recall(symbol_map[v]),
                                               cm.precision(symbol_map[v]),
                                               cm.fscore(symbol_map[v]))
    print '+' + '-'*44 + '+'        
    print '|%7s |%10.3f |%10.3f |%10.3f |' % ('AVG', 
                                              cm.recall(),
                                              cm.precision(),
                                              cm.fscore())
    print '+' + '-'*44 + '+'    
    
def experiment():
    vowel_sq = ['I','e:','|:','}']
    cm_sq_mfcc_test, cm_sq_mfcc_train, sm_sq_mfcc = classify(vowels=vowel_sq)
    
    fid = open(os.path.join(cfg_dumpdir, 'sq_mfcc.pkl'), 'wb')
    cPickle.dump(cm_sq_mfcc_test, fid, -1)
    cPickle.dump(cm_sq_mfcc_train, fid, -1)
    cPickle.dump(sm_sq_mfcc, fid, -1)
    fid.close()
    
    cm_sq_form_test, cm_sq_form_train, sm_sq_form = classify(vowels=vowel_sq,
                                                             features='formants')
    fid = open(os.path.join(cfg_dumpdir, 'sq_formants.pkl'), 'wb')
    cPickle.dump(cm_sq_form_test,fid, -1)
    cPickle.dump(cm_sq_form_train, fid, -1)
    cPickle.dump(sm_sq_form, fid, -1)
    fid.close()
    
    cm_all_mfcc_test, cm_all_mfcc_train, sm_all_mfcc = classify()
    
    fid = open(os.path.join(cfg_dumpdir, 'all_mfcc.pkl'), 'wb')
    cPickle.dump(cm_all_mfcc_test, fid, -1)
    cPickle.dump(cm_all_mfcc_train, fid, -1)
    cPickle.dump(sm_all_mfcc, fid, -1)
    fid.close()
    
    cm_all_form_test, cm_all_form_train, sm_all_form = classify()
    
    fid = open(os.path.join(cfg_dumpdir, 'all_form.pkl'), 'wb')
    cPickle.dump(cm_all_form_test, fid, -1)
    cPickle.dump(cm_all_form_train, fid, -1)
    cPickle.dump(sm_all_form, fid, -1)
    fid.close()
    
    print '+-------------------+'
    print '| VOWEL SQUARE MFCC |'
    print '+-------------------+'    
    print
    print 'HELD OUT'
    print 
    pprint_cm_scores(cm_sq_mfcc_test, sm_sq_mfcc)
    print 
    pprint_cm(cm_sq_mfcc_test, sm_sq_mfcc)
    print 
    print 'REDISTRIBUTION'
    print 
    pprint_cm_scores(cm_sq_mfcc_train,sm_sq_mfcc)
    print
    pprint_cm(cm_sq_mfcc_train, sm_sq_mfcc)
    print
    print '+-----------------------+'
    print '| VOWEL SQUARE FORMANTS |'
    print '+-----------------------+'
    print 
    print 'HELD OUT'
    print 
    pprint_cm_scores(cm_sq_form_test, sm_sq_form)
    print
    pprint_cm(cm_sq_form_test, sm_sq_form)
    print 
    print 'REDISTRIBUTION'
    print
    pprint_cm_scores(cm_sq_form_train, sm_sq_form)
    print
    pprint_cm(cm_sq_form_train, sm_sq_form)
    print
    print '+-----------------+'
    print '| ALL VOWELS MFCC |'
    print '+-----------------+'    
    print
    print 'HELD OUT'
    print 
    pprint_cm_scores(cm_all_mfcc_test, sm_all_mfcc)
    print 
    pprint_cm(cm_all_mfcc_test, sm_all_mfcc)
    print 
    print 'REDISTRIBUTION'
    print 
    pprint_cm_scores(cm_all_mfcc_train,sm_all_mfcc)
    print
    pprint_cm(cm_all_mfcc_train, sm_all_mfcc)
    print
    print '+---------------------+'
    print '| ALL VOWELS FORMANTS |'
    print '+---------------------+'
    print 
    print 'HELD OUT'
    print 
    pprint_cm_scores(cm_all_form_test, sm_all_form)
    print
    pprint_cm(cm_all_form_test, sm_all_form)
    print 
    print 'REDISTRIBUTION'
    print
    pprint_cm_scores(cm_all_form_train, sm_all_form)
    print
    pprint_cm(cm_all_form_train, sm_sq_form)
    print    
        
    
    
if __name__ == '__main__':
    experiment()
    
    
    
    


    
    
    
    
    
    
    
    
    
    
    
                
    