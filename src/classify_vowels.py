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

def vowel_dict_to_X_y(data, scale=True):
    X = np.vstack(data[x] for x in sorted(data.keys()))
    y = np.hstack(np.ones(data[sorted(data.keys())[x]].shape[0],dtype=np.int32) * x
                  for x in range(len(data)))
    symbol_map = dict(zip(sorted(data.keys()), range(len(data))))
    if scale:
        return preprocessing.scale(X), y, symbol_map
    else:
        return X,y, symbol_map
    
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

def get_formant_data(vowels):
    ifa_corpus = IFA()
    cgn_corpus = CGN()
    corpus = MergedCorpus([ifa_corpus, cgn_corpus])
    fm = FormantsMeasure(corpus)
    data = fm.sample(vowels)
    X,y, symbol_map = vowel_dict_to_X_y(data)
    return split_data(X, y, 0.1), symbol_map

def get_mfcc_data(vowels=None):
    ifa_corpus = IFA()
    cgn_corpus = CGN()
    corpus = MergedCorpus([ifa_corpus, cgn_corpus])
    mm = MFCCMeasure(corpus)   
    if vowels is None:
        vowels = mm._nobs.keys() 
    data = mm.sample(vowels)
    X,y, symbol_map=vowel_dict_to_X_y(data)
    return split_data(X, y, 0.1), symbol_map

def print_results(cm, name):
    print '-'*21
    print '%s classification:' % name
    print '-'*21
    print 'f-score:\t%.3f' % cm.fscore()
    print 'precision:\t%.3f' % cm.precision()
    print 'recall:     \t%.3f' % cm.recall()
    print '-'*21
    
def repr_results(cm, name):
    res = '-'*21+'\n'
    res += '%s classification:\n' % name
    res += '-'*21+'\n'
    res += 'f-score:  \t%.3f\n' % cm.fscore()
    res += 'precision:\t%.3f\n' % cm.precision()
    res += 'recall:   \t%.3f\n' % cm.recall()
    res += '-'*21+'\n'
    
def lda_clf(dataset):
    """perform lda classification"""
    clf = lda.LDA()
    clf.fit(dataset.X_train, dataset.y_train)
    pred = clf.predict(dataset.X_test)
    cm = ConfusionMatrix(dataset.y_test, pred)
    #print_results(cm, 'LDA')
    return cm,repr_results(cm,'LDA')
    
def svm_clf(dataset, optimize=True, verbose=True):
    param_grid = [{'kernel':['rbf'], 'gamma':[0.0,1e-2,1e-3,1e-4,1e-5,1e-6], 'C':[1e0,1e1,1e2,1e3]},
                  #{'kernel':['linear'], 'C':[1e0,1e1,1e2,1e3]},
                  {'kernel':['poly'], 'degree':[3,4], 'gamma':[0.0,1e-2,1e-3,1e-4], 'C':[1e0,1e1,1e2]},
                  #{'kernel':['sigmoid'], 'degree':[1,2,3,4,5], 'C':[1e0,1e1,1e2,1e3]}
                  ]
    scores = [('precision', precision_score),
              ('recall', recall_score),
              ('f1', f1_score)]

    score_name, score_func = scores[2]
    clf = GridSearchCV(SVC(), param_grid, score_func=score_func,verbose=2, n_jobs=1, pre_dispatch=None)
    clf.fit(dataset.X_train, dataset.y_train, cv=StratifiedKFold(dataset.y_train, 3), verbose=2)
    predicted = clf.predict(dataset.X_test)
    if verbose:
        print 'Classification report for the best estimator: '
        print clf.best_estimator
        print 'Tuned for %s with optimal value: %.3f' % (score_name, score_func(dataset.y_test, predicted))
        print classification_report(dataset.y_test, predicted)
        print 'Grid scores:'
        pprint(clf.grid_scores_)
        print
    cm = ConfusionMatrix(dataset.y_test, predicted)
    return cm,repr_results(cm, 'SVM')

def logreg_clf(dataset, verbose=True):
    param_grid = [{'penalty':['l1','l2'], 'C':[1e-1,1e0,1e1,1e2],}]
    scores = [('precision', precision_score),
              ('recall', recall_score),
              ('f1', f1_score)]

    score_name, score_func = scores[2]
    clf = GridSearchCV(LogisticRegression(), param_grid, score_func=score_func,verbose=2, n_jobs=1, pre_dispatch=None)
    clf.fit(dataset.X_train, dataset.y_train, cv=StratifiedKFold(dataset.y_train, 3), verbose=2)
    predicted = clf.predict(dataset.X_test)
    if verbose:
        print 'Classification report for the best estimator: '
        print clf.best_estimator
        print 'Tuned for %s with optimal value: %.3f' % (score_name, score_func(dataset.y_test, predicted))
        print classification_report(dataset.y_test, predicted)
        print 'Grid scores:'
        pprint(clf.grid_scores_)
        print
    cm = ConfusionMatrix(dataset.y_test, predicted)
    return cm,repr_results(cm, 'LogisticRegression')   

def knn_clf(dataset, verbose=True):
    param_grid = [{'n_neighbors':[10,25,50], 'weights':['uniform','distance']}]
    scores = [('precision', precision_score),
              ('recall', recall_score),
              ('f1', f1_score)]

    score_name, score_func = scores[2]
    clf = GridSearchCV(KNeighborsClassifier(), param_grid, score_func=score_func,verbose=2, n_jobs=1, pre_dispatch=None)
    clf.fit(dataset.X_train, dataset.y_train, cv=StratifiedKFold(dataset.y_train, 3), verbose=2)
    predicted = clf.predict(dataset.X_test)
    if verbose:
        print 'Classification report for the best estimator: '
        print clf.best_estimator
        print 'Tuned for %s with optimal value: %.3f' % (score_name, score_func(dataset.y_test, predicted))
        print classification_report(dataset.y_test, predicted)
        print 'Grid scores:'
        pprint(clf.grid_scores_)
        print
    cm = ConfusionMatrix(dataset.y_test, predicted)
    return cm,repr_results(cm, 'KNN')     

def rnn_clf(dataset, verbose=True):
    param_grid = [{'radius':[0.5,1.0,2.0,10.], 'weights':['uniform','distance']}]
    scores = [('precision', precision_score),
              ('recall', recall_score),
              ('f1', f1_score)]

    score_name, score_func = scores[2]
    clf = GridSearchCV(RadiusNeighborsClassifier(), param_grid, score_func=score_func,verbose=2, n_jobs=1, pre_dispatch=None)
    clf.fit(dataset.X_train, dataset.y_train, cv=StratifiedKFold(dataset.y_train, 3), verbose=2)
    predicted = clf.predict(dataset.X_test)
    if verbose:
        print 'Classification report for the best estimator: '
        print clf.best_estimator
        print 'Tuned for %s with optimal value: %.3f' % (score_name, score_func(dataset.y_test, predicted))
        print classification_report(dataset.y_test, predicted)
        print 'Grid scores:'
        pprint(clf.grid_scores_)
        print
    cm = ConfusionMatrix(dataset.y_test, predicted)
    return cm,repr_results(cm, 'RNN')   

def all_vowels_classification(verbose=True):
    time0 = time()
    print 'Gathering data...',
    # prepare corpora
    ifa_corpus = IFA()
    cgn_corpus = CGN()
    corpus = MergedCorpus([ifa_corpus, cgn_corpus])
    
    # gather mfcc data
    mm = MFCCMeasure(corpus)
    vowels = mm._nobs.keys()
    data = mm.sample(vowels)
    X,y, symbol_map=vowel_dict_to_X_y(data)
    ds = split_data(X,y,0.1)
    print 'done. Time: %.3fs' % (time()- time0)
    
    # now throw the book at it
    
    # 1. lda classification
    print 'Running LDA...',
    time0 = time()
    lda_cm,lda_result = lda_clf(ds)
    print 'done. Time: %.3fs' % (time() - time0)
    
    # 2. svm classification
    # 2.1 lda dimensionality reduction
    print 'Running svm classification with lda dimensionality reduction...'
    ds_lda = dimred(ds, method='lda')
    time0=time()    
    svm_lda_cm,svm_lda_result = svm_clf(ds_lda)
    print 'SVM classification... done. Time: %.3fs' % (time() - time0)
    # 2.2 lle dimensionality reduction
    print 'Running svm classification with lle dimensionality reduction...'
    time0=time()
    ds_lle = dimred(ds, method='lle')
    svm_lle_cm, svm_lle_result = svm_clf(ds_lle)
    print 'SVM classification...done. Time: %.3fs' % (time()-time0)
    # 2.3 mlle dimensionality reduction
    print 'Running svm classification with mlle dimensionality reduction...'
    time0=time()
    ds_mlle = dimred(ds, method='mlle')
    svm_mlle_cm, svm_mlle_result = svm_clf(ds_mlle)
    print 'SVM classification...done. Time: %.3fs' % (time() - time0)
    # 2.4 isomap dimred
    print 'Running svm classification with isomap dimensionality reduction...'
    time0=time()
    ds_isomap = dimred(ds, method='isomap')
    svm_isomap_cm, svm_isomap_result = svm_clf(ds_isomap)
    print 'SVM classification...done. Time: %.3fs' % (time() - time0)
    
    # 3. logistic regression
    # 3.1 lda dimred
    print 'Running logreg classification with lda dimensionality reduction...'
    time0=time()
    logreg_lda_cm, logreg_lda_result = logreg_clf(ds_lda)
    print 'Logistic Regression...done. Time %.3fs' % (time()-time0)
    # 3.2 lle
    print 'Running logreg with lle dimensionality reduction...'
    time0 = time()
    logreg_lle_cm, logreg_lle_result = logreg_clf(ds_lle)
    print 'logistic regression...done. Time %.3fs' % (time() - time0)
    # 3.3 mlle
    print 'Running logreg with mlle dimensionality reduction...'
    time0=time()
    logreg_mlle_cm, logreg_mlle_result = logreg_clf(ds_mlle)
    print 'logistic regression... done. Time: %.3fs' % (time()-time0)
    # 3.4 isomap
    print 'Running logreg with isomap dimensionality reduction...'
    time0 = time()
    logreg_isomap_cm, logreg_isomap_result = logreg_clf(ds_isomap)
    print 'logistic regression... done. Time; %.3fs' % (time() - time0)
    
    # knn
    
    print 'Saving confusion matrices...',
    fid = open(os.path.join(cfg_dumpdir, 'lda_cm.pkl'), 'wb')
    cPickle.dump(lda_cm, fid)
    fid.close()
    
    fid = open(os.path.join(cfg_dumpdir, 'svm_lda_cm.pkl'),'wb')
    cPickle.dump(svm_lda_cm, fid)
    fid.close()
    
    fid = open(os.path.join(cfg_dumpdir, 'svm_lle_cm.pkl'), 'wb')
    cPickle.dump(svm_lle_cm, fid)
    fid.close()
    
    fid = open(os.path.join(cfg_dumpdir, 'svm_mlle_cm.pkl'), 'wb')
    cPickle.dump(svm_mlle_cm, fid)
    fid.close()
    
    fid = open(os.path.join(cfg_dumpdir, 'svm_isomap_cm.pkl'), 'wb')
    cPickle.dump(svm_isomap_cm, fid)
    fid.close()
    
    fid = open(os.path.join(cfg_dumpdir, 'logreg_lda_cm.pkl'), 'wb')
    cPickle.dump(logreg_lda_cm, fid)
    fid.close()
    
    fid = open(os.path.join(cfg_dumpdir, 'logreg_lle_cm.pkl'), 'wb')
    cPickle.dump(logreg_lle_cm, fid)
    fid.close()
    
    fid = open(os.path.join(cfg_dumpdir, 'logreg_mlle_cm.pkl'), 'wb')
    cPickle.dump(logreg_mlle_cm, fid)
    fid.close()
    
    fid = open(os.path.join(cfg_dumpdir, 'logreg_isomap_cm.pkl'), 'wb')
    cPickle.dump(logreg_isomap_cm, fid)
    fid.close()
    print 'done.'    
    
    print '-'*21
    print 'LDA'    
    print lda_result
    print '-'*21
    print 'SVM LDA'
    print svm_lda_result
    print '-'*21
    print 'SVM LLE'
    print svm_lle_result
    print '-'*21
    print 'SVM MLLE'
    print svm_mlle_result
    print '-'*21
    print 'SVM ISOMAP'
    print svm_isomap_result
    print '-'*21
    print 'LOGREG LDA'
    print logreg_lda_result
    print '-'*21
    print 'LOGREG LLE'
    print logreg_lle_result
    print '-'*21
    print 'LOGREG MLLE'
    print logreg_mlle_result
    print '-'*21
    print 'LOGREG ISOMAP'
    print logreg_isomap_result
    
    
if __name__ == '__main__':
    all_vowels_classification()
    
    


    
    
    
    
    
    
    
    
    
    
    
                
    