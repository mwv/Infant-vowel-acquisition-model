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
vowels.clustering.classifiers:

'''

import numpy as np
from sklearn import lda, qda, svm, metrics, neighbors
from sklearn.linear_model import LogisticRegression


def lda_clf(X_train, y_train, X_test):
    """ classify based on lda"""
    clf = lda.LDA()
    clf.fit(X_train, y_train)
    return clf.predict(X_test)

def qda_clf(X_train, y_train, X_test):
    """ classify based on qda"""
    clf = qda.QDA()
    clf.fit(X_train, y_train)
    return clf.predict(X_test)

csvm_params = dict(C=[None, 1e-1,1e0,1e1,1e10,1e100],
                   kernel=['rbf','poly','sigmoid'],
                   gamma=[0.0,1e-1,1e0,1e1,1e10],
                   degree=[2,3,5,10],
                   shrinking=[True,False],
                   scale_C=[True,False])

def csvm(X_train, y_train, X_test,
         C=None,
         kernel='rbf',
         degree=3,
         gamma=0.0,
         shrinking=True,
         scale_C=True
         ):
    """c-support vector classification with radial basis functions"""
    clf = svm.SVC(C=C,kernel=kernel,degree=degree,gamma=gamma,shrinking=shrinking,scale_C=scale_C)
    clf.fit(X_train, y_train)
    return clf.predict(X_test)

linearsvm_params = dict(penalty=['l1','l2'],
                        loss=['l1','l2'],
                        fit_intercept=[True, False],
                        intercept_scaling=[1e-1,1e0,1e1,1e10],
                        scale_C=[True,False])

def linearsvm(X_train, y_train, X_test,
              penalty='l2',
              loss='l2',
              fit_intercept=True,
              intercept_scaling=1.0,
              scale_C=False):
    """linear support vector classification"""
    clf = svm.LinearSVC(penalty=penalty,
                        loss=loss,
                        fit_intercept=fit_intercept,
                        intercept_scaling=intercept_scaling,
                        scale_C=scale_C)
    clf.fit(X_train, y_train)
    return clf.predict(X_test)

logreg_params = dict(penalty=['l1','l2'],
                     C=[1e-1,1e0,1e1,1e10],
                     fit_intercept=[True,False],
                     intercept_scaling=[1e-1,1e0,1e1,1e10],
                     scale_C=[True, False])

def logreg(X_train, y_train, X_test,
           penalty='l2', 
           C=1.0,
           fit_intercept=True, 
           intercept_scaling=1.0, 
           scale_C=False):
    """logistic regression"""
    clf = LogisticRegression(X=0.1, 
                             penalty=penalty, 
                             C=C, 
                             fit_intercept=fit_intercept, 
                             intercept_scaling=intercept_scaling, 
                             scale_C=scale_C)
    clf.fit(X_train, y_train)
    return clf.predict(X_test)

knn_params = [dict(n_neighbors=[10,25,50,100], weights='distance')]
knn_params = dict(n_neighbors=[10,25,50,100], 
                  weights=['distance','uniform'], 
                  algorithm=['auto','ball_tree','kd_tree','brute'],
                  leaf_size=['10','30','50','100'],
                  minkowski_p=[1,2])

def knn(X_train, y_train, X_test, 
        n_neighbors=50, 
        weights='distance', 
        algorithm='auto', 
        leaf_size=30, 
        minkowski_p=2):
    """k-nearest neighbor classification"""
    clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, 
                                         algorithm=algorithm,
                                         leaf_size=leaf_size,
                                         p=minkowski_p)
    clf.fit(X_train, y_train)
    return clf.predict(X_test)

rknn_params = dict(radius=[0.5,1.0,2.0,10.0],
                   weights=['uniform', 'distance'],
                   algorithm=['audio','ball_tree','kd_tree', 'brute'],
                   leaf_size=['10','30','50','100'],
                   minkowski_p=[1,2])

def rknn(X_train, y_train, X_test, 
         radius=1.0, 
         weights='uniform',
         algorithm='auto', 
         leaf_size=30, 
         minkowski_p=2):
    """radius-based k-nearest neighbor classification"""
    clf = neighbors.RadiusNeighborsClassifier(radius=radius, 
                                              weights=weights,
                                              algorithm=algorithm, 
                                              leaf_size=leaf_size, 
                                              p=minkowski_p)
    clf.fit(X_train, y_train)
    return clf.predict(X_test)

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




    

