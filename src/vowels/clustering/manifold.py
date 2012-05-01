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
vowels.clustering.manifold:

compare manifold learning methods
'''

import numpy as np

from time import time

from sklearn.utils.fixes import qr_economic
from sklearn import manifold, decomposition, lda, qda

def pca(X, y, n_components=None):
    X_pca = decomposition.RandomizedPCA(n_components=n_components).fit_transform(X, y)
    return X_pca

def lda(X, y, n_components=None):
    X2 = X.copy()
    X2.flat[::X.shape[1] + 1] += 0.01
    X2.flat = lda.LDA(n_components=n_components).fit_transform(X2, y)
    
def isomap(X, y, n_neighbors=None):
    X_iso = manifold.Isomap(n_neighbors=n_neighbors,method='standard').fit_transform(X, y)
    return X_iso


