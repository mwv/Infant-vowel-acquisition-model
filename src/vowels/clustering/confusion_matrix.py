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
vowels.clustering.confusion_matrix:

'''

import numpy as np
from bidict import bidict

class ConfusionMatrix(object):
    """Confusion matrix class.

    Provides methods for various evaluation measures:
    accuracy, precision, recall, specificity, f-scores
    """
    
    def __init__(self, actual, predicted):
        """
        Arguments:
        
        """
        
        data = list(zip(actual, predicted))
        self.items = []
        for (p,a) in data:
            if not p in self.items:
                self.items.append(p)
            if not a in self.items:
                self.items.append(a)
        self.nitems = len(self.items)
        self.item_map = bidict(zip(self.items,range(self.nitems)))
        self.matrix = np.zeros((self.nitems,self.nitems),dtype='uint')
        for (a,p) in data:
            self.matrix[self.item_map[a], self.item_map[p]] += 1

    def __repr__(self):
        s = '\n'
        for i in range(self.nitems):
            s += ('\t'.join(map(str,self.matrix[i,:])) + '\n')
        return s + '\n'

    def accuracy(self, item=None):
        """Return accuracy.
        If no item is provided, overall accuracy is returned.
        """
        if item is None:
            return np.trace(self.matrix) / np.sum(self.matrix)
        else:
            idx = self.item_map[item]
            return self.matrix[idx,idx] / sum(self.matrix[idx,:])
        
    def fscore(self, item=None, beta=1.):
        """Return F-score. If an item is given, the F-score is calculated as:
        F = (1+beta^2) * (precision*recall) / (beta^2 * precision + recall)

        If no item is given, the weighted average of the individual f-scores is returned.
        """
        f_func = lambda p,r: (1. + beta**2) * (p * r) / (beta**2 * p + r)
        
        if item is None:
            fs = f_func(self._p_arr(), self._r_arr())
            fs[np.isnan(fs)] = 0.
            f = np.sum(fs * self._weights()) / self.nitems
            if np.isnan(f):
                return 0
            else:
                return f
            
        else:
            f = f_func(self.precision(item), self.recall(item))
            if np.isnan(f):
                return 0
            else:
                return f

    def precision(self, item=None):
        """Return precision.
        If no item is provided, weighted average of item precision scores is returned.
        """
        r = self._p_arr()
        if item is None:
            # return weighted average of precision scores
            return np.sum(r * self._weights()) / self.nitems
        else:
            return r[self.item_map[item]]

    def recall(self, item=None):
        """Return recall.
        If no item is provided, weighted average of item recall scores is returned.
        """
        r = self._r_arr()
        if item is None:
            # return weighted average of recall scores
            return np.sum(r * self._weights()) / self.nitems
        else:
            return r[self.item_map[item]]

    def specificity(self, item=None):
        """Return specificity.
        If no item is provided, weighted average of item specificity scores is returned.
        """
        if item is None:
            return np.array([self.specificity(x) for x in self.items]) * self._weights()
        else:
            idx = self.item_map[item]
            mask = [[(True if x == idx or y == idx else False)
                     for x in range(self.nitems)]
                    for y in range(self.nitems)]
            tn = np.sum(np.ma.masked_array(self.matrix, mask=mask))
            spec = tn / np.sum(self.matrix[:,idx])
            if np.isinf(spec):
                return 0.
            else:
                return spec

    def _weights(self):
        """Convenience method, return weights for items in weighted functions
        (precision, recall, specificity, f-scores)
        """
        return np.sum(self.matrix, axis=1) / np.array([np.sum(self.matrix) / self.nitems] * self.nitems)

    def _p_arr(self):
        """Return an array with all precision scores"""
        r = self.matrix.diagonal() / np.sum(self.matrix, axis=0)
        r[np.isnan(r)] = 0.
        return r

    def _r_arr(self):
        """Return an array with all recall scores"""
        r = self.matrix.diagonal() / np.sum(self.matrix, axis=1)
        r[np.isnan(r)] = 0.
        return r

    def latex_table(self):
        """Return string containing nicely laid out latex tabular code for the matrix"""
        raise NotImplementedError, 'latex_table not implemented yet.'


        

        

    
        
        


