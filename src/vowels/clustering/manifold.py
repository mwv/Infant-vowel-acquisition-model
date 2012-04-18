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

from time import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.utils.fixes import qr_economic
from sklearn import manifold, decomposition, lda




