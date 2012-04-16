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
config.paths:

'''

from __future__ import division

__author__ = 'Maarten Versteegh'
__date__ = 'Apr 7, 2012'

import os

# absolute path to THIS module
_module_path = os.path.dirname(os.path.abspath(__file__))

cfg_datadir = os.path.realpath(os.path.join(_module_path, '../../../../data/'))
cfg_celexdir = os.path.join(cfg_datadir, 'CELEX')
cfg_childesdir = os.path.join(cfg_datadir, 'childes')
cfg_ifadir = os.path.join(cfg_datadir, 'ifa')
cfg_dumpdir = os.path.join(cfg_datadir, 'dumps')
if not os.path.exists(cfg_dumpdir):
    os.makedirs(cfg_dumpdir)
    
cfg_figdir = os.path.join(cfg_datadir, 'figs')
if not os.path.exists(cfg_figdir):
    os.makedirs(cfg_figdir)


