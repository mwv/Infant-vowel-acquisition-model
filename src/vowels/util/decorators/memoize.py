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
vowels.util.decorators.memoize:

memoizing decorator

'''

import functools

class memoize(object):
    def __init__(self, func):
        self.func = func
        self.cache = {}
        
    def __call__(self, *args, **kwargs):
        key = (args, frozenset(sorted(kwargs.items())))
        try:
            return self.cache[key]
        except KeyError:
            value = self.func(*args, **kwargs)
            self.cache[key] = value
            return value
        except TypeError: # args/kwargs are uncachable
            return self.func(*args, **kwargs)
        
    def __repr__(self):
        return self.func.__doc__
    
    def __get__(self, obj, objtype):
        return functools.partial(self.__call__, obj)
