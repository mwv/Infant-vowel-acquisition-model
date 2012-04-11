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
vowels.data_collection.praat_interact:

'''

from subprocess import Popen, PIPE

class PraatError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return self.value
    
def run_praat(*args):
    """run praat with arguments and return result as c string"""
    p = Popen(['praat'] + list(args),
              shell=False,
              stdin=PIPE,
              stdout=PIPE,
              stderr=PIPE)
    p.wait()
    stdout = p.stdout
    stderr = p.stderr
    if p.returncode:
        raise PraatError(''.join(stderr.readlines()))
    else:
        return stdout.readlines()