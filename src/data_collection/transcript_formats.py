#!/usr/bin/python
# -*- coding: utf-8 -*-
""" 
"""

from __future__ import division

__author__ = 'Maarten Versteegh'
__date__ = 'Fri Apr  6 22:06:03 2012'

from bidict import bidict

vowels_sampa = ['i:', # liep
                'y:', # buut
                'e:', # leeg
                '|:', # deuk
                'a:', # laat
                'o:', # boom
                'u:', # boek
                'I',  # lip
                'E',  # leg
                'A',  # lat
                'O',  # bom
                '}',  # put
                '@',  # schwa
                'i::', # analyse
                'y::', # centrifuge
                'E:',  # scene
                '/:',  # freule
                'Q:',  # zone
                'EI',  # wijs
                '/I',  # huis
                'Au']  # koud

disc_to_sampa = bidict({
    ' ':' ',
    'p':'p',
    'b':'b',
    't':'t',
    'd':'d',
    'k':'k',
    'g':'g',
    'N':'N',
    'm':'m',
    'n':'n',
    'l':'l',
    'r':'r',
    'f':'f',
    'v':'v',
    'T':'T',
    'D':'D',
    's':'s',
    'z':'z',
    'S':'S',
    'Z':'Z',
    'j':'j',
    'x':'x',
    'G':'G',
    'h':'h',
    'w':'w',
    '+':'pf',
    '=':'ts',
    'J':'tS',
    '_':'dZ',
    'C':'N,',
    'F':'m,',
    'H':'n,',
    'P':'l,',
    'R':'r*',
    'i':'i:',
    '!':'i::',
    '#':'A:',
    'a':'a:',
    '$':'0:',
    'u':'u:',
    '3':'3:',
    'y':'y:',
    '(':'y::',
    ')':'E:',
    '*':'/:',
    '<':'Q:',
    'e':'e:',
    '|':'|:',
    'o':'o:',
    '1':'eI',
    '2':'aI',
    '4':'OI',
    '5':'@U',
    '6':'aU',
    '7':'I@',
    '8':'E@',
    '9':'U@',
    'K':'EI',
    'L':'/I',
    'M':'Au',
    'W':'ai',
    'B':'au',
    'X':'Oy',
    'I':'I',
    'Y':'Y',
    'E':'E',
    '/':'/',
    '{':'{',
    '&':'{',
    'A':'A',
    'Q':'Q',
    'V':'V',
    'O':'O',
    'U':'U',
    '}':'}',
    '@':'@',
    '^':'/~:',
    'c':'{~',
    'q':'A~:',
    '0':'{~:',
    '-':'-',
    "'":"\"",
    "\"":'%',
    '.':'.'})
    


