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
vowels.util.standards:

standards for transcription plus convenience functions for conversion
'''

from bidict import bidict

vowels_sampa = set(
                ['i:', # liep
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
                'Au',  # koud
                '{', # trap (eng)
                '6', # open schwa
                '3', # nurse (eng)
                '2', # deux (fr)
                '9', # neuf (fr)
                '&', # open front rounded
                'U', # foot (eng)
                'V', # strut (eng)
                'Y', # huebsch (de)
                'E~', # vacc_in_ (fr)
                'A~', # croiss_ant_ (fr)
                'O~', # c_o_nge (fr)
                'Y~']) # parf_um_ (fr)

class PhoneSymbolError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return self.value

def sampa_to_cgn(p):
    try:
        return _sampa_to_cgn[p]
    except KeyError:
        raise PhoneSymbolError, '%s is not a valid sampa symbol' % p
        
def cgn_to_sampa(p):
    try:
        return _sampa_to_cgn[:p]
    except KeyError:
        raise PhoneSymbolError, '%s is not a valid cgn symbol' % p

def disc_to_sampa(p):
    try:
        return _disc_to_sampa[p]
    except KeyError:
        raise PhoneSymbolError, '%s is not a valid disc symbol' % p

def sampa_to_disc(p):
    try:
        return _disc_to_sampa[:p]
    except KeyError:
        raise PhoneSymbolError, '%s is not a valid sampa symbol' % p
    
def sampa_to_unicode(p):
    try:
        return _sampa_to_unicode_ipa[p]
    except KeyError:
        raise PhoneSymbolError, '%s is not a valid sampa vowel' % p
    
    
# (vowels only) unicode character codes
_sampa_to_unicode_ipa = bidict({
'i:':'i',
'y:':'y',
'e:':'e',
'|:':ur'\u00F8',
'2':ur'\u00F8',
'a:':ur'a',
'o:':'o:',
'u:':'u',
'I':ur'\u026A',
'E':ur'\u025B',
'A':ur'\u0251',
'O':ur'\u0254',
'}':ur'\u028F',
'@':ur'\u0259',
'i::':'y::',
'E:':ur'\025B:',
'/:':ur'\u00F8:',
'Q:':ur'\u0254:',
'EI':ur'\u025Bi',
'/I':ur'\u0153\u028F',
'Au':ur'\u0251\u028A'})

                    

_sampa_to_cgn = bidict({
' ' : ' ',                       
'p' : 'p',
'b' : 'b',
't' : 't',
'd' : 'd',
'k' : 'k',
'g' : 'g',
'f' : 'f',
'v' : 'v',
's' : 's',
'z' : 'z',
'S' : 'S',
'Z' : 'Z',
'x' : 'x',
'G' : 'G',
'h' : 'h',
'N' : 'N',
'm' : 'm',
'n' : 'n',
'nj' : 'J',
'l' : 'l',
'r' : 'r',
'w' : 'w',
'j' : 'j',
'I' : 'I',
'E' : 'E',
'A' : 'A',
'O' : 'O',
'}' : 'Y',
'i:' : 'i',
'y:' : 'y',
'e:' : 'e',
'|:' : '2',
'a:' : 'a',
'o:' : 'o',
'u:' : 'u',
'@' : '@',
'EI' : 'E+',
'/I' : '9+',
'Au' : 'O+',
'E:' : 'E:',
'/:' : '9:',
'Q:' : 'O:',
'E~' : 'E~',
'A~' : 'A~',
'O~' : 'O~',
'Y~' : 'Y~'
})

_sampa_injection_map = {
'i:':'i:',
'y:':'y:',
'e:':'e:',
'|:':'|:', 
'a:':'a:',
'u:':'u:',
'I':'I',
'E':'E',
'A':'A',
'O':'O',
'}':'}',
'@':'@',
'i::':'i:',
'y::':'y:',
'E:':'E',
'/:':'|:',
'Q:':'O',
'EI':'EI',
'/I':'/I',
'Au':'Au',
'{':'E',
'6':'@',
'2':'|:',
'9':'}',
'U':'u:',
'V':'A',
'Y':'y:',
'E~':'E',
'A~':'A',
'O~':'O',
'Y~':'}',
'A+':'Au'}
                        

_sampa_to_htk_vowels = {
'i:':'i', # liep
'y:':'y', # buut
'e:':'e', # leeg
'|:':'eu', # deuk
'a:':'a', # laat
'o:':'o', # boom
'u:':'u', # boek
'I':'ic', # lip
'E':'ec', # leg
'A':'ac', # lat
'O':'oc', # bom
'}':'yc', # put
'@':'@', # schwa
'i::':'i', # analyse
'y::':'y', # centrifuge
'E:':'ec', # scene (fr)
'/:':'eu', # freule (fr)
'Q:':'oc', # zone (fr)
'EI':'ei', # wijs
'/I':'ui', # huis
'Au':'au', # koud
'{':'ec', # trap (eng)
'6':'@', # open schwa
'2':'eu', # deux (fr)
'9':'yc', # neuf (fr)
'U':'u', # foot (eng)
'V':'ac', # strut (eng)
'Y':'y', # huebsch (de)
'E~':'ec', # vacc_in_ (fr)
'A~':'ac', # cross_ant_ (fr)
'O~':'oc', # conge (fr)
'Y~':'yc', # parf_um_ (fr)
'A+':'au'}


_disc_to_sampa = bidict({
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


