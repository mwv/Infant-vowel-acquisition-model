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
vowels.data_collection.cgn_preproc:

functions to preprocess the cgn corpus for audio measurements:
1. split long cgn wave files into chunks according to specifications in .mlf files
2. convert htk output in .mlf files to separate textgrids

'''

from __future__ import division

import os
import re

import scikits.audiolab
import glob
from collections import namedtuple

from ..config.paths import cfg_cgndir, cfg_cgnmlfdir, cfg_cgnwavdir, cfg_cgntgdir

namep = re.compile(r'out(?P<name>[\w\d]+)_\d+.mlf$')
chunkp = re.compile(r'[\w\d]+__(?P<start>\d+)-(?P<end>\d+).rec$')

Interval = namedtuple('Interval', ['xmin','xmax','mark'])

def write_textgrid(intervals, outfile):
    fid = open(outfile, 'w')
    fid.write('File type = "ooTextFile"\n')
    fid.write('Object class = "TextGrid"\n')
    fid.write('\n')
    fid.write('xmin = %.5f\n' % intervals[0].xmin)
    fid.write('xmax = %.5f\n' % intervals[-1].xmax)
    fid.write('tiers? <exists>\n')
    fid.write('size = 1\n')
    fid.write('item []:\n')
    fid.write('    item [1]:\n')
    fid.write('        class = "IntervalTier"\n')
    fid.write('        name = "phone alignment"\n')
    fid.write('        xmin = %.5f\n' % intervals[0].xmin)
    fid.write('        xmax = %.5f\n' % intervals[-1].xmax)
    fid.write('        intervals: size = %d\n' % len(intervals))
    for idx, interval in enumerate(intervals):
        fid.write('        intervals [%d]:\n' % (idx + 1))
        fid.write('            xmin = %.5f\n' % interval.xmin)
        fid.write('            xmax = %.5f\n' % interval.xmax)
        fid.write('            text = "%s"\n' % interval.mark)
    fid.close()

def convert_mlf_path(mlf_path, outpath):
    for f in glob.iglob(os.path.join(mlf_path, '*.mlf')):
        mlf_to_textgrids(f, outpath)

def mlf_to_textgrids(mlf_file, outpath):
    """extract all chunks from .mlf file and output individually
    to textgrids"""
    basename = namep.search(mlf_file.split(os.sep)[-1]).group('name')

    chunk = []
    chunking=False
    start, end = None, None
    for line in open(mlf_file,'r'):
        if line.startswith('#'):
            continue
        if line.startswith('.'):
            chunking=False
            write_textgrid(chunk,
                           os.path.join(outpath,
                                        basename + '__%s-%s.ltg' % (start,end)))
            chunk = []
            start, end = None, None
            
        if chunking:
            data = line.strip('\n').split(' ')[:3]
            interval = Interval(int(data[0])/10000000,
                                int(data[1])/10000000,
                                data[2])
            chunk.append(interval)

        if line.startswith('\"'):
            start, end = chunkp.search(line.strip('\n').split('/')[-1][:-1]).groups()
            chunking = True

def get_chunks_from_mlf(mlf_file):
    basename = mlf_file.split(os.sep)[-1]
    name = namep.search(basename).group('name')
    chunks = []
    for line in open(mlf_file,'r'):
        if line.startswith('#'):
            # first line in file
            continue
        if line.startswith('\"'):
            rec_fname = line.strip('\n').split(os.sep)[-1][:-1]
            start, end = chunkp.search(rec_fname).groups()
            chunks.append((int(start),int(end)))
    return name, chunks

def get_all_chunks(path):
    """return a dict from basenames to chunks from a path"""
    d = {}
    for file in glob.iglob(os.path.join(path, '*.mlf')):
        basename, chunks = get_chunks_from_mlf(file)
        try:
            d[basename].extend(chunks)
        except:
            d[basename] = chunks
    return d

def split_wav(wavfile, chunks, outpath):
    wave, fs, enc = scikits.audiolab.wavread(wavfile)
    basename = wavfile.split(os.sep)[-1][:-4]
    for (start, end) in chunks:
        start_sample = start/10000000 * fs
        end_sample = end/10000000 * fs
        scikits.audiolab.wavwrite(wave[start_sample:end_sample],
                                  os.path.join(outpath, basename + '__%d-%d.wav' % (start, end)),
                                  fs=fs,
                                  enc=enc)

def split_wavs(mlf_path, outpath):
    chunks = get_all_chunks(mlf_path)
    fpath = os.path.join(cfg_cgndir, 'wavs_orig')
    for basename  in chunks:
        wavfile = os.path.join(fpath, 'comp-f','nl', basename+'.wav')
        if not os.path.exists(wavfile):
            wavfile = os.path.join(fpath, 'comp-o','nl', basename+'.wav')
        split_wav(wavfile, chunks[basename], outpath)
        
if __name__ == '__main__':
    # split the audio files:
    split_wavs(cfg_cgnmlfdir, cfg_cgnwavdir)
    
    # extract textgrids from htk .mlf files
    convert_mlf_path(cfg_cgnmlfdir, cfg_cgntgdir)
    

    
    
            
        


