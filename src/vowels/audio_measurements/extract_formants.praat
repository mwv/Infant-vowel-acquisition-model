# praat script to dump formants into text files

form Files
sentence indir .
sentence outdir .
endform

Create Strings as file list... wavs 'indir$'*.wav
nfiles = Get number of strings

for idx from 1 to nfiles
    select Strings wavs
    fname$ = Get string... idx
    Read from file... 'indir$''fname$'
    basename$ = fname$ - ".wav"
    select Sound 'basename$'
    To Formant (burg)... 0.01 5 5500 0.025 50
    select Formant 'basename$'
    Down to Table... no yes 6 no 3 no 3 no
    select Table 'basename$'
    Write to table file... 'outdir$''basename$'.table
    select Table 'basename$'
    Remove
    select Formant 'basename$'
    Remove
    select Sound 'basename$'
    Remove
endfor
exit
