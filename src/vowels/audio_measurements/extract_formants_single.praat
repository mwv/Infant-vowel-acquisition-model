# take name of wav file from stdin and dump formant table to stdout
form File
sentence filename
endform
Read from file... 'filename$'
To Formant (burg)... 0.01 5 5500 0.025 50
List... no yes 6 no 3 no 3 no
exit
