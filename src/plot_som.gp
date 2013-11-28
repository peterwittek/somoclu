set datafile commentschars "#!%"
set pm3d at b
set pm3d map
set nocolorbox
unset surface
set isosamples 100,100
set term png
set output 'som.png'
splot 'umat.umx'  matrix
