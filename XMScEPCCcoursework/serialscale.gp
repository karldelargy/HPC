# comment out the following two lines if you want to see the graph
# on the screen using: gnuplot -persist serialscale.gp
set term pdf
set output "serialscale.pdf"
set logscale x 10
set xlabel "NCELL"
set ylabel "MCOPs"
plot [][0:1000]"serialscale.dat" using 5:11 \
title "Serial code" with linespoint
