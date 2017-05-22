#!/bin/bash 

for M in `seq 9.8 0.05 15.0`; do 
    echo "getPWGH_"$M$".in"
    ./getPWGH < "getPWGH_"$M$".in"
    mv "PWGH_average.dat" $M"_PWGH_average.dat"
    done