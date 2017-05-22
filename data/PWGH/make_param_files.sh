 #!/bin/bash 

for M in `seq 9.8 0.05 15.0`; do 
    fname="getPWGH_"$M".in"
    M=$(echo "e("$M"*l(10.0))" | bc -l)
    echo $M
    echo "0.27                                         ! Omega_0" > $fname
    echo "0.70                                         ! h (= H_0/100)" >> $fname
    echo "0.80                                         ! sigma8" >> $fname
    echo "0.95                                         ! nspec" >> $fname
    echo "0.02298                                      ! Omega_b_h2" >> $fname
    echo $M"                                    ! M_0  (h^{-1} Msun)" >> $fname
    echo "0.0                                          ! z_0" >> $fname
    echo "1                                            ! median (0) or averages (1)" >> $fname
    done