#!/bin/bash

###############################################################################
# This is just an example while we include the calculation of weights in the  #
# simulation code.                                                            #
#-----------------------------------------------------------------------------#
#                      THE ELF-DANNA-GALAC Task force                         #
#                      - Arlette Melo Galindo                                 #
#                      - Miguel A. Sánchez Conde                              #
#                      - Sergio Hernández Cadena                              #
#-----------------------------------------------------------------------------#
#             September-2024                                                  #
###############################################################################

nparticles=(317189 231659 169147 119011 77634 45292 23147 10628 4639 1648)
elower=(100 157 246 386 605 950 1490 2335 3660 5740)
eupper=(157 246 386 605 950 1490 2335 3660 5740 9000)

outpath=/lustre/hawcz01/scratch/userspace/sergio/EDCTestsScienceCase2

for i in "${!nparticles[@]}"; do 

    idx=`printf "%04d" ${i}`

    echo -e "Running for sim ${idx}"

    n=${nparticles[$i]}
    emin=${elower[$i]}
    emax=${eupper[$i]}

    python dmSphObs.py \
    --dmmas 10000 \
    --emin ${emin} \
    --emax ${emax} \
    --nparticles ${n} \
    --bfield 25 \
    --ofname SphPhotonObserver.txt \
    --odir ${outpath}/Sim${idx}

done