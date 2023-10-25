#!/bin/sh

### LSF syntax
#BSUB -alloc_flags ipisolate
#BSUB -nnodes 20                  #number of nodes
#BSUB -W 720                      #walltime in minutes
#BSUB -G fnops4ss                   #account
#BSUB -e run_myerrors.%J.txt             #stderr
#BSUB -o run_myoutput.%J.txt             #stdout
#BSUB -J smart                    #name of job
#BSUB -q pbatch               #queue to use

### Shell scripting
date; hostname
echo -n 'JobID is '; echo $LSB_JOBID
cd /g/g92/kong11/workspace/Projects/smart/2B/3d_pressure_estimation_ufno/src

firsthost=`jsrun --nrs 1 -r 1 /bin/hostname`

export MASTER_ADDR=$firsthost
export MASTER_PORT=23456

### Launch parallel executable
jsrun -r 4 -g 1 --bind none -c 10 --smpiargs="-disable_gpu_hooks" python main.py

echo 'Done'
