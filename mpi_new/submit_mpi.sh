#! /bin/bash -l
# The -l specifies that we are loading modules
#
## Walltime limit
#$ -l h_rt=2:00:00
#
## Give the job a name.
#$ -N mpi_v
#
## Redirect error output to standard output
#$ -j y
#
## What project to use. "paralg" is the project for the class
#$ -P paralg
#
## Ask for nodes with 4 cores, 4 cores total (so 1 node)
#$ -pe mpi_4_tasks_per_node 4

# Want more flags? Look here:
# http://www.bu.edu/tech/support/research/system-usage/running-jobs/submitting-jobs/

# Load the correct modules
module load gcc/7.2.0  # compiler
module load mvapich2/2.3_gcc-7.2.0  # consistent mpi compile

# Immediately form fused output/error file, besides the one with the default name.
exec >  ./mpi_v.scc.out 2>&1
mpic++ time_dependent_v.cpp -o mpi_v
# Invoke mpirun.
# SGE sets $NSLOTS as the total number of processors (8 for this example) 
mpirun ./mpi_v

exit

