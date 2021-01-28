# Parallel-Computing-Contrast-Enhancement
This code is based on an image contrast enhancement sequential code that it is parallelized using MPI (distributed memory) OpenMP (shared memory) and CUDA (GPGPU)

# MPI - Compile and Run:

mpicc contrast.cpp contrast-enhancement.cpp histogram-equalization.cpp -o contrast

mpirun -np X ./contrast

Note: X is the number of processes that are launched.

# OpenMP - Compile and Run:

export OMP_NUM_THREADS=X

Note: X is the number of threads that are launched.

gcc -fopenmp -o contrast contrast.cpp contrast-enhancement.cpp histogram-equalization.cpp

./contrast

# CUDA - Compile and Run:

nvcc contrast.cpp contrast-enhancement.cpp histogram-equalization.cpp -o contrast

./contrast

Note: You need a Nvidia graphic card.
