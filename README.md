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


The application needs two input images:

in.ppm
in.pgm

PGM
A PGM file is a grayscale image file saved in the portable gray map (PGM) format and encoded with one or two bytes (8 or 16 bits) per pixel.
(https://fileinfo.com/extension/pgm)

PPM
A PPM file is a 24-bit color image formatted using a text format. It stores each pixel with a number from 0 to 65536, which specifies the color of the pixel.
(https://fileinfo.com/extension/ppm)

The application returns three images:

PGM

HSL
HSL (hue, saturation, lightness) is an alternative representation of the RGB color model.
(https://en.wikipedia.org/wiki/HSL_and_HSV)

YUV
It is a color encoding system typically used as part of a color image pipeline.
(https://en.wikipedia.org/wiki/YUV)
