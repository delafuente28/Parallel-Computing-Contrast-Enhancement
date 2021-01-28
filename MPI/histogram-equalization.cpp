#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"
#include <mpi.h>

const int root = 0;

void histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin)
{
    int i;

    for (i = 0; i < nbr_bin; i++){
        hist_out[i] = 0;
    }

    for (i = 0; i < img_size; i++){
        hist_out[img_in[i]] ++;
    }
}

void histogram_equalization(unsigned char * img_out, unsigned char * img_in, 
                            int * hist_in, int img_size, int nbr_bin)
{
    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Who am I
    MPI_Comm_size(MPI_COMM_WORLD, &size); // How many processes

    int img_size_tmp = img_size * size;

    int *lut = (int *)malloc(sizeof(int) * nbr_bin);
    
    int i, cdf, min, d;
    /* Construct the LUT by calculating the CDF */
    cdf = 0;
    min = 0;
    i = 0;
    
    int *buf_hist_in = (int *)malloc(nbr_bin * sizeof(int));
    //Une los histogramas que existen en cada proceso y hace un broadcast del resultado.
    MPI_Allreduce(hist_in, buf_hist_in, nbr_bin, MPI_INT, MPI_SUM, MPI_COMM_WORLD); 

    while(min == 0)
    {
        min = buf_hist_in[i++];
    }

    d = img_size_tmp - min;

    for(i = 0; i < nbr_bin; i++)
    {
        cdf += buf_hist_in[i];
        //lut[i] = (cdf - min)*(nbr_bin - 1)/d;
        lut[i] = (int)(((float)cdf - min)*255/d + 0.5);
        if(lut[i] < 0){
            lut[i] = 0;
        }
    }

    /* Get the result image */
    for(i = 0; i < img_size; i++)//(i = rank*part_size; i <= (rank*part_size)+part_size; i++)
    {
        if(lut[img_in[i]] > 255)
        {
            img_out[i] = 255;
        }
        else{
            img_out[i] = (unsigned char)lut[img_in[i]];
        }
        
    }
    
    free(buf_hist_in);
}