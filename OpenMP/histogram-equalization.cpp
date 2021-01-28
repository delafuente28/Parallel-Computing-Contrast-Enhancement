#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"
#include <time.h>
#include <omp.h>

void histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin)
{
    int i, j, numThreads;

    for (i = 0; i < nbr_bin; i++)
        {
            hist_out[i] = 0;
        }

    #pragma omp parallel
    {
        numThreads = omp_get_num_threads();
    }

    int** hist_buff = (int**)malloc(numThreads * sizeof(int*));



    #pragma omp parallel
    {
        int id = omp_get_thread_num();

        hist_buff[id] = (int *)calloc(nbr_bin , sizeof(int));
        
        #pragma omp for
            for (i = 0; i < img_size; i++) 
            {
                hist_buff[id][img_in[i]]++;
            }
    }

    for (i = 0; i < numThreads; i++)
    {
        for (j = 0; j < nbr_bin; j++)
        {
            hist_out[j] += hist_buff[i][j];
        }
    }  

    free(hist_buff);
}

void histogram_equalization(unsigned char * img_out, unsigned char * img_in, 
                            int * hist_in, int img_size, int nbr_bin)
{
    int *lut = (int *)malloc(sizeof(int)*nbr_bin);
    int i, cdf, min, d, t;
    /* Construct the LUT by calculating the CDF */
    cdf = 0;
    min = 0;
    i = 0;

    while(min == 0)
    {
        min = hist_in[i++];
    }
    
    d = img_size - min;
    
    // #pragma omp parallel for schedule(dynamic) ordered // The execution time is not improved.
    for(i = 0; i < nbr_bin; i++)
    {
        cdf += hist_in[i];
        //lut[i] = (cdf - min)*(nbr_bin - 1)/d;
        lut[i] = (int)(((float)cdf - min)*255/d + 0.5);
        if(lut[i] < 0)
        {
            lut[i] = 0;
        }
    }

    #pragma omp parallel
    {
        /* Get the result image */
        #pragma omp for schedule(static)
            for(i = 0; i < img_size; i++)
            {
                    if(lut[img_in[i]] > 255)
                    {
                        img_out[i] = 255;
                    }
                    else
                    {
                        img_out[i] = (unsigned char)lut[img_in[i]];
                    }
            }
    }

    free(lut);
}