#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"
#include <assert.h>

// Histogram Kernel
__global__ void HistogramKernel(int * hist, unsigned char * img_in, int img_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= img_size) return; // If i value is greater than img_size, kernel execution will finish.
    
	// Stride is the total number of threads in grid
    int stride = blockDim.x * gridDim.x;
    
	// All threads handle blockDim.x * gridDim.x consecutive elements
    while (i < img_size) 
    {
		atomicAdd(&(hist[img_in[i]]), 1);
		i += stride;
	}
	// wait for all other threads in the block to finish
	__syncthreads();
}

// Get result image Kernel
__global__ void GetResultImageKernel(unsigned char * img_out_dev, unsigned char * img_in_dev, int * lut_dev, int img_size)
{
    int i = (blockDim.x * blockIdx.x) + threadIdx.x;

    if (i >= img_size) return; // If i value is greater than img_size, kernel execution will finish.

    // Stride is the total number of threads in grid
    int stride = blockDim.x * gridDim.x;
    
    while(i < img_size)
    {
        if(lut_dev[img_in_dev[i]] > 255)
        {
            img_out_dev[i] = 255;
        }
        
        if(lut_dev[img_in_dev[i]] <= 255)
        {
            img_out_dev[i] = (unsigned char)lut_dev[img_in_dev[i]];
        }
        i+=stride;
    }
    // Wait for all other threads in the block to finish
	__syncthreads();
}

void histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin)
{
    int i;
    for (i = 0; i < nbr_bin; i++)
    {
        hist_out[i] = 0;
    }

    int * hist_out_buff;
    unsigned char * img_in_dev;

    // Allocate device memory
    size_t memSize = nbr_bin * sizeof(int);
    cudaMalloc((void **)&hist_out_buff, memSize);
    cudaMalloc((void **)&img_in_dev, img_size * sizeof(unsigned char));

    // Host to device memory copy
    cudaMemcpy(img_in_dev, img_in, img_size, cudaMemcpyHostToDevice);

    // Check for any CUDA errors
    checkCUDAError("cudaMemcpyHostToDevice");

    // Number of blocks - if the division is not exact, the resulting number is rounded up. 
    int numBlocks = (int)ceil(img_size / (float)numThreadsPerBlock);
    
    dim3 dimGrid(numBlocks);
    dim3 dimBlock(numThreadsPerBlock);
    HistogramKernel<<<dimGrid, dimBlock>>>(hist_out_buff, img_in_dev, img_size); //Kernel

    // Block until the device has completed
    cudaThreadSynchronize();

    // Check if kernel execution generated an error
    checkCUDAError("Kernel execution");

    // Device to host copy
    cudaMemcpy(hist_out, hist_out_buff, memSize, cudaMemcpyDeviceToHost);

    // Check for any CUDA errors
    checkCUDAError("cudaMemcpyDeviceToHost");

    // Free device memory
    cudaFree(hist_out_buff);
    cudaFree(img_in_dev);
}

void histogram_equalization(unsigned char * img_out, unsigned char * img_in, int * hist_in, int img_size, int nbr_bin)
{
    int *lut = (int *)malloc(sizeof(int)*nbr_bin);
    int i, cdf, min, d;
    /* Construct the LUT by calculating the CDF */
    cdf = 0;
    min = 0;
    i = 0;
    
    while(min == 0)
    {
        min = hist_in[i++];
    }

    d = img_size - min;
   
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

    unsigned char * img_out_buff;
    unsigned char * img_in_dev;
    int * lut_dev;

    // Allocate device memory
    cudaMalloc((void **)&img_out_buff, img_size * sizeof(unsigned char));
    cudaMalloc((void **)&img_in_dev, img_size * sizeof(unsigned char));
    cudaMalloc((void **)&lut_dev, nbr_bin * sizeof(int));

    // Host to device memory copy
    cudaMemcpy(img_in_dev, img_in, img_size, cudaMemcpyHostToDevice); // Copy image to device
    cudaMemcpy(lut_dev, lut, nbr_bin * sizeof(int), cudaMemcpyHostToDevice); // Copy lut to device

    // Check for any CUDA errors
    checkCUDAError("cudaMemcpyHostToDevice");

    // Number of blocks - if the division is not exact, the resulting number is rounded up. 
    int numBlocks = (int)ceil(img_size / (float)numThreadsPerBlock);

    dim3 dimGrid(numBlocks);
    dim3 dimBlock(numThreadsPerBlock);

    /* Get the result image */
    GetResultImageKernel<<<dimGrid, dimBlock>>>(img_out_buff, img_in_dev, lut_dev, img_size); // Kernel
    
    // Block until the device has completed
    cudaThreadSynchronize();

    // Check for any CUDA errors
    checkCUDAError("Get Result Image Kernel Execution");

    // Device to host copy
    cudaMemcpy(img_out, img_out_buff,  sizeof(unsigned char) * img_size, cudaMemcpyDeviceToHost);

    // Check for any CUDA errors
    checkCUDAError("cudaMemcpyDeviceToHost");

    // Free device memory
    cudaFree(img_in_dev);
    cudaFree(lut_dev);
    cudaFree(img_out_buff);
}

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err) );
        exit(-1);
    }
}