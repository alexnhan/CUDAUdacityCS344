/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/


#include "reference_calc.cpp"
#include "utils.h"
#include <cmath>
#include <iostream>
#include <stdio.h>

#define MIN 0
#define MAX 1
__global__ void reduceMinMax(float * in, float * out, const int op)
{
    extern __shared__ float sdata[]; // initializing shared memory for each block of threads
    int index = blockDim.x*blockIdx.x + threadIdx.x; // getting absolute location
    int threadIndex = threadIdx.x; // thread index between 0 and 1023
    sdata[threadIndex] = in[index]; // move global memory into local shared memory
    __syncthreads(); // wait until all thread indices are written into shared memory
    for(int s=blockDim.x/2; s>0; s >>= 1) // reduce by half everytime loop
    {
        if(threadIndex < s)
        {
            if(op == MIN)
            {
                sdata[threadIndex] = min(sdata[threadIndex],sdata[threadIndex+s]);
            }
            else
            {
                sdata[threadIndex] = max(sdata[threadIndex],sdata[threadIndex+s]);
            }
        }
        // if s is odd, and you naively divide by 2, you will skip an operation with one of the indices due to flooring of s; ie. 3/2 = 1
        if(s != 1 && (s % 2) == 1)
            s++; // thus, just increment s by 1 to make it even again, to assure that you compare on all values
        __syncthreads(); // wait until all threads have completed; the min or max value of this block will be in sdata[0]
    }
    if(threadIndex == 0) // store the result of each block into out location
    {
        out[blockIdx.x] = sdata[0];
    }
}

__global__ void atomicComputeHist(float * vals, unsigned int * hist, float lumMin, float lumRange, int numBins)
{
    int pixel = blockDim.x*blockIdx.x+threadIdx.x;
    int bin = ((vals[pixel]-lumMin)/lumRange)*numBins;
    atomicAdd(&hist[bin],1);
}

__global__ void inclusiveScan(unsigned int * hist, int numBins) // Hillis Steele Scan
{
    int index = threadIdx.x+blockDim.x*blockIdx.x;
    for(int i=1;i<=numBins;i <<= 1)
    {
        int otherVal = index-i;
        unsigned int val = 0;
        if(otherVal >= 0)
           val=hist[otherVal];
        __syncthreads();
        if(otherVal >= 0)
            hist[index]+=val;
        __syncthreads();
    }
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */

    int numPixels = numCols*numRows; // total number of pixels in the picture
    float * device_LL;
    checkCudaErrors(cudaMalloc((void **) &device_LL, sizeof(float)*numPixels)); // allocate memory of logLuminance on GPU
    checkCudaErrors(cudaMemcpy(device_LL, d_logLuminance, sizeof(float)*numPixels,cudaMemcpyDeviceToDevice)); // move logLuminance values from CPU to GPU memory
    
    // 1) Finding min and max values
    int thread = 1024;
    dim3 threads(thread,1,1); // have 1024 threads in each block
    int block = numPixels/thread;
    dim3 blocks(block,1,1); // calculate number of blocks
    float * tempMin;
    float * tempMax;
    float * min;
    float * max;
    checkCudaErrors(cudaMalloc((void **)&tempMin, sizeof(float)*block));
    checkCudaErrors(cudaMalloc((void **)&tempMax, sizeof(float)*block));
    cudaMalloc((void **)&min, sizeof(float));
    cudaMalloc((void **)&max, sizeof(float));
    reduceMinMax<<<blocks, threads, sizeof(float)*thread>>>(device_LL, tempMin, MIN); // have min values of each block stored in tempMin
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    reduceMinMax<<<blocks, threads, sizeof(float)*thread>>>(device_LL, tempMax, MAX); // have max values of each block stored in tempMax
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
    // now compute absolute min and max of the temp values
    thread=block;
    dim3 newthreads(thread,1,1); // have 1 thread and all blocks
    dim3 newblocks(1,1,1);
    reduceMinMax<<<newblocks, newthreads, sizeof(float)*thread>>>(tempMin, min, MIN);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    reduceMinMax<<<newblocks, newthreads, sizeof(float)*thread>>>(tempMax, max, MAX);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpy(&min_logLum, min, sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&max_logLum, max, sizeof(float), cudaMemcpyDeviceToHost));
    printf("Min: %f\n",min_logLum);
    printf("Max: %f\n",max_logLum);

    // 2) Finding range
    float range = max_logLum - min_logLum;
    
    // 3) Generating histogram
    unsigned int * histCount;
    checkCudaErrors(cudaMalloc((void **)&histCount, sizeof(unsigned int)*numBins));
    thread=1024;
    dim3 histthreads(thread,1,1);
    block=numPixels/thread;
    dim3 histblocks(block,1,1);
    checkCudaErrors(cudaMemset(histCount,0,sizeof(unsigned int)*numBins)); // initialize hist vals to 0
    atomicComputeHist<<<histblocks, histthreads>>>(device_LL,histCount,min_logLum,range,numBins);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
    // 4) Performing inclusive scan to get cdf of histogram values
    thread=numBins;
    dim3 cdfThreads(thread,1,1);
    block=1;
    dim3 cdfBlocks(block,1,1);
    inclusiveScan<<<cdfBlocks, cdfThreads>>>(histCount, numBins);
    checkCudaErrors(cudaMemcpy(d_cdf,histCount,sizeof(unsigned int)*numBins,cudaMemcpyDeviceToDevice));

    // clean up allocated memory on GPU
    checkCudaErrors(cudaFree(device_LL));
    checkCudaErrors(cudaFree(tempMin));
    checkCudaErrors(cudaFree(tempMax));
    checkCudaErrors(cudaFree(histCount));
    checkCudaErrors(cudaFree(min));
    checkCudaErrors(cudaFree(max));
}
