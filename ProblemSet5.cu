/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/


#include "utils.h"
#include "reference.cpp"

__global__
void yourHisto(const unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               const unsigned int numVals, const unsigned int numBins)
{
  //TODO fill in this kernel to calculate the histogram
  //as quickly as possible
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    int tid = threadIdx.x;
    if(idx >= numVals)
        return;
    extern __shared__ unsigned int tempHisto[]; // shared memory for each block of threads
    tempHisto[tid] = 0; // initialize bins to 0
    __syncthreads();
    atomicAdd(&tempHisto[vals[idx]],1); // populate shared histogram for each block of threads
    __syncthreads(); // wait until all threads have contributed to the temporary histogram
    unsigned int histVal = tempHisto[tid]; // take the value at the thread index for each shared histogram
    atomicAdd(&histo[tid],histVal); // update global histogram with the value
    
  //Although we provide only one kernel skeleton,
  //feel free to use more if it will help you
  //write faster code
}

void computeHistogram(const unsigned int* const d_vals, //INPUT
                      unsigned int* const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems)
{
  //TODO Launch the yourHisto kernel
    int threads = numBins;
    int blocks  = numElems/threads + 1;
    yourHisto<<<blocks, threads, sizeof(unsigned int)*numBins>>>(d_vals, d_histo, numElems, numBins);
  //if you want to use/launch more than one kernel,
  //feel free
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  /*delete[] h_vals;
  delete[] h_histo;
  delete[] your_histo;*/
}
