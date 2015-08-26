//Udacity HW 4
//Radix Sorting

#include "reference_calc.cpp"
#include "utils.h"

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */
__global__ void hist(unsigned int * inputVals, unsigned int * histVals, unsigned int bitLocation)
{
    int index = threadIdx.x + blockDim.x*blockIdx.x;
    int bin = ((inputVals[index] & bitLocation) == 0) ? 0 : 1;
    atomicAdd(&histVals[bin],1);
}

__global__ void compact(unsigned int * inputVals, unsigned int * predicate, unsigned int bitLocation, int compactVal)
{
    int index = threadIdx.x + blockDim.x*blockIdx.x;
    int p = (inputVals[index] & bitLocation == compactVal) ? 1 : 0;
    predicate[index] = p;
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{
    unsigned int numBits = 32;
    unsigned int * histVals;
    unsigned int * predicate;
    
    int threads = 1024;
    int blocks = numElems/threads;
    checkCudaErrors(cudaMalloc((void**)&predicate), numElems*sizeof(int));
    checkCudaErrors(cudaMalloc((void**)&histVals, 2*sizeof(unsigned int));
    for(int i=0;i<numBits;i++)
    {
        unsigned int bitLoc = 1 << i;
        // 1) Histogram of number of occurences of each bit
        hist<<<blocks,threads>>>(d_inputVals, histVals, bitLoc);
        // Using compaction to compute a predicate array for 0's
        compact<<<blocks,threads>>>(d_inputVals, predicate, bitLoc, 0);
        // Exclusive scan on predicate array to get index values of 0's
        
    }
    
    checkCudaErrors(cudaFree(histVals));
    checkCudaErrors(cudaFree(predicate));
                    
}
