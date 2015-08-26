//Udacity HW 4
//Radix Sorting

#include "reference_calc.cpp"
#include "utils.h"
#include <iostream>
using namespace std;
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
__global__ void hist(unsigned int * inputVals, unsigned int * histVals, unsigned int bitLocation, int numElems)
{
    int index = threadIdx.x + blockDim.x*blockIdx.x;
    if(index >= numElems)
        return;
    int bin = ((inputVals[index] & bitLocation) == 0) ? 0 : 1;
    atomicAdd(&histVals[bin],1);
}

__global__ void pred(unsigned int * inputVals, unsigned int * predicate, unsigned int bitLocation, int compactVal, int numElems)
{
    int index = threadIdx.x + blockDim.x*blockIdx.x;
    if(index >= numElems)
        return;
    int p = ((inputVals[index] & bitLocation) == compactVal) ? 1 : 0;
    __syncthreads();
    predicate[index] = p;
}

__global__ void exclusiveScan(unsigned int * sPredicate, int numElems)
{
    // Sum at current index is the sum of all elements preceding it
    int index = threadIdx.x + blockDim.x*blockIdx.x;
    if(index >= numElems)
        return;
    int val;
    if(index != 0)
    {
        val = sPredicate[index-1];
        for(int i=index-2;i>=0;i--)
        {
            val += sPredicate[i];
        }
        __syncthreads();
    }
    __syncthreads();
    if(index == 0)
        sPredicate[index] = 0;
    else
        sPredicate[index] = val;
    
    // Tried to implement Blelloch Scan, but it only worked for certain arrays.
    /*
    for(int i=1;i<numElems;i <<= 1) // reduce
    {
        int otherVal = index - i;
        int val;
        if(otherVal >= 0 && (index%2)==1 && (((otherVal+1)/i)%2) == 1)
            val = predicate[otherVal];
        __syncthreads();
        if(otherVal >= 0 && (index%2)==1 && (((otherVal+1)/i)%2) == 1)
            predicate[index]+=val;
        __syncthreads();
    }
    if(index == (numElems-1))
        predicate[index] = 0; // reset last element to identity
    __syncthreads();
    for(int i=numElems/2; i>0; i >>= 1) // downstream
    {
        int otherVal = index - i;
        int L;
        int LplusR;
        if(otherVal >= 0 && (index%2)==1 && (((otherVal+1)/i)%2) == 1)
        {
            L = predicate[index];
            LplusR = predicate[index]+predicate[otherVal];
        }
        __syncthreads();
        if(otherVal >= 0 && (index%2)==1 && (((otherVal+1)/i)%2) == 1)
        {
            predicate[index] = LplusR;
            predicate[otherVal] = L;
        }
        __syncthreads();
    }*/
}

__global__ void move(unsigned int * inputVals, unsigned int * inputPos, unsigned int * outputVals, unsigned int * outputPos, unsigned int * predicate, unsigned int * scannedPredicate, unsigned int * histVals, int movingOnes, int numElems)
{
    int index = threadIdx.x + blockDim.x*blockIdx.x;
    if(index >= numElems)
        return;
    int HV = histVals[0];
    __syncthreads();
    if(HV != numElems) {
        if(predicate[index]==1)
        {
            int outIdx;
            //int outIdx = ((movingOnes==1) ? (scannedPredicate[index] + histVals[0]) : scannedPredicate[index]);
            if(movingOnes == 0)
                outIdx = scannedPredicate[index];
            else
                outIdx = scannedPredicate[index] + HV;
            int inVal = inputVals[index];
            int inPos = inputPos[index];
            outputVals[outIdx] = inVal;
            outputPos[outIdx] = inPos;
        }
    }
    __syncthreads();
}

void testMove(unsigned int * inVals, unsigned int * inPos, unsigned int * outVals, unsigned int * outPos)//, int numElems)
{
    int numElems = 2000;
    /*unsigned int h_a[numElems];
    unsigned int h_b[numElems];
    int a = numElems*200;
    for(int i=0;i<numElems;i++)
    {
        h_a[i] = a;
        h_b[i] = a;
        a--;
    }*/
    int aSize = sizeof(unsigned int)*numElems;  
    unsigned int * d_inputVals;
    unsigned int * d_inputPos;
    unsigned int * d_outputVals;
    unsigned int * d_outputPos;
    cudaMalloc(&d_inputVals, aSize);
    cudaMalloc(&d_inputPos, aSize);
    cudaMalloc(&d_outputVals, aSize);
    cudaMalloc(&d_outputPos, aSize);
    checkCudaErrors(cudaMemcpy(d_inputVals, inVals, aSize, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(d_inputPos, inPos, aSize, cudaMemcpyDeviceToDevice));
    unsigned int * histVals;
    unsigned int * predicate;
    unsigned int * scannedPredicate;
    cudaMalloc(&histVals, sizeof(unsigned int)*2);
    cudaMalloc(&predicate, aSize);
    cudaMalloc(&scannedPredicate, aSize);
    int threads = 1024;
    int blocks = numElems/threads + 1;
    for(int i=0;i<32;i++)
    {
        unsigned int bitLoc = 1 << i;
        // 1) Histogram of number of occurences of each bit
        checkCudaErrors(cudaMemset(histVals, 0, 2*sizeof(unsigned int)));
        hist<<<blocks,threads>>>(d_inputVals, histVals, bitLoc, numElems);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
        // Using compaction to compute a predicate array for 0's
        pred<<<blocks,threads>>>(d_inputVals, predicate, bitLoc, 0, numElems);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaMemcpy(scannedPredicate, predicate, aSize, cudaMemcpyDeviceToDevice));
        // Exclusive scan on predicate array to get index values of 0's
        exclusiveScan<<<blocks,threads>>>(scannedPredicate, numElems);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
        // Move input data into output data based on index specified in predicate array
        move<<<blocks,threads>>>(d_inputVals, d_inputPos, d_outputVals, d_outputPos, predicate, scannedPredicate, histVals, 0, numElems);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
        // Use compaction to compute predicate array for 1's
        pred<<<blocks,threads>>>(d_inputVals, predicate, bitLoc, bitLoc, numElems);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaMemcpy(scannedPredicate, predicate,aSize, cudaMemcpyDeviceToDevice));
        // Exclusive scan on predicate array to get index values of 1's
        exclusiveScan<<<blocks,threads>>>(scannedPredicate, numElems);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
        // Move input data into output data based on index specified in predicate array plus offset from 0's
        move<<<blocks,threads>>>(d_inputVals, d_inputPos, d_outputVals, d_outputPos, predicate, scannedPredicate, histVals, 1, numElems);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
        // Copy output values into input values to update the sorted list
        checkCudaErrors(cudaMemcpy(d_inputVals, d_outputVals, aSize, cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpy(d_inputPos, d_outputPos, aSize, cudaMemcpyDeviceToDevice));
    }
    unsigned int pv[numElems];
    cudaMemcpy(pv, d_outputVals, aSize, cudaMemcpyDeviceToHost);
    for(int i=0;i<numElems;i++)
        cout << pv[i] << endl;
    
    checkCudaErrors(cudaMemcpy(outVals, d_outputVals, aSize, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(outPos, d_outputPos, aSize, cudaMemcpyDeviceToDevice));
    cudaFree(d_inputVals);
    cudaFree(d_inputPos);
    cudaFree(d_outputVals);
    cudaFree(d_outputPos);
    cudaFree(histVals);
    cudaFree(scannedPredicate);
    cudaFree(predicate);
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{
    testMove(d_inputVals, d_inputPos, d_outputVals, d_outputPos);//, numElems);
}
