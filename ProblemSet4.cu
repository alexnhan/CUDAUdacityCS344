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

__global__ void exclusiveScan(unsigned int * hist, unsigned int * oldVals, int numBins, int j, int numElems) // Hillis Steele Scan
{
    int index = threadIdx.x + j*numBins;
    if(index >= numElems)
        return;
    for(int i=1;i<=numBins;i <<= 1)
    {
        int otherVal = index-i;
        unsigned int val = 0;
        if(otherVal-j*numBins >= 0)
           val=hist[otherVal];
        __syncthreads();
        if(otherVal-j*numBins >= 0)
            hist[index]+=val;
        __syncthreads();
    }
    int lastVal=0;
    if(j>0)
        lastVal = hist[j*numBins-1]+1;
    __syncthreads();
    int a = hist[index]+lastVal;
    __syncthreads();
    hist[index] = a - oldVals[index];
}

__global__ void move(unsigned int * inputVals, unsigned int * inputPos, unsigned int * outputVals, unsigned int * outputPos, unsigned int * predicate0, unsigned int * scannedPredicate0, unsigned int * predicate1, unsigned int * scannedPredicate1, unsigned int * histVals, int numElems)
{
    int index = threadIdx.x + blockDim.x*blockIdx.x;
    if(index >= numElems)
        return;
    int HV = histVals[0];
    __syncthreads();
    if(predicate0[index] == 1) {
        outputVals[scannedPredicate0[index]] = inputVals[index];
        __syncthreads();
        outputPos[scannedPredicate0[index]] = inputPos[index];
        __syncthreads();
    }
    else if(predicate1[index] == 1) {
        outputVals[scannedPredicate1[index]+HV] = inputVals[index];
        __syncthreads();
        outputPos[scannedPredicate1[index]+HV] = inputPos[index];
        __syncthreads();
    }
    __syncthreads();
}
/*void testScan()
{
    int threads=1024;
    int blocks = 4000/threads + 1;
    int a[4000];
    unsigned int *b;
    unsigned int *c;
    cudaMalloc(&b, sizeof(int)*4000);
    cudaMalloc(&c, sizeof(int)*4000);
    for(int i=0;i<4000;i++)
        a[i]=1;
    cudaMemcpy(b,a,sizeof(int)*4000,cudaMemcpyHostToDevice);
    cudaMemcpy(c,a,sizeof(int)*4000,cudaMemcpyHostToDevice);
    for(int i=0;i<blocks;i++)
        exclusiveScan<<<1,threads>>>(b,c,1024,i,4000);
    cudaMemcpy(a,b,sizeof(int)*4000,cudaMemcpyDeviceToHost);
    for(int i=0;i<4000;i++)
        cout << a[i] << endl;
}*/
void testMove(unsigned int * inVals, unsigned int * inPos, unsigned int * outVals, unsigned int * outPos)//, int numElems)
{
    int numElems = 3000;
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
    unsigned int * predicate0;
    unsigned int * scannedPredicate0;
    unsigned int * predicate1;
    unsigned int * scannedPredicate1;
    cudaMalloc(&histVals, sizeof(unsigned int)*2);
    cudaMalloc(&predicate0, aSize);
    cudaMalloc(&scannedPredicate0, aSize);
    cudaMalloc(&predicate1, aSize);
    cudaMalloc(&scannedPredicate1, aSize);
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
        pred<<<blocks,threads>>>(d_inputVals, predicate0, bitLoc, 0, numElems);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaMemcpy(scannedPredicate0, predicate0, aSize, cudaMemcpyDeviceToDevice));
        // Use compaction to compute predicate array for 1's
        pred<<<blocks,threads>>>(d_inputVals, predicate1, bitLoc, bitLoc, numElems);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaMemcpy(scannedPredicate1, predicate1,aSize, cudaMemcpyDeviceToDevice));
        for(int j=0;j<blocks;j++) {
            // Exclusive scan on predicate array to get index values of 0's
            exclusiveScan<<<1,threads>>>(scannedPredicate0, predicate0, threads, j, numElems);
            cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
            // Exclusive scan on predicate array to get index values of 1's
            exclusiveScan<<<1,threads>>>(scannedPredicate1, predicate1, threads, j, numElems);
            cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
        }
        // Move input data into output data based on index specified in predicate array
        move<<<blocks,threads>>>(d_inputVals, d_inputPos, d_outputVals, d_outputPos, predicate0, scannedPredicate0, predicate1, scannedPredicate1, histVals, numElems);
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
    cudaFree(scannedPredicate0);
    cudaFree(predicate0);
    cudaFree(scannedPredicate1);
    cudaFree(predicate1);
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{
    //testScan();
    testMove(d_inputVals, d_inputPos, d_outputVals, d_outputPos);//, numElems);
}
