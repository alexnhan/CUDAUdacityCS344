//Udacity HW 6
//Poisson Blending

/* Background
   ==========

   The goal for this assignment is to take one image (the source) and
   paste it into another image (the destination) attempting to match the
   two images so that the pasting is non-obvious. This is
   known as a "seamless clone".

   The basic ideas are as follows:

   1) Figure out the interior and border of the source image
   2) Use the values of the border pixels in the destination image 
      as boundary conditions for solving a Poisson equation that tells
      us how to blend the images.
   
      No pixels from the destination except pixels on the border
      are used to compute the match.

   Solving the Poisson Equation
   ============================

   There are multiple ways to solve this equation - we choose an iterative
   method - specifically the Jacobi method. Iterative methods start with
   a guess of the solution and then iterate to try and improve the guess
   until it stops changing.  If the problem was well-suited for the method
   then it will stop and where it stops will be the solution.

   The Jacobi method is the simplest iterative method and converges slowly - 
   that is we need a lot of iterations to get to the answer, but it is the
   easiest method to write.

   Jacobi Iterations
   =================

   Our initial guess is going to be the source image itself.  This is a pretty
   good guess for what the blended image will look like and it means that
   we won't have to do as many iterations compared to if we had started far
   from the final solution.

   ImageGuess_prev (Floating point)
   ImageGuess_next (Floating point)

   DestinationImg
   SourceImg

   Follow these steps to implement one iteration:

   1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
      Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
             else if the neighbor in on the border then += DestinationImg[neighbor]

      Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

   2) Calculate the new pixel value:
      float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
      ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]


    In this assignment we will do 800 iterations.
   */



#include "utils.h"
#include <thrust/host_vector.h>
#include "reference_calc.cpp"

__global__ void computeMaskKernel(uchar4 * d_sourceImg, int * d_mask, size_t numPixels)
{
    int pixel = threadIdx.x + blockDim.x*blockIdx.x;
    if(pixel >= numPixels)
        return;
    uchar4 imagePixel = d_sourceImg[pixel];
    if(imagePixel.x != 255 && imagePixel.y != 255 && imagePixel.z != 255)
        d_mask[pixel] = 1;
}

__global__ void computeInteriorBorderKernel(int * d_mask, int * d_interior, int * d_border, size_t numPixels, size_t numCols)
{
    int pixel = threadIdx.x + blockDim.x*blockIdx.x;
    if(pixel >= numPixels)
        return;
    if(d_mask[pixel])
    {
        if(d_mask[pixel+1] && d_mask[pixel-1] && d_mask[pixel+numCols] && d_mask[pixel-numCols])
        {
            d_interior[pixel] = 1;
        }
        else
        {
            d_border[pixel] = 1;
        }
    }
}

__global__ void separateIntoChannels(float * d_sourceRed, float * d_sourceGreen, float * d_sourceBlue, float * d_destRed, float * d_destGreen, float * d_destBlue, uchar4 * d_sourceImg, uchar4 * d_destImg, size_t numPixels)
{
    int pixel = threadIdx.x + blockDim.x*blockIdx.x;
    if(pixel >= numPixels)
        return;
    uchar4 sourcePixel = d_sourceImg[pixel];
    uchar4 destPixel = d_destImg[pixel];
    d_sourceRed[pixel] = sourcePixel.x;
    d_sourceGreen[pixel] = sourcePixel.y;
    d_sourceBlue[pixel] = sourcePixel.z;
    d_destRed[pixel] = destPixel.x;
    d_destGreen[pixel] = destPixel.y;
    d_destBlue[pixel] = destPixel.z;
}

__global__ void jacobiKernel(float * d_in, float * d_out, float * d_sourceColor, float * d_destColor, int * d_interior, int * d_border, size_t numPixels, size_t numCols)
{
    int pixel = threadIdx.x + blockDim.x*blockIdx.x;
    if(pixel >= numPixels)
        return;
    float sum1=0.f, sum2=0.f, newVal=0.f;
    if(d_interior[pixel])
    {
        if(d_interior[pixel+1])
        {
            sum1+=d_in[pixel+1];
            sum2+=d_sourceColor[pixel]-d_sourceColor[pixel+1];
        }
        else if(d_border[pixel+1])
        {
            sum1+=d_destColor[pixel+1];
            sum2+=d_sourceColor[pixel]-d_sourceColor[pixel+1];
        }
        if(d_interior[pixel-1])
        {
            sum1+=d_in[pixel-1];
            sum2+=d_sourceColor[pixel]-d_sourceColor[pixel-1];
        }
        else if(d_border[pixel-1])
        {
            sum1+=d_destColor[pixel-1];
            sum2+=d_sourceColor[pixel]-d_sourceColor[pixel-1];
        }
        if(d_interior[pixel+numCols])
        {
            sum1+=d_in[pixel+numCols];
            sum2+=d_sourceColor[pixel]-d_sourceColor[pixel+numCols];
        }
        else if(d_border[pixel+numCols])
        {
            sum1+=d_destColor[pixel+numCols];
            sum2+=d_sourceColor[pixel]-d_sourceColor[pixel+numCols];
        }
        if(d_interior[pixel-numCols])
        {
            sum1+=d_in[pixel-numCols];
            sum2+=d_sourceColor[pixel]-d_sourceColor[pixel-numCols];
        }
        else if(d_border[pixel-numCols])
        {
            sum1+=d_destColor[pixel-numCols];
            sum2+=d_sourceColor[pixel]-d_sourceColor[pixel-numCols];
        }
        newVal = (sum1+sum2)/4.f;
        d_out[pixel] = min(255.f, max(0.f, newVal));
    }
    else
    {
        d_out[pixel] = d_destColor[pixel];
    }
}

__global__ void recreateOutputImageKernel(uchar4 * d_blendedImg, float * d_red0, float * d_green0, float * d_blue0, size_t numPixels)
{
    int pixel = threadIdx.x + blockDim.x*blockIdx.x;
    if(pixel >= numPixels)
        return;
    uchar4 tempBlend;
    tempBlend.x = (char)d_red0[pixel];
    tempBlend.y = (char)d_green0[pixel];
    tempBlend.z = (char)d_blue0[pixel];
    d_blendedImg[pixel] = tempBlend;
}

void your_blend(const uchar4* const h_sourceImg,  //IN
                const size_t numRowsSource, const size_t numColsSource,
                const uchar4* const h_destImg, //IN
                uchar4* const h_blendedImg) //OUT
{

  /* To Recap here are the steps you need to implement */
  
    /* 1) Compute a mask of the pixels from the source image to be copied
        The pixels that shouldn't be copied are completely white, they
        have R=255, G=255, B=255.  Any other pixels SHOULD be copied. */
    uchar4* d_sourceImg; // array to hold h_sourceImg
    int * d_mask; // predicate array for holding which pixel should be copied
    size_t numPixels = numRowsSource*numColsSource;
    int maskKernelThreads = 1024;
    int maskKernelBlocks = numPixels/maskKernelThreads + 1;
    unsigned int sizePicture_uchar4 = sizeof(uchar4)*numPixels;
    unsigned int sizePicture_int = sizeof(int)*numPixels;
    checkCudaErrors(cudaMalloc(&d_sourceImg, sizePicture_uchar4));
    checkCudaErrors(cudaMalloc(&d_mask, sizePicture_int));
    checkCudaErrors(cudaMemcpy(d_sourceImg, h_sourceImg, sizePicture_uchar4, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(d_mask, 0, sizePicture_int));
    computeMaskKernel<<<maskKernelBlocks, maskKernelThreads>>>(d_sourceImg, d_mask, numPixels);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
    /* 2) Compute the interior and border regions of the mask.  An interior
        pixel has all 4 neighbors also inside the mask.  A border pixel is
        in the mask itself, but has at least one neighbor that isn't. */
    int * d_interior;
    int * d_border;
    checkCudaErrors(cudaMalloc(&d_interior, sizePicture_int));
    checkCudaErrors(cudaMalloc(&d_border, sizePicture_int));
    checkCudaErrors(cudaMemset(d_interior, 0, sizePicture_int));
    checkCudaErrors(cudaMemset(d_border, 0, sizePicture_int));
    computeInteriorBorderKernel<<<maskKernelBlocks, maskKernelThreads>>>(d_mask, d_interior, d_border, numPixels, numColsSource);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
    /* 3) Separate out the incoming image into three separate channels */
    uchar4 * d_destImg;
    checkCudaErrors(cudaMalloc(&d_destImg, sizePicture_uchar4));
    checkCudaErrors(cudaMemcpy(d_destImg, h_destImg, sizePicture_uchar4, cudaMemcpyHostToDevice));
    float * d_sourceRed;
    float * d_sourceGreen;
    float * d_sourceBlue;
    float * d_destRed;
    float * d_destGreen;
    float * d_destBlue;
    unsigned int sizePicture_float = sizeof(float)*numPixels;
    checkCudaErrors(cudaMalloc(&d_sourceRed, sizePicture_float));
    checkCudaErrors(cudaMalloc(&d_sourceGreen, sizePicture_float));
    checkCudaErrors(cudaMalloc(&d_sourceBlue, sizePicture_float));
    checkCudaErrors(cudaMalloc(&d_destRed, sizePicture_float));
    checkCudaErrors(cudaMalloc(&d_destGreen, sizePicture_float));
    checkCudaErrors(cudaMalloc(&d_destBlue, sizePicture_float));
    separateIntoChannels<<<maskKernelBlocks, maskKernelThreads>>>(d_sourceRed, d_sourceGreen, d_sourceBlue, d_destRed, d_destGreen, d_destBlue, d_sourceImg, d_destImg, numPixels);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    /* 4) Create two float(!) buffers for each color channel that will
        act as our guesses.  Initialize them to the respective color
        channel of the source image since that will act as our intial guess. */
    float * d_red0;
    float * d_red1;
    float * d_green0;
    float * d_green1;
    float * d_blue0;
    float * d_blue1;
    checkCudaErrors(cudaMalloc(&d_red0, sizePicture_float));
    checkCudaErrors(cudaMalloc(&d_red1, sizePicture_float));
    checkCudaErrors(cudaMalloc(&d_green0, sizePicture_float));
    checkCudaErrors(cudaMalloc(&d_green1, sizePicture_float));
    checkCudaErrors(cudaMalloc(&d_blue0, sizePicture_float));
    checkCudaErrors(cudaMalloc(&d_blue1, sizePicture_float));
    checkCudaErrors(cudaMemcpy(d_red0, d_sourceRed, sizePicture_float, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(d_green0, d_sourceGreen, sizePicture_float, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(d_blue0, d_sourceBlue, sizePicture_float, cudaMemcpyDeviceToDevice));
    
    /* 5) For each color channel perform the Jacobi iteration described 
        above 800 times. */
    for(int i=0; i<800; i++)
    {
        jacobiKernel<<<maskKernelBlocks, maskKernelThreads>>>(d_red0, d_red1, d_sourceRed, d_destRed, d_interior, d_border, numPixels, numColsSource);
        checkCudaErrors(cudaMemcpy(d_red0, d_red1, sizePicture_float, cudaMemcpyDeviceToDevice));
        jacobiKernel<<<maskKernelBlocks, maskKernelThreads>>>(d_green0, d_green1, d_sourceGreen, d_destGreen, d_interior, d_border, numPixels, numColsSource);
        checkCudaErrors(cudaMemcpy(d_green0, d_green1, sizePicture_float, cudaMemcpyDeviceToDevice));
        jacobiKernel<<<maskKernelBlocks, maskKernelThreads>>>(d_blue0, d_blue1, d_sourceBlue, d_destBlue, d_interior, d_border, numPixels, numColsSource);
        checkCudaErrors(cudaMemcpy(d_blue0, d_blue1, sizePicture_float, cudaMemcpyDeviceToDevice));
    }
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    /* 6) Create the output image by replacing all the interior pixels
        in the destination image with the result of the Jacobi iterations.
        Just cast the floating point values to unsigned chars since we have
        already made sure to clamp them to the correct range. */
    uchar4 * d_blendedImg;
    checkCudaErrors(cudaMalloc(&d_blendedImg, sizePicture_uchar4));
    recreateOutputImageKernel<<<maskKernelBlocks, maskKernelThreads>>>(d_blendedImg, d_red0, d_green0, d_blue0, numPixels);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpy(h_blendedImg, d_blendedImg, sizePicture_uchar4, cudaMemcpyDeviceToHost));
    
     /* Since this is final assignment we provide little boilerplate code to
      help you.  Notice that all the input/output pointers are HOST pointers.

      You will have to allocate all of your own GPU memory and perform your own
      memcopies to get data in and out of the GPU memory.

      Remember to wrap all of your calls with checkCudaErrors() to catch any
      thing that might go wrong.  After each kernel call do:

      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      to catch any errors that happened while executing the kernel.
  */
    
    // Freeing allocated GPU memory
    checkCudaErrors(cudaFree(d_sourceImg));
    checkCudaErrors(cudaFree(d_mask));
    checkCudaErrors(cudaFree(d_interior));
    checkCudaErrors(cudaFree(d_border));
    checkCudaErrors(cudaFree(d_sourceRed));
    checkCudaErrors(cudaFree(d_sourceBlue));
    checkCudaErrors(cudaFree(d_sourceGreen));
    checkCudaErrors(cudaFree(d_destRed));
    checkCudaErrors(cudaFree(d_destBlue));
    checkCudaErrors(cudaFree(d_destGreen));
    checkCudaErrors(cudaFree(d_destImg));
    checkCudaErrors(cudaFree(d_red0));
    checkCudaErrors(cudaFree(d_red1));
    checkCudaErrors(cudaFree(d_green0));
    checkCudaErrors(cudaFree(d_green1));
    checkCudaErrors(cudaFree(d_blue0));
    checkCudaErrors(cudaFree(d_blue1));
    checkCudaErrors(cudaFree(d_blendedImg));

  /* The reference calculation is provided below, feel free to use it
     for debugging purposes. 
   */

  /*
    uchar4* h_reference = new uchar4[srcSize];
    reference_calc(h_sourceImg, numRowsSource, numColsSource,
                   h_destImg, h_reference);

    checkResultsEps((unsigned char *)h_reference, (unsigned char *)h_blendedImg, 4 * srcSize, 2, .01);
    delete[] h_reference; */
    
}
