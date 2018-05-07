#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>
#include "gpu_histogram.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

__global__ void cuGetHist(uchar *src, unsigned *hist, unsigned width, unsigned height)
{
    __shared__ unsigned temp[256];
    temp[threadIdx.x] = 0;
    __syncthreads();

    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    int offset = blockDim.x * gridDim.x;

    // compute histogram
    int id = tidx;
    while (id < width * height)
    {
        atomicAdd(&(temp[src[id]]), 1);
        id += offset;
    }

    __syncthreads();
    atomicAdd(&(hist[threadIdx.x]), temp[threadIdx.x]);
}

__global__ void cuTgtMap(uchar *src, uchar *dst, unsigned *hist, double *accums, unsigned width, unsigned height)
{
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    int offset = blockDim.x * gridDim.x;

    int id = tidx;
    while (id < 256)
    {
        hist[id] = (unsigned)(255.0 * accums[id] + 0.5);
        id += offset;
    }

    __syncthreads();

    // histogram map
    id = tidx; // clear id
    while (id < width * height)
    {
        dst[id] = (uchar)hist[src[id]];
        id += offset;
    }
}

void cuHistEqual(cv::Mat &src, cv::Mat &dst)
{
    if (!src.isContinuous())
    {
        std::cout << "The source image is not continuous!" << std::endl;
        exit(EXIT_FAILURE); 
    }
    if (!src.data)
        return;

    if (src.type() != CV_8UC1)
        return;

    unsigned width = src.size().width;
    unsigned height = src.size().height;

    // calculate histogram of source image
    uchar *devSrc, *devDst;
    unsigned *devHist;
    
    CHECK(cudaMalloc((void**)&devSrc, sizeof(unsigned) * width * height));
    CHECK(cudaMemcpy(devSrc, src.data, sizeof(uchar) * width * height, cudaMemcpyHostToDevice));
    CHECK(cudaMalloc((void**)&devDst, sizeof(unsigned) * width * height));
    CHECK(cudaMalloc((void**)&devHist, sizeof(unsigned) * 256));
    CHECK(cudaMemset(devHist, 0, sizeof(unsigned) * 256));

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    int blocks = prop.multiProcessorCount;
    cuGetHist<<<blocks * 2, 256>>>(devSrc, devHist, width, height);
    CHECK(cudaDeviceSynchronize());

    unsigned *hostHist = (unsigned*)malloc(sizeof(unsigned) * 256);
    cudaMemcpy(hostHist, devHist, sizeof(unsigned) * 256, cudaMemcpyDeviceToHost);
    
    double *devAccums, *accums;
    accums = (double*)malloc(sizeof(double) * 256);
    CHECK(cudaMalloc((void**)&devAccums, sizeof(double) * 256));

    accums[0] = hostHist[0];
    for (int i = 1; i < 256; ++i)
    {
        accums[i] = accums[i - 1] + (double)hostHist[i] / (double)(width * height);
    }

    CHECK(cudaMemcpy(devAccums, accums, sizeof(double) * 256, cudaMemcpyHostToDevice));
    cuTgtMap<<<blocks * 2, 256>>>(devSrc, devDst, devHist, devAccums, width, height);
    CHECK(cudaDeviceSynchronize());

    uchar *hostDst = (uchar*)malloc(sizeof(uchar) * width * height);
    CHECK(cudaMemcpy(hostDst, devDst, sizeof(uchar) * width * height, cudaMemcpyDeviceToHost));

    dst = cv::Mat(height, width, CV_8UC1, hostDst);

    // free device data
    CHECK(cudaFree(devDst));
    CHECK(cudaFree(devSrc));
    CHECK(cudaFree(devHist));
}

void cuHistMatch(cv::Mat &src, cv::Mat &dst, cv::Mat &tgt)
{

}

void cuHistMatch(cv::Mat &src, cv::Mat &dst, double hgram[])
{

}
