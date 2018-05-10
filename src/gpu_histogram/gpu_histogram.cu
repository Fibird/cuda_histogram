#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>
#include "gpu_histogram.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

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

__global__ void cuTgtMap(uchar *src, uchar *dst, unsigned *hist, unsigned width, unsigned height)
{
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    int offset = blockDim.x * gridDim.x;

    // histogram map
    int id = tidx; // clear id
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
    uchar *devSrc;
    CHECK(cudaMalloc((void**)&devSrc, sizeof(unsigned) * width * height));
    CHECK(cudaMemcpy(devSrc, src.data, sizeof(uchar) * width * height, cudaMemcpyHostToDevice));

    uchar *devDst;
    CHECK(cudaMalloc((void**)&devDst, sizeof(unsigned) * width * height));

    unsigned *devHist;
    CHECK(cudaMalloc((void**)&devHist, sizeof(unsigned) * 256));
    CHECK(cudaMemset(devHist, 0, sizeof(unsigned) * 256));

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    int blocks = prop.multiProcessorCount;
    cuGetHist<<<blocks * 2, 256>>>(devSrc, devHist, width, height);
    CHECK(cudaDeviceSynchronize());

    unsigned hostHist[256];
    cudaMemcpy(hostHist, devHist, sizeof(unsigned) * 256, cudaMemcpyDeviceToHost);
    
    double accums[256];

    accums[0] = hostHist[0];
    hostHist[0] = (int)(255.0 * accums[0] + 0.5);
    for (int i = 1; i < 256; ++i)
    {
        accums[i] = accums[i - 1] + (double)hostHist[i] / (double)(width * height);
        hostHist[i] = (int)(255.0 * accums[i] + 0.5);
    }

    CHECK(cudaMemcpy(devHist, hostHist, sizeof(unsigned) * 256, cudaMemcpyHostToDevice));
    cuTgtMap<<<blocks * 2, 256>>>(devSrc, devDst, devHist, width, height);
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
    if (!tgt.isContinuous())
    {
        std::cout << "The source image is not continuous!" << std::endl;
        exit(EXIT_FAILURE); 
    }

    if (!tgt.data)
        return;

    if (tgt.type() != CV_8UC1)
        return;

    unsigned width = tgt.size().width;
    unsigned height = tgt.size().height;
     
    // calculate histogram of source image
    uchar *devTgt;
    
    CHECK(cudaMalloc((void**)&devTgt, sizeof(unsigned) * width * height));
    CHECK(cudaMemcpy(devTgt, tgt.data, sizeof(uchar) * width * height, cudaMemcpyHostToDevice));

    unsigned *devHist;
    CHECK(cudaMalloc((void**)&devHist, sizeof(unsigned) * 256));
    CHECK(cudaMemset(devHist, 0, sizeof(unsigned) * 256));

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    int blocks = prop.multiProcessorCount;
    cuGetHist<<<blocks * 2, 256>>>(devTgt, devHist, width, height);
    CHECK(cudaDeviceSynchronize());

    unsigned tgtHist[256];
    CHECK(cudaMemcpy(tgtHist, devHist, sizeof(unsigned) * 256, cudaMemcpyDeviceToHost));
    cuHistMatch(src, dst, tgtHist, width * height);

    CHECK(cudaFree(devHist));
    CHECK(cudaFree(devTgt));
}

void cuHistMatch(cv::Mat &src, cv::Mat &dst, unsigned hgram[], unsigned hgSize)
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
    uchar *devSrc;
    
    CHECK(cudaMalloc((void**)&devSrc, sizeof(unsigned) * width * height));
    CHECK(cudaMemcpy(devSrc, src.data, sizeof(uchar) * width * height, cudaMemcpyHostToDevice));
    

    unsigned *devHist;
    CHECK(cudaMalloc((void**)&devHist, sizeof(unsigned) * 256));
    CHECK(cudaMemset(devHist, 0, sizeof(unsigned) * 256));

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    int blocks = prop.multiProcessorCount;
    cuGetHist<<<blocks * 2, 256>>>(devSrc, devHist, width, height);
    CHECK(cudaDeviceSynchronize());

    unsigned hostHist[256];
    cudaMemcpy(hostHist, devHist, sizeof(unsigned) * 256, cudaMemcpyDeviceToHost);
    
    double srcAccums[256], tgtAccums[256];
    srcAccums[0] = hostHist[0] / (width * height);
    tgtAccums[0] = hgram[0] / hgSize;
    for (int i = 1; i < 256; ++i)
    {
        srcAccums[i] = srcAccums[i - 1] + (double)hostHist[i] / (width * height);
        tgtAccums[i] = tgtAccums[i - 1] + (double)hgram[i] / hgSize; 
    }

    // using group map law(GML)
    unsigned hostMins[256];
    int start = 0, end = 0, last_start = 0, last_end = 0;
    for (int i = 0; i < 256; ++i)
    {
        double min_value = abs(tgtAccums[i] - srcAccums[0]); 
        for (int j = 1; j < 256; ++j)
        {
            double temp = abs(tgtAccums[i] - srcAccums[j]); 
            if (temp <= min_value)
            {
                min_value = temp;
                end = j;
            }
        }
        if (start != last_start || end != last_end)
        {
            for (int t = start; t <= end; ++t)
            {
                // get relationship of mapping
                hostMins[t] = i;
            }
            last_start = start;
            last_end = end;
            start = last_end + 1;
        }
    } 
    
    unsigned *devMins;
    CHECK(cudaMalloc((void**)&devMins, sizeof(unsigned) * 256));
    CHECK(cudaMemcpy(devMins, hostMins, sizeof(unsigned) * 256, cudaMemcpyHostToDevice));

    uchar *devDst;
    CHECK(cudaMalloc((void**)&devDst, sizeof(unsigned) * width * height));

    cuTgtMap<<<blocks * 2, 256>>>(devSrc, devDst, devMins, width, height);
    CHECK(cudaDeviceSynchronize());

    uchar *hostDst = (uchar*)malloc(sizeof(uchar) * width * height);
    CHECK(cudaMemcpy(hostDst, devDst, sizeof(uchar) * width * height, cudaMemcpyDeviceToHost));

    dst = cv::Mat(height, width, CV_8UC1, hostDst);

    // free device data
    CHECK(cudaFree(devDst));
    CHECK(cudaFree(devSrc));
    CHECK(cudaFree(devHist));
    CHECK(cudaFree(devMins));
}
