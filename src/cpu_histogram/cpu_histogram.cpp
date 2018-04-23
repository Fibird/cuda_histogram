#include <opencv2/core/core.hpp>
#include <iostream>
#include <cstdlib>
#include "cpu_histogram.h"

using namespace cv;

// Data of Mat must be continuous and gray scale, aka src can not be cut
// Or check mat if continuous before using this function
void hist_equal(const Mat &src, Mat &dst)
{
    if (!src.isContinuous())
    {
        std::cout << "The source image is not continuous!" << std::endl;
        exit(EXIT_FAILURE); 
    }
    uchar *src_data = src.data;
    unsigned int rows = src.rows, cols = src.cols;

    // calculate histogram of source image
    int hist[256];
    memset(hist, 0, 256 * sizeof(int));
    for (unsigned int i = 0; i < rows * cols; ++i)
    {
       hist[src_data[i]]++; 
    }
    // normalize the histogram
    double normal[256];
    unsigned int img_size = rows * cols;
    for (int i = 0; i < 256; ++i)
    {
        normal[i] = ((double) hist[i]) / img_size;
    }
    // compute cumulative histogram
    double cumulative[256];
    double temp = 0.f;
    for (int i = 0; i < 256; ++i)
    {
        temp += normal[i];
        cumulative[i] = temp;
    }
    // round the cumulative histogram
    for (int i = 0; i < 256; ++i)
    {
        hist[i] = (int)(255.0f * cumulative[i] + 0.5);    
    }
    // map new image
    if (!dst.data)
    {
        dst.create(src.size(), src.type());
    }
    uchar *dst_data = dst.data;
    for (unsigned int i = 0; i < rows * cols; ++i)
    {
        dst_data[i] = (uchar)hist[src_data[i]];    
    }
}


void hist_match(const Mat &src, Mat &dst, const Mat &tgt)
{
    if (!tgt.isContinuous())
    {
        std::cout << "The source image is not continuous!" << std::endl;
        exit(EXIT_FAILURE); 
    }
    uchar *tgt_data = tgt.data;
    unsigned int rows = tgt.rows, cols = tgt.cols;

    // calculate histogram of source image
    int hist[256];
    memset(hist, 0, 256 * sizeof(int));
    for (unsigned int i = 0; i < rows * cols; ++i)
    {
       hist[tgt_data[i]]++; 
    }
    // normalize the histogram
    double normal[256];
    unsigned int img_size = rows * cols;
    for (int i = 0; i < 256; ++i)
    {
        normal[i] = ((double) hist[i]) / img_size;
    }
    hist_match(src, dst, normal);
}

void hist_match(const Mat &src, Mat &dst, const double hgram[])
{
    if (!src.isContinuous())
    {
        std::cout << "The source image is not continuous!" << std::endl;
        exit(EXIT_FAILURE); 
    }
    uchar *src_data = src.data;
    unsigned int rows = src.rows, cols = src.cols;

    // calculate histogram of source image
    int hist[256];
    memset(hist, 0, 256 * sizeof(int));
    for (unsigned int i = 0; i < rows * cols; ++i)
    {
       hist[src_data[i]]++; 
    }
    // normalize the histogram
    double normal[256];
    unsigned int img_size = rows * cols;
    for (int i = 0; i < 256; ++i)
    {
        normal[i] = ((double) hist[i]) / img_size;
    }
    // compute cumulative histogram
    double src_cumulative[256], tgt_cumulative[256];
    double temp1 = 0.f, temp2 = 0.f;
    for (int i = 0; i < 256; ++i)
    {
        temp1 += normal[i];
        temp2 += hgram[i];
        src_cumulative[i] = temp1;
        tgt_cumulative[i] = temp2;
    }
    
    // using group map law(GML)
    int min_ids[256];
    int start = 0, end = 0, last_start = 0, last_end = 0;
    for (int i = 0; i < 256; ++i)
    {
        double min_value = abs(tgt_cumulative[i] - src_cumulative[0]); 
        for (int j = 1; j < 256; ++j)
        {
            double temp = abs(tgt_cumulative[i] - src_cumulative[j]); 
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
                min_ids[t] = i;
            }
            last_start = start;
            last_end = end;
            start = last_end + 1;
        }
    } 
    
    if (!dst.data)
    {
        dst.create(src.size(), src.type());
    }

    // map dst image according relationship
    uchar *dst_data = dst.data;
    for (unsigned int i = 0; i < rows * cols; ++i)
    {
        dst_data[i] = (uchar)min_ids[src_data[i]];    
    }
}
