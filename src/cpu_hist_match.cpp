/**
 * Author: Liu Chaoyang
 * E-mail: chaoyanglius@gmail.com
 * 
 * histgrams equalize using C++
 * Copyright (C) 2018 Liu Chaoyang
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

// implement histogram match from target image
void hist_match(Mat &src, Mat &dst, Mat &tgt);
// implement histogram match from target histogram
void hist_match(Mat &src, Mat &dst, double hgram[]);

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        cout << "Usage: hist_equal <image name> <target image name>" << endl;
        return -1;
    }
    Mat img_src, img_rst, img_tgt;
    img_src = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    img_tgt = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);

    if (!img_src.data)
    {
        cout << "Can not find image file!" << endl;
        return -1;
    }
    double hgram[256];

    // TODO:set target histogram
    hist_match(img_src, img_rst, img_tgt);
//    hist_match(img_src, img_rst, hgram);
    bool flag = imwrite("images/result2.jpg", img_rst);    
    return 0;
}

void hist_match(Mat &src, Mat &dst, Mat &tgt)
{
    if (!tgt.isContinuous())
    {
        cout << "The source image is not continuous!" << endl;
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

void hist_match(Mat &src, Mat &dst, double hgram[])
{
    if (!src.isContinuous())
    {
        cout << "The source image is not continuous!" << endl;
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
    for (int i = 0; i < 256; ++i)
    {
        double min_value = abs(tgt_cumulative[i] - src_cumulative[0]);
        int min_index = 0;
        for (int j = 1; j < 256; ++j)
        {
            double temp = abs(tgt_cumulative[i] - src_cumulative[j]);
            /* temp = abs(src_cumulative[i]- tgt_cumulative[j]); // single map law(SML)*/
            if (temp < min_value)
            {
                min_value = temp;
                min_index = j;
            }
        }
        min_ids[i] = min_index;
    } 
    
    // map dst image
    if (!dst.data)
    {
        dst.create(src.size(), src.type());
    }
    uchar *dst_data = dst.data;
    for (unsigned int i = 0; i < rows * cols; ++i)
    {
        dst_data[i] = (uchar)min_ids[src_data[i]];    
    }
}
