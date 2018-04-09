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
 
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <cstdlib>

using namespace std;
using namespace cv;

// Data of Mat must be continuous and gray scale, aka src can not be cut
// Or check mat if continuous before using this function
void hist_equal(const Mat &src, Mat &dst)
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

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        cout << "Usage: hist_equal <image name>" << endl;
        return -1;
    }
    Mat img_src, img_rst;
    img_src = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

    if (!img_src.data)
    {
        cout << "Can not find image file!" << endl;
        return -1;
    }

    hist_equal(img_src, img_rst);

    imwrite("images/result.jpg", img_rst);
    return 0;
}
