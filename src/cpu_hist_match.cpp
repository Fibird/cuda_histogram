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
    double hgram[256];

    // TODO:set target histogram

    hist_match(img_src, img_rst, hgram);

    imwrite("images/result.jpg", img_rst);    
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
   // TODO:implement histogram match from target histogram 
}


