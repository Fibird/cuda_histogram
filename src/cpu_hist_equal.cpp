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

using namespace std;
using namespace cv;

void hist_equal(const Mat src, Mat dst)
{
    // TODO:histgram equalize codes
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        cout << "Usage: hist_equal <image name>" << endl;
        return -1;
    }
    Mat img_src, img_rst;
    img_src = imread(argv[1], CV_LOAD_IMAGE_COLOR);

    if (!img_src.data)
    {
        cout << "Can not find image file!" << endl;
        return -1;
    }

    hist_equal(img_src, img_rst);

    imwrite("image/result.jpg", img_rst);

    return 0;
}
