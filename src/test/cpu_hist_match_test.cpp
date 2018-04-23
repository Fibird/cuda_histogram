#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "cpu_histogram.h"

using namespace std;
using namespace cv;

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

    hist_match(img_src, img_rst, img_tgt);
//    hist_match(img_src, img_rst, hgram);
    imwrite("images/hist_match_result.jpg", img_rst);    
    return 0;
}
