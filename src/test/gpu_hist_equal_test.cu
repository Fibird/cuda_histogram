#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include "gpu_histogram.h"

using namespace std;
using namespace cv;

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

    cuHistEqual(img_src, img_rst);

    imwrite("result/gpuEqualResult.jpg", img_rst);
    return 0;
}
