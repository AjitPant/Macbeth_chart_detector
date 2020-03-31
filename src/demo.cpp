
#include "../include/utils.hpp"

#include<iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"

using namespace cv;
using namespace std;


int main(int argc, char **argv)
{
    if (argc != 2)
    {
        cerr << "Must input the path of the input image. Ex: ./demo image.jpg" << endl;
        return -1;
    }

    // Read input image
    Mat src = imread(argv[1]);
    if(!src.data)                              
    {
       cout<<"Could not open or find the image" << std::endl ;
       return -1;
    }

    vector<vector<Point2f>> chart_centers;

    find_charts(src, chart_centers);

    for(auto centers:chart_centers)
    {


        for(Point2f center:centers)
            circle(src, center, 5,Scalar(255, 255, 255), -1);


    }
    namedWindow("output", WINDOW_NORMAL);
    imshow("output",src);
    waitKey(0);

}


