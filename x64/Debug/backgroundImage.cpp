#include <iostream>
#include <cstdlib>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <numeric>
#include <vector>

#include "utilities/General.h"
#include "VoxelReconstruction.h"

using namespace nl_uu_science_gmt;
using namespace std;

int main(int argc, char** argv)
{

	for(int i = 1; i < 5; i++ ){
		string camera_number = std::to_string(i);

		const int width  = 644;
		const int height = 486;
		cv::Mat avgImg = cv::Mat::zeros(height,width,CV_32FC3);


		cv::VideoCapture cap("data/cam" + camera_number + "/background.avi");

		if (!cap.isOpened()) {
			return -1;
		}
		
		double count = cap.get(cv::CAP_PROP_FRAME_COUNT);

		std::cout << "Total frames" << count << std::endl;
		//Take every 5th frame (frame_count)
		for (int frame_count = 0; frame_count < cap.get(cv::CAP_PROP_FRAME_COUNT); frame_count+=5) {
			std::cout << frame_count << std::endl;
			cap.set(cv::CAP_PROP_POS_FRAMES, frame_count);
			cv::Mat frame;
			
			if (!cap.read(frame)) {
				std::cout << "Failed to extract a frame.\n" << std::endl;
				return -1;
			}

			cv::accumulate(frame, avgImg);
			
		}

		avgImg.convertTo(avgImg, CV_8UC3, 1.0/(count/5));

		cv::imshow("Image",avgImg);
		cv::waitKey(0);	
		std::string image_path =  "data/cam" + camera_number + "/background.png"; 
		cv::imwrite(image_path, avgImg); 
        
	}
return 0;

}
