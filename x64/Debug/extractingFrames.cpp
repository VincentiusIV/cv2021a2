/*
 //* 
 //* Extracting Frames From Background.avi for 4 Cameras
 //* 
#include <iostream>
#include <cstdlib>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sstream>

#include "utilities/General.h"
#include "VoxelReconstruction.h"

using namespace nl_uu_science_gmt;
using namespace std;

int main(int argc, char** argv)
{
	for(int i = 1; i < 5; i++ ){

		string camera_number = std::to_string(i);

		cv::VideoCapture cap("data/cam" + camera_number + "/background.avi");

		if (!cap.isOpened()) {
			return -1;
		}

		//Take every 5th frame (frame_count)
		for (int frame_count = 0; frame_count < cap.get(cv::CAP_PROP_FRAME_COUNT); frame_count+=5) {
			cap.set(cv::CAP_PROP_POS_FRAMES, frame_count);
			cv::Mat frame;
			if (!cap.read(frame)) {
				std::cout << "Failed to extract a frame.\n" << std::endl;
				return -1;
				}
			std::string image_path =  "background_cam" + camera_number + "/frame" + std::to_string(frame_count) + ".png"; 
			cv::imwrite(image_path, frame); 
			
		}
	}

return 0;

}*/

