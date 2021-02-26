/*
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
	(void)argc;
    (void)argv;

	std::cout << "Loading Images..." << std::endl;
	std::vector<cv::String> fileNames;
    cv::glob("test/*.png", fileNames, false);
    std::cout << "Images found: " << fileNames.size() << std::endl;

	cv::Size patternSize(9, 6);
    std::vector<std::vector<cv::Point2f>> imagePointList = std::vector<std::vector<cv::Point2f>>();
    std::vector<std::vector<cv::Point3f>> objectPointsList = std::vector<std::vector<cv::Point3f>>();

    int checkerBoard[2] = { 10,7 };
    int fieldSize = 115;
    std::vector<cv::Point3f> objectPoints;
    for (int i = 1; i < checkerBoard[1]; i++) {
        for (int j = 1; j < checkerBoard[0]; j++) {
            objectPoints.push_back(cv::Point3f(j * fieldSize, i * fieldSize, 0));
        }
    }
	std::cout << "Object Points: " << objectPoints << std::endl;

	cv::Size frameSize;
    cv::Matx33f cameraMatrix(cv::Matx33f::eye());
    cv::Vec<float, 5> distCoeffs(0, 0, 0, 0, 0);
    std::vector<cv::Mat> rotationVectors, translationVectors;
    std::vector<double> reprojectionErrors = std::vector<double>();

	// Detect feature points
    for (int i = 0; i < fileNames.size(); i++)
    {
		std::cout << "Number " << i << std::endl;
        //Defining object to store x and y coordinates for image points
        std::vector<std::vector<cv::Point2f>> imagePointList = std::vector<std::vector<cv::Point2f>>();
        //Defining object to store x, y, z coordinates for world points
        std::vector<std::vector<cv::Point3f>> objectPointsList = std::vector<std::vector<cv::Point3f>>();

        std::cout << fileNames[i] << std::endl;
        cv::Mat img = cv::imread(fileNames[i]);
        cv::Mat gray;
        //Convert image to grey scale to make calibration easier
        cv::cvtColor(img, gray, cv::COLOR_RGB2GRAY); 
        std::vector<cv::Point2f> imagePoints;

        bool patternFound = cv::findChessboardCorners(gray, patternSize, imagePoints, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);
		std::cout << "Pattern found " << patternFound << std::endl;

        if(patternFound){

            std::cout << "Found pattern..." << std::endl;

            frameSize = cv::Size(gray.cols, gray.rows);
            cv::cornerSubPix(gray, imagePoints,cv::Size(11,11), frameSize, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));
            //Push world coordinates to objectPoints
            objectPointsList.push_back(objectPoints);
            //Push image coordinates to imagePoints
            imagePointList.push_back(imagePoints);

            cv::drawChessboardCorners(img, patternSize, imagePoints, patternFound);
            //Calculate reprojection error
            float error = cv::calibrateCamera(objectPointsList, imagePointList, frameSize, cameraMatrix, distCoeffs, rotationVectors, translationVectors);

            std::cout << "Reprojection Error" << error << std::endl;
            reprojectionErrors.push_back( error);
        }
    }

    //Calculation of average reprojection error over all images
    double sum = std::accumulate(reprojectionErrors.begin(), reprojectionErrors.end(), 0.0);
    double mean = sum / reprojectionErrors.size();

    //Calculation of standard deviation of reprojection error
    double var = 0;
    for(int n = 0; n < reprojectionErrors.size(); n++ )
    {
      var += (reprojectionErrors[n] - mean) * (reprojectionErrors[n] - mean);
    }
    var /= reprojectionErrors.size();
    double stdev = sqrt(var);

    //Defining object to store x and y coordinates for image points
    imagePointList.clear();
    //Defining object to store x, y, z coordinates for world points
    objectPointsList.clear();

    for (int i = 0; i < fileNames.size(); i++)
    {
      //Only use images where reprojection error is smaller or equal to reprojection error mean + 1 SD
      if(reprojectionErrors[i] <= mean + stdev)
      {
        std::cout << fileNames[i] << std::endl;
        cv::Mat img = cv::imread(fileNames[i]);
        cv::Mat gray;
        //Convert image to grey scale to make calibration easier
        cv::cvtColor(img, gray, cv::COLOR_RGB2GRAY); 
        std::vector<cv::Point2f> imagePoints;

        bool patternFound = cv::findChessboardCorners(gray, patternSize, imagePoints, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);

        if(patternFound){

            std::cout << "Found pattern..." << std::endl;

            frameSize = cv::Size(gray.cols, gray.rows);
            cv::cornerSubPix(gray, imagePoints,cv::Size(11,11), frameSize, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));
            //Push world coordinates to objectPoints
            objectPointsList.push_back(objectPoints);
            //Push image coordinates to imagePoints
            imagePointList.push_back(imagePoints);

            cv::drawChessboardCorners(img, patternSize, imagePoints, patternFound);
            //Calculate reprojection error
            float error = cv::calibrateCamera(objectPointsList, imagePointList, frameSize, cameraMatrix, distCoeffs, rotationVectors, translationVectors);

            std::cout << "Reprojection Error" << error << std::endl;
            reprojectionErrors[i] += error;
        }
      }
    }

    //Calculate final camera matrix, distortion coefficients, rotation matrix and translation vector and reprojection error
    float error = cv::calibrateCamera(objectPointsList, imagePointList, frameSize, cameraMatrix, distCoeffs, rotationVectors, translationVectors);
    std::cout << "Calibration finished, Reprojection error = " << error << "\nK =\n" << cameraMatrix << "\nk=\n" << distCoeffs << std::endl;
    std::cout << "cameraMatrix : " << cameraMatrix << std::endl;
    std::cout << "distCoeffs : " << distCoeffs << std::endl;
    std::cout << "Total Reprojection Error" << error << std::endl;



	//VoxelReconstruction::showKeys();
	//VoxelReconstruction vr("data" + std::string(PATH_SEP), 4);
	//vr.run(argc, argv);

	//return EXIT_SUCCESS;


		
    

return 0;

}
*/
