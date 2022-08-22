#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/core/operations.hpp>
#include <vector>

class ColorMatching
{

	cv::Mat frame;
	cv::Mat redHistogram, greenHistogram, blueHistogram;

public:
	ColorMatching(int rows, int cols)
	{
		frame = cv::Mat::ones(rows, cols, CV_32F);
	}
};

class ColorModel
{

public:
	ColorModel()
	{
		colorMatchings = std::vector<ColorMatching*>();
	}
	~ColorModel()
	{
		for (size_t i = 0; i < colorMatchings.size(); i++)
			delete colorMatchings[i];
	}

	std::vector<ColorMatching*> colorMatchings; // size equal to num of clusters.

};