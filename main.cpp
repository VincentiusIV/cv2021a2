

#include <cstdlib>
#include <string>

#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>

#include "utilities/General.h"
#include "VoxelReconstruction.h"

using namespace nl_uu_science_gmt;

int main(int argc, char** argv)
{
	cv::Mat test(100, 100, CV_8U);
	test.at<uchar>(0,0);

	VoxelReconstruction::showKeys();
	VoxelReconstruction vr("data" + std::string(PATH_SEP), 4);
	vr.run(argc, argv);

	return EXIT_SUCCESS;
}
