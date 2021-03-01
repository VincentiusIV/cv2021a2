/*
 * Scene3DRenderer.cpp
 *
 *  Created on: Nov 15, 2013
 *      Author: coert
 */

#include "Scene3DRenderer.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <stddef.h>
#include <string>
#include <iostream>

#include "../utilities/General.h"

using namespace std;
using namespace cv;

namespace nl_uu_science_gmt
{

/**
 * Constructor
 * Scene properties class (mostly called by Glut)
 */
Scene3DRenderer::Scene3DRenderer(
		Reconstructor &r, const vector<Camera*> &cs) :
				m_reconstructor(r),
				m_cameras(cs),
				m_num(4),
				m_sphere_radius(1850)
{
	m_width = 640;
	m_height = 480;
	m_quit = false;
	m_paused = false;
	m_rotate = false;
	m_camera_view = true;
	m_show_volume = true;
	m_show_grd_flr = true;
	m_show_cam = true;
	m_show_org = true;
	m_show_arcball = false;
	m_show_info = true;
	m_fullscreen = false;

	// Read the checkerboard properties (XML)
	FileStorage fs;
	fs.open(m_cameras.front()->getDataPath() + ".." + string(PATH_SEP) + General::CBConfigFile, FileStorage::READ);
	if (fs.isOpened())
	{
		fs["CheckerBoardWidth"] >> m_board_size.width;
		fs["CheckerBoardHeight"] >> m_board_size.height;
		fs["CheckerBoardSquareSize"] >> m_square_side_len;
	}
	fs.release();

	m_current_camera = 0;
	m_previous_camera = 0;

	m_number_of_frames = m_cameras.front()->getFramesAmount();
	m_current_frame = 0;
	m_previous_frame = -1;

	const int H = 0;
	const int S = 0;
	const int V = 0;
	m_h_threshold = H;
	m_ph_threshold = H;
	m_s_threshold = S;
	m_ps_threshold = S;
	m_v_threshold = V;
	m_pv_threshold = V;

	createTrackbar("Frame", VIDEO_WINDOW, &m_current_frame, m_number_of_frames - 2);
	createTrackbar("H", VIDEO_WINDOW, &m_h_threshold, 255);
	createTrackbar("S", VIDEO_WINDOW, &m_s_threshold, 255);
	createTrackbar("V", VIDEO_WINDOW, &m_v_threshold, 255);
	const int maxElement = 3, maxSize = 21;
	createTrackbar("Erosion Element", VIDEO_WINDOW, &erosionElement, maxElement);
	createTrackbar("Erosion Kernel Size", VIDEO_WINDOW, &erosionSize, maxSize);
	createTrackbar("Dilation Element", VIDEO_WINDOW, &dilationElement, maxElement);
	createTrackbar("Dilation Kernel Size", VIDEO_WINDOW, &dilationSize, maxSize);

	// TODO: Automatic threshhold calculation
	// Andrea
	// Two suggestions: 
	// 1.	Assume that a good segmentation has little noise(few isolated white pixels, few isolated black pixels).
	// 		Implement a function that tries out values and optimizes the amount of noise. Be creative in how you approach this. Perhaps the functions erode and dilate can help.
	// 2.	Make a manual segmentation of a frame into foregroundand background(e.g.in Paint).
	//		Then implement a function that finds the optimal thresholds by comparing the algorithmís output to the manual segmentation.The XOR function might be of use here.
	
	createFloorGrid();
	setTopView();

}

/**
 * Deconstructor
 * Free the memory of the floor_grid pointer vector
 */
Scene3DRenderer::~Scene3DRenderer()
{
	for (size_t f = 0; f < m_floor_grid.size(); ++f)
		for (size_t g = 0; g < m_floor_grid[f].size(); ++g)
			delete m_floor_grid[f][g];
}

/**
 * Process the current frame on each camera
 */
bool Scene3DRenderer::processFrame()
{
	for (size_t c = 0; c < m_cameras.size(); ++c)
	{
		if (m_current_frame == m_previous_frame + 1)
		{
			m_cameras[c]->advanceVideoFrame();
		}
		else if (m_current_frame != m_previous_frame)
		{
			m_cameras[c]->getVideoFrame(m_current_frame);
		}
		assert(m_cameras[c] != NULL);
		processForeground(m_cameras[c]);
	}



	return true;
}

double calculateNoise(Mat img) {
	double noise = 0.0;
	
	for (int y = 0; y < img.rows; y++)
	{
		for (int x = 0; x < img.cols; x++)
		{
			auto pixel = img.at<uchar>(y, x);

			int equalNeighbourCount = 0;
			// Go over neighbours of current pixel.
			for (int i = -1; i < 2; i++)
			{
				for (int j = -1; j < 2; j++)
				{
					if (i == 0 && j == 0)
						continue;
					int xi = x + i;
					int yj = y + j;
					if (xi < 0 || xi >= img.cols)
						continue;
					if (yj < 0 || yj >= img.rows)
						continue;
					auto neighbour = img.at<uchar>(yj, xi);
					if (pixel == neighbour)
						++equalNeighbourCount;
				}
			}

			if (equalNeighbourCount < 5)
			{
				double pixelNoise = 1.0 - (double)equalNeighbourCount / 8.0;
				// If pixel is black, count it doubly.
				if (pixel == 0)
					pixelNoise *= 2.0;
				noise += pixelNoise;
			}
		}
	}
	return noise;
}

void Scene3DRenderer::ApplyThresholds(std::vector<cv::Mat>& channels, nl_uu_science_gmt::Camera* camera, cv::Mat& foreground, int ht, int st, int vt)
{
	Mat tmp, background;

	absdiff(channels[0], camera->getBgHsvChannels().at(0), tmp);
	threshold(tmp, foreground, ht, 255, CV_THRESH_BINARY);

	// Background subtraction S
	absdiff(channels[1], camera->getBgHsvChannels().at(1), tmp);
	threshold(tmp, background, st, 255, CV_THRESH_BINARY);
	bitwise_and(foreground, background, foreground);

	// Background subtraction V
	absdiff(channels[2], camera->getBgHsvChannels().at(2), tmp);
	threshold(tmp, background, vt, 255, CV_THRESH_BINARY);
	bitwise_or(foreground, background, foreground);
}

/**
 * Separate the background from the foreground
 * ie.: Create an 8 bit image where only the foreground of the scene is white (255)
 */
void Scene3DRenderer::processForeground(Camera* camera)
{
	assert(!camera->getFrame().empty());
	Mat hsv_image;
	cvtColor(camera->getFrame(), hsv_image, CV_BGR2HSV);  // from BGR to HSV color space

	vector<Mat> channels;
	split(hsv_image, channels);  // Split the HSV-channels for further analysis

	// Background subtraction H
	static float lastNoise = 100000000000000000;
	const int MAX_ITER = 0;
	RNG rng;

	for (int i = 0; i < MAX_ITER; i++)
	{
		Mat foreground;
		// 1. set hsv thresholds to random values, dont set these to above 230 so that we dont get a completely black image.
		int ht = rng.uniform(0, 230), st = rng.uniform(0, 230), vt = rng.uniform(0, 230);

		// 2. try them out
		ApplyThresholds(channels, camera, foreground, ht, st, vt);
		
		// 3. check noise, see if its lower than noise current hsv thresholds
		float noise = calculateNoise(foreground);

		if (noise < lastNoise)
		{
			lastNoise = noise;
			m_h_threshold = ht;
			m_s_threshold = st;
			m_v_threshold = vt;
			cout << "Found better thresholds h:" << ht << ",s:" << st << "v:" << vt << endl;
		}
	}

	Mat foreground;
	ApplyThresholds(channels, camera, foreground, m_h_threshold, m_s_threshold, m_v_threshold);

	// Post process the foreground image

	// Apply erosion/dilation. Either can be turned off by setting element to 0.
	if (erosionElement != 0)
	{
		int erosionType = (erosionElement == 1) ? MORPH_RECT : ((erosionElement == 2) ? MORPH_CROSS : MORPH_ELLIPSE);
		Mat kernel = getStructuringElement(erosionType, Size(2 * erosionSize + 1, 2 * erosionSize + 1), Point(erosionSize, erosionSize));
		erode(foreground, foreground, kernel);
	}

	if (dilationElement != 0)
	{
		int dilationType = (dilationElement == 1) ? MORPH_RECT : ((dilationElement == 2) ? MORPH_CROSS : MORPH_ELLIPSE);
		Mat kernel = getStructuringElement(dilationType, Size(2 * dilationSize + 1, 2 * dilationSize + 1), Point(dilationSize, dilationSize));
		dilate(foreground, foreground, kernel);
	}

	// TODO: Post-processing: blob detection or Graph cuts (Seam finding) could work.

	camera->setForegroundImage(foreground);
}

/**
 * Set currently visible camera to the given camera id
 */
void Scene3DRenderer::setCamera(
		int camera)
{
	m_camera_view = true;

	if (m_current_camera != camera)
	{
		m_previous_camera = m_current_camera;
		m_current_camera = camera;
		m_arcball_eye.x = m_cameras[camera]->getCameraPlane()[0].x;
		m_arcball_eye.y = m_cameras[camera]->getCameraPlane()[0].y;
		m_arcball_eye.z = m_cameras[camera]->getCameraPlane()[0].z;
		m_arcball_up.x = 0.0f;
		m_arcball_up.y = 0.0f;
		m_arcball_up.z = 1.0f;
	}
}

/**
 * Set the 3D scene to bird's eye view
 */
void Scene3DRenderer::setTopView()
{
	m_camera_view = false;
	if (m_current_camera != -1)
		m_previous_camera = m_current_camera;
	m_current_camera = -1;

	m_arcball_eye = vec(0.0f, 0.0f, 10000.0f);
	m_arcball_centre = vec(0.0f, 0.0f, 0.0f);
	m_arcball_up = vec(0.0f, 1.0f, 0.0f);
}

/**
 * Create a LUT for the floor grid
 */
void Scene3DRenderer::createFloorGrid()
{
	const int size = m_reconstructor.getSize() / m_num;
	const int z_offset = 3;

	// edge 1
	vector<Point3i*> edge1;
	for (int y = -size * m_num; y <= size * m_num; y += size)
		edge1.push_back(new Point3i(-size * m_num, y, z_offset));

	// edge 2
	vector<Point3i*> edge2;
	for (int x = -size * m_num; x <= size * m_num; x += size)
		edge2.push_back(new Point3i(x, size * m_num, z_offset));

	// edge 3
	vector<Point3i*> edge3;
	for (int y = -size * m_num; y <= size * m_num; y += size)
		edge3.push_back(new Point3i(size * m_num, y, z_offset));

	// edge 4
	vector<Point3i*> edge4;
	for (int x = -size * m_num; x <= size * m_num; x += size)
		edge4.push_back(new Point3i(x, -size * m_num, z_offset));

	m_floor_grid.push_back(edge1);
	m_floor_grid.push_back(edge2);
	m_floor_grid.push_back(edge3);
	m_floor_grid.push_back(edge4);
}

} /* namespace nl_uu_science_gmt */
