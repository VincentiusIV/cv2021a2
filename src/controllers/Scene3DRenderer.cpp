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
				m_sphere_radius(1850),
				m_colormodels(std::vector<ColorModel>())
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

	const int H = 10;
	const int S = 10;
	const int V = 100;
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
	const int maxElement = 2, maxSize = 21;
	createTrackbar("Pre Erode Element", VIDEO_WINDOW, &preErosionElement, maxElement);
	createTrackbar("Pre Erode Size", VIDEO_WINDOW, &preErosionSize, maxSize);
	createTrackbar("Erosion Element", VIDEO_WINDOW, &erosionElement, maxElement);
	createTrackbar("Erosion Kernel Size", VIDEO_WINDOW, &erosionSize, maxSize);
	createTrackbar("Dilation Element", VIDEO_WINDOW, &dilationElement, maxElement);
	createTrackbar("Dilation Kernel Size", VIDEO_WINDOW, &dilationSize, maxSize);

	createFloorGrid();
	setTopView();



	// TODO: Cluster voxels, all voxels should have a label.
	createColorModels();
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

	// TODO: Online Phase

	// Go over voxels belonginng to the same group.
	// Make a histogram of colors of those.
	// Compare distance between existing color models, pick closest one.

	return true;
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

void Scene3DRenderer::createColorModels()
{
	// Assumes visible voxels are already labeled.
	// Go over all voxels, separate them into lists according to label.
		// Ignore voxels below certain height to ignore trousers.
	// Make color histogram for each group of voxels.
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
	static float prevNoise = 1000000000000000;
	const int MAX_ITER = 0;
	static RNG rng;
	bool foundBetter = false;
	for (int i = 0; i < MAX_ITER; i++)
	{
		Mat foreground;
		// 1. set hsv thresholds to random values...
		// - hue & saturation thresholds greater than 100 tend to remove more detail from the foreground than remove noise.
		//   its fine to keep some noise outside the silhoutte since we can easily remove this with erosion afterwards.
		// - value thresholds greater than 100 (in combination with medium/high hue/saturation threshhold) 
		//   have a good chance at wiping the entire image, clearly undesirable.
		int ht = rng.uniform(0, 80), st = rng.uniform(0, 80), vt = rng.uniform(0, 80);
		// 2. try out random thresholds.
		ApplyThresholds(channels, camera, foreground, ht, st, vt);
		// 3. check , see if its lower than noise current hsv thresholds
		double noise = 0.0;
		// CalculateNoise(foreground, noise); // Not a great estimator for voxel reconstruction, see report.
		// Instead, calculate noise based on the amount of contours

		// 4. Erode temporarily to remove most noise outside silhoutte and exaggerate noise inside.
		int erosionType = (preErosionElement == 0) ? MORPH_RECT : ((preErosionElement == 1) ? MORPH_CROSS : MORPH_ELLIPSE);
		Mat kernel = getStructuringElement(erosionType, Size(2 * preErosionSize + 1, 2 * preErosionSize + 1), Point(preErosionSize, preErosionSize));
		erode(foreground, foreground, kernel);
		// 5. Find contours.
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		findContours(foreground, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
		noise = contours.size();
		// 6. If lower noise, use it.
		if (noise < prevNoise)
		{
			prevNoise = noise;
			m_h_threshold = ht;
			m_s_threshold = st;
			m_v_threshold = vt;
			cout << "Found better thresholds h:" << m_h_threshold << ",s:" << m_s_threshold << ",v:" << m_v_threshold << endl;
			foundBetter = true;
			Mat drawing = Mat::zeros(foreground.size(), CV_8UC3);
			for (size_t i = 0; i < contours.size(); i++)
			{
				Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
				drawContours(drawing, contours, (int)i, color, 2, LINE_8, hierarchy, 0);
			}    
			imshow("Contours", drawing);
		}
	}

	Mat foreground;
	ApplyThresholds(channels, camera, foreground, m_h_threshold, m_s_threshold, m_v_threshold);

	// Find n draw contours

	// Apply erosion/dilation. Either can be turned off by setting element to 0.
	int erosionType = (erosionElement == 0) ? MORPH_RECT : ((erosionElement == 1) ? MORPH_CROSS : MORPH_ELLIPSE);
	Mat erodeKernel = getStructuringElement(erosionType, Size(2 * erosionSize + 1, 2 * erosionSize + 1), Point(erosionSize, erosionSize));
	erode(foreground, foreground, erodeKernel);

	int dilationType = (dilationElement == 0) ? MORPH_RECT : ((dilationElement == 1) ? MORPH_CROSS : MORPH_ELLIPSE);
	Mat dilateKernel = getStructuringElement(dilationType, Size(2 * dilationSize + 1, 2 * dilationSize + 1), Point(dilationSize, dilationSize));
	dilate(foreground, foreground, dilateKernel);

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
