/*
 * Scene3DRenderer.h
 *
 *  Created on: Nov 15, 2013
 *      Author: coert
 */

#ifndef SCENE3DRENDERER_H_
#define SCENE3DRENDERER_H_

#include <opencv2/core/core.hpp>
#include <opencv2/core/operations.hpp>
#include <vector>

#define NOMINMAX
#ifdef _WIN32
#include <Windows.h>
#endif

#include "arcball.h"
#include "Camera.h"
#include "Reconstructor.h"

namespace nl_uu_science_gmt
{

// For one person on one camera.
struct ColorMatching
{
	ColorMatching(int rows, int cols, int type)
	{
		frame = cv::Mat::zeros(rows, cols, type);
		redHistogram = cv::Mat();
		greenHistogram = cv::Mat();
		blueHistogram = cv::Mat();
		personIdx = 0;
	}

	int personIdx;
	cv::Mat frame;
	cv::Mat redHistogram, greenHistogram, blueHistogram;
};

struct ColorModel
{
	ColorModel()
	{
		colorMatchings = std::vector<ColorMatching*>();
	}

	std::vector<ColorMatching*> colorMatchings; // size equal to num of clusters.
};

class Scene3DRenderer
{
	Reconstructor &m_reconstructor;          // Reference to Reconstructor
	const std::vector<Camera*> &m_cameras;  // Reference to camera's vector
	std::vector<ColorModel*> m_colormodels_offline; // Color models for each camera, offline. 
	std::vector<ColorModel*> m_colormodels_online; // Color models for each camera, online. 
	
	std::vector<std::vector<int>> &m_calibrationFrames;  // Frames used for calibration
	const int m_num;                        // Floor grid scale
	const float m_sphere_radius;            // ArcBall sphere radius

	int m_square_side_len;                   // Checkerboard square side length (mm)
	cv::Size m_board_size;                   // WxH checkerboard

	int m_width, m_height;                   // GL window size
	float m_aspect_ratio;                    // GL window frustrum aspect ratio

	/*
	 * Three vectors for the arcball that allows for manipulating the scene with
	 * the mouse like a globe
	 */
	vec m_arcball_eye;                        // arcball "eye" vector
	vec m_arcball_centre;                     // arcball "centre" vector
	vec m_arcball_up;                         // arcball "up" vector

	bool m_camera_view;                       // flag if scene viewed from a camera
	bool m_show_volume;                       // flag draw half-space edges
	bool m_show_grd_flr;                      // flag draw grid on floor
	bool m_show_cam;                          // flag draw cameras into scene
	bool m_show_org;                          // flag draw origin into scene
	bool m_show_arcball;                      // flag make arcball visible in scene
	bool m_show_info;                         // flag draw information (text) into scene
	bool m_fullscreen;                        // flag GL is full screen

	bool m_quit;                              // flag status is quit next iteration
	bool m_paused;                            // flag status is pause video
	bool m_rotate;                            // flag auto rotate GL scene

	long m_number_of_frames;                  // number of video frames
	int m_current_frame;                      // current frame index
	int m_previous_frame;                     // previously drawn frame index

	int m_current_camera;                     // number of currently selected camera view point
	int m_previous_camera;                    // number of previously selected camera view point

	int m_h_threshold;                        // Hue threshold number for background subtraction
	int m_ph_threshold;                       // Hue threshold value at previous iteration (update awareness)
	int m_s_threshold;                        // Saturation threshold number for background subtraction
	int m_ps_threshold;                       // Saturation threshold value at previous iteration (update awareness)
	int m_v_threshold;                        // Value threshold number for background subtraction
	int m_pv_threshold;                       // Value threshold value at previous iteration (update awareness)

	int targetNumOfContours = 1;		      // Target number of contours that should appear in the image.

	int preErosionElement = 2;				// 0=Off, 1=Rect, 2=Cross, 3=Ellipse.
	int preErosionSize = 1;					// Erosion kernel size.
	int erosionElement = 0;					// 0=Off, 1=Rect, 2=Cross, 3=Ellipse.
	int erosionSize = 0;					// Erosion kernel size.
	int dilationElement;					  // 0=Off, 1=Rect, 2=Cross, 3=Ellipse.
	int dilationSize;					      // Dilation kernel size.

	int m_minVoxelTrackHeight = 900;		// min height for voxels to track.
	int m_maxVoxelTrackHeight = 10000;		// max height for voxels to track.

	int m_clusterCount;
	int m_kmeans_attempts;
	std::vector<cv::Vec3i> centersCurrentFrame, centersLastFrame;

	// edge points of the virtual ground floor grid
	std::vector<std::vector<cv::Point3i*> > m_floor_grid;

	void createFloorGrid();

#ifdef _WIN32
	HDC _hDC;
#endif

public:
	Scene3DRenderer(Reconstructor &, const std::vector<Camera*> &);
	virtual ~Scene3DRenderer();
	void FindClusters(cv::Mat& labels, cv::Mat& centers, std::vector<cv::Vec3i>& coords);
	void PlotHistogram(int histSize, cv::Mat& b_hist, cv::Mat& g_hist, cv::Mat& r_hist, int histIdx, cv::Mat frame);
	void UpdateColorModelFrames(int histIdx, bool online, cv::Mat& labels, cv::Mat& labelLookup);
	void UpdateHistograms(int histIdx, bool online);
	void ApplyThresholds(std::vector<cv::Mat>& channels, nl_uu_science_gmt::Camera* camera, cv::Mat& foreground, int ht, int st, int vt);
	void processForeground(Camera*);
	bool processFrame();
	void processTracking();
	void setCamera(int);
	void setTopView();

	const std::vector<Camera*>& getCameras() const
	{
		return m_cameras;
	}

	bool isCameraView() const
	{
		return m_camera_view;
	}

	void setCameraView(
			bool cameraView)
	{
		m_camera_view = cameraView;
	}

	int getCurrentCamera() const
	{
		return m_current_camera;
	}

	void setCurrentCamera(
			int currentCamera)
	{
		m_current_camera = currentCamera;
	}

	bool isShowArcball() const
	{
		return m_show_arcball;
	}

	void setShowArcball(
			bool showArcball)
	{
		m_show_arcball = showArcball;
	}

	bool isShowCam() const
	{
		return m_show_cam;
	}

	void setShowCam(
			bool showCam)
	{
		m_show_cam = showCam;
	}

	bool isShowGrdFlr() const
	{
		return m_show_grd_flr;
	}

	void setShowGrdFlr(
			bool showGrdFlr)
	{
		m_show_grd_flr = showGrdFlr;
	}

	bool isShowInfo() const
	{
		return m_show_info;
	}

	void setShowInfo(
			bool showInfo)
	{
		m_show_info = showInfo;
	}

	bool isShowOrg() const
	{
		return m_show_org;
	}

	void setShowOrg(
			bool showOrg)
	{
		m_show_org = showOrg;
	}

	bool isShowVolume() const
	{
		return m_show_volume;
	}

	void setShowVolume(
			bool showVolume)
	{
		m_show_volume = showVolume;
	}

	bool isShowFullscreen() const
	{
		return m_fullscreen;
	}

	void setShowFullscreen(
			bool showFullscreen)
	{
		m_fullscreen = showFullscreen;
	}

	int getCurrentFrame() const
	{
		return m_current_frame;
	}

	void setCurrentFrame(
			int currentFrame)
	{
		m_current_frame = currentFrame;
	}

	bool isPaused() const
	{
		return m_paused;
	}

	void setPaused(
			bool paused)
	{
		m_paused = paused;
	}

	bool isRotate() const
	{
		return m_rotate;
	}

	void setRotate(
			bool rotate)
	{
		m_rotate = rotate;
	}

	long getNumberOfFrames() const
	{
		return m_number_of_frames;
	}

	void setNumberOfFrames(
			long numberOfFrames)
	{
		m_number_of_frames = numberOfFrames;
	}

	bool isQuit() const
	{
		return m_quit;
	}

	void setQuit(
			bool quit)
	{
		m_quit = quit;
	}

	int getPreviousFrame() const
	{
		return m_previous_frame;
	}

	void setPreviousFrame(
			int previousFrame)
	{
		m_previous_frame = previousFrame;
	}

	int getHeight() const
	{
		return m_height;
	}

	int getWidth() const
	{
		return m_width;
	}

	void setSize(
			int w, int h, float a)
	{
		m_width = w;
		m_height = h;
		m_aspect_ratio = a;
	}

	const vec& getArcballCentre() const
	{
		return m_arcball_centre;
	}

	const vec& getArcballEye() const
	{
		return m_arcball_eye;
	}

	const vec& getArcballUp() const
	{
		return m_arcball_up;
	}

	float getSphereRadius() const
	{
		return m_sphere_radius;
	}

	float getAspectRatio() const
	{
		return m_aspect_ratio;
	}

	const std::vector<std::vector<cv::Point3i*> >& getFloorGrid() const
	{
		return m_floor_grid;
	}

	int getNum() const
	{
		return m_num;
	}

	Reconstructor& getReconstructor() const
	{
		return m_reconstructor;
	}

#ifdef _WIN32
	HDC getHDC() const
	{
		return _hDC;
	}

	void setHDC(const HDC hDC)
	{
		_hDC = hDC;
	}
#endif

	int getPreviousCamera() const
	{
		return m_previous_camera;
	}

	int getHThreshold() const
	{
		return m_h_threshold;
	}

	int getSThreshold() const
	{
		return m_s_threshold;
	}

	int getVThreshold() const
	{
		return m_v_threshold;
	}

	int getPHThreshold() const
	{
		return m_ph_threshold;
	}

	int getPSThreshold() const
	{
		return m_ps_threshold;
	}

	int getPVThreshold() const
	{
		return m_pv_threshold;
	}

	void setPHThreshold(
			int phThreshold)
	{
		m_ph_threshold = phThreshold;
	}

	void setPSThreshold(
			int psThreshold)
	{
		m_ps_threshold = psThreshold;
	}

	void setPVThreshold(
			int pvThreshold)
	{
		m_pv_threshold = pvThreshold;
	}

	void setHThreshold(
			int threshold)
	{
		m_h_threshold = threshold;
	}

	void setSThreshold(
			int threshold)
	{
		m_s_threshold = threshold;
	}

	void setVThreshold(
			int threshold)
	{
		m_v_threshold = threshold;
	}

	const cv::Size& getBoardSize() const
	{
		return m_board_size;
	}

	int getSquareSideLen() const
	{
		return m_square_side_len;
	}
};

} /* namespace nl_uu_science_gmt */

#endif /* SCENE3DRENDERER_H_ */
