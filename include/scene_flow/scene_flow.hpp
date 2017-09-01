#pragma once

#include <scene_flow/pdflow_cudalib.h>
#include <vector>
//#include <opencv2/core/core.hpp>

class SceneFlow {
public:
	enum ImageType {RGB24, BGR24, GRAY8, DEPTH16, DEPTH32};

	SceneFlow();
	~SceneFlow();

	void loadRGBDFrames(const unsigned char * image1, const unsigned char * depth1, const unsigned char * image2, const unsigned char * depth2, const ImageType &image_type = GRAY8, const ImageType &depth_type = DEPTH16);
//	void loadRGBDFrames(const cv::Mat &rgb1, const cv::Mat &depth1, const cv::Mat &rgb2, const cv::Mat &depth2);

	void computeFlow();

	void getFlowField(float * x, float * y, float * z);
	void getFlowField(float * xyz);

//	void getFlowImages(cv::Mat &vx, cv::Mat &vy, cv::Mat &vz);
//	void getFlowImage(cv::Mat &flow);
//	cv::Mat getFlowVisualizationImage();

	void setRGBDImageParameters(unsigned int im_width, unsigned int im_height, float intr_fx, float intr_fy, float intr_cx, float intr_cy);
	void setCoarseToFineParameters(unsigned int max_rows, unsigned int max_cols, unsigned int num_levels, unsigned int num_fine_max_iter = 100);
	void setOptimizationParameters(float par_mu, float par_lambda_i, float par_lambda_d);

private:
	unsigned int width, height;
	float fx, fy, cx, cy;

	unsigned int rows, cols, ctf_levels, fine_max_iter;
	float mu, lambda_i, lambda_d;

	std::vector<unsigned int> max_iter;
	float g_mask[25];
	bool is_initialized;

//	cv::Mat intensity_img_buffer, depth_img_buffer;
	float *image_buffer, *depth_buffer;
	float *dx, *dy, *dz;

	CSF_cuda csf_host, *csf_device;

	void initialize();
	void cleanUp();
	void prepareRGBDImagePair(const unsigned char * image, const unsigned char * depth, const ImageType &image_type = GRAY8, const ImageType &depth_type = DEPTH16);
//	void prepareRGBDImagePair(const cv::Mat &rgb, const cv::Mat &depth);
};
