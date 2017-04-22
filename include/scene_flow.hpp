#ifndef SCENE_FLOW_HPP
#define SCENE_FLOW_HPP

#include <pdflow_cudalib.h>
#include <opencv2/core/core.hpp>

class SceneFlow {
private:
	unsigned int width, height;
	float fx, fy, cx, cy;

	unsigned int rows, cols, ctf_levels, fine_max_iter;
	float mu, lambda_i, lambda_d;

	std::vector<unsigned int> max_iter;
	float g_mask[25];
	bool is_initialized;

	cv::Mat intensity_img_buffer, depth_img_buffer;
	float *dx, *dy, *dz;

	CSF_cuda csf_host, *csf_device;

	void cleanUp();
	void prepareRGBDImagePair(const cv::Mat &rgb, const cv::Mat &depth);
public:
	SceneFlow();
	~SceneFlow();

	void initialize();
	void loadRGBDFrames(const cv::Mat &rgb1, const cv::Mat &depth1, const cv::Mat &rgb2, const cv::Mat &depth2);
	void computeFlow();
	void getFlowImages(cv::Mat &vx, cv::Mat &vy, cv::Mat &vz);
	void getFlowImage(cv::Mat &flow);

	cv::Mat getFlowVisualizationImage();

	void setRGBDImageParameters(unsigned int im_width, unsigned int im_height, float intr_fx, float intr_fy, float intr_cx, float intr_cy);
	void setCoarseToFineParameters(unsigned int max_rows, unsigned int max_cols, unsigned int num_levels, unsigned int num_fine_max_iter = 100);
	void setOptimizationParameters(float par_mu, float par_lambda_i, float par_lambda_d);
};

#endif /* SCENE_FLOW_HPP */
