#ifndef SCENE_FLOW_HPP
#define SCENE_FLOW_HPP

#include <scene_flow/pdflow_cudalib.h>
#include <opencv2/core/core.hpp>

class SceneFlow {
private:
	unsigned int width, height;
	float fx, fy, cx, cy;

	unsigned int rows, cols, ctf_levels;
	float mu, lambda_i, lambda_d;

	unsigned int * max_iter;
	float g_mask[25];
	bool is_initialized;

	cv::Mat intensity1, intensity2, depth1, depth2;

	CSF_cuda csf_host, *csf_device;

public:
	SceneFlow();
	~SceneFlow();

	void initialize();

	void setRGBDImageParameters(unsigned int im_width, unsigned int im_height, float intr_fx, float intr_fy, float intr_cx, float intr_cy);
	void setCoarseToFineParameters(unsigned int max_rows, unsigned int max_cols, unsigned int num_levels, unsigned int fine_max_iter = 100);
	void setOptimizationParameters(float par_mu, float par_lambda_i, float par_lambda_d);
};

#endif /* SCENE_FLOW_HPP */
