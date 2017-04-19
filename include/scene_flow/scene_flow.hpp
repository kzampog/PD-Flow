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

	cv::Mat intensity1, intensity2, depth1, depth2;

	CSF_cuda csf_host, *csf_device;

public:
	SceneFlow();
	~SceneFlow();

};

#endif /* SCENE_FLOW_HPP */
