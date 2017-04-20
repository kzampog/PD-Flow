#include <scene_flow/scene_flow.hpp>

SceneFlow::SceneFlow() {
	is_initialized = false;
	max_iter = NULL;
	int v_mask[5] = {1,4,6,4,1};
	for (unsigned int i = 0; i < 5; i++) {
		for (unsigned int j = 0; j < 5; j++) {
			g_mask[i+5*j] = float(v_mask[i]*v_mask[j])/256.f;
		}
	}
	setRGBDImageParameters(640, 480, 528.0, 528.0, 319.5, 239.5);
	setCoarseToFineParameters(480, 640, 6, 100);
	setOptimizationParameters(75.0, 0.04, 0.35);
}

SceneFlow::~SceneFlow() {
	if (max_iter) {
		free(max_iter);
	}

	if (is_initialized) {
		csf_host.freeDeviceMemory();
	}
}

void SceneFlow::initialize() {


	is_initialized = true;
}

void SceneFlow::setRGBDImageParameters(unsigned int im_width, unsigned int im_height, float intr_fx, float intr_fy, float intr_cx, float intr_cy) {
	width = im_width;
	height = im_height;
	fx = intr_fx;
	fy = intr_fy;
	cx = intr_cx;
	cy = intr_cy;
}

void SceneFlow::setCoarseToFineParameters(unsigned int max_rows, unsigned int max_cols, unsigned int num_levels, unsigned int fine_max_iter) {
	rows = max_rows;
	cols = max_cols;
	ctf_levels = num_levels;
	if (max_iter) {
		free(max_iter);
	}
	max_iter = (unsigned int *)malloc(ctf_levels*sizeof(unsigned int));
	for (int i = ctf_levels - 1; i >= 0; i--) {
		if (i >= ctf_levels - 1)
			max_iter[i] = fine_max_iter;
		else
			max_iter[i] = std::max<unsigned int>(max_iter[i+1] - 15, 15);
	}
}

void SceneFlow::setOptimizationParameters(float par_mu, float par_lambda_i, float par_lambda_d) {
	mu = par_mu;
	lambda_i = par_lambda_i;
	lambda_d = par_lambda_d;
}
