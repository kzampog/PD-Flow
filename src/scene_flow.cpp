#include <scene_flow/scene_flow.hpp>

SceneFlow::SceneFlow() {
	width = 640;
	height = 480;
	fx = 528.0;
	fy = 528.0;
	cx = 319.5;
	cy = 239.5;

	rows = 480;
	cols = 640;
	ctf_levels = 6;

	mu = 75.0;
	lambda_i = 0.04;
	lambda_d = 0.35;
}

SceneFlow::~SceneFlow() {

}
