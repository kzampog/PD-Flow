#include <scene_flow/scene_flow.hpp>
#include <cmath>
//#include <opencv2/imgproc/imgproc.hpp>

//#include <iostream>

//std::string cvTypeToString(int type) {
//	std::string r;
//
//	uchar depth = type & CV_MAT_DEPTH_MASK;
//	uchar chans = 1 + (type >> CV_CN_SHIFT);
//	switch ( depth ) {
//		case CV_8U:  r = "8U"; break;
//		case CV_8S:  r = "8S"; break;
//		case CV_16U: r = "16U"; break;
//		case CV_16S: r = "16S"; break;
//		case CV_32S: r = "32S"; break;
//		case CV_32F: r = "32F"; break;
//		case CV_64F: r = "64F"; break;
//		default:     r = "User"; break;
//	}
//	r += "C";
//	r += (chans+'0');
//
//	return r;
//}

void SceneFlow::cleanUp() {
    max_iter.clear();
	if (dx) free(dx);
	if (dy) free(dy);
	if (dz) free(dz);
	if (image_buffer) free(image_buffer);
	if (depth_buffer) free(depth_buffer);
	if (is_initialized) csf_host.freeDeviceMemory();
}

void SceneFlow::prepareRGBDImagePair(const unsigned char * image, const unsigned char * depth, const ImageType &image_type, const ImageType &depth_type) {
	if (image_type == SceneFlow::GRAY8) {
		for (int u = 0; u < width; u++) {
			for (int v = 0; v < height; v++) {
				image_buffer[u*height + v] = float(image[v*width + u]);
			}
		}
	} else if (image_type == SceneFlow::RGB24) {
		for (int u = 0; u < width; u++) {
			for (int v = 0; v < height; v++) {
				image_buffer[u*height + v] = 0.299f*float(image[3*v*width + 3*u + 0]) + 0.587f*float(image[3*v*width + 3*u + 1]) + 0.114f*float(image[3*v*width + 3*u + 2]);
			}
		}
	} else if (image_type == SceneFlow::BGR24) {
		for (int u = 0; u < width; u++) {
			for (int v = 0; v < height; v++) {
				image_buffer[u*height + v] = 0.114f*float(image[3*v*width + 3*u + 0]) + 0.587f*float(image[3*v*width + 3*u + 1]) + 0.299f*float(image[3*v*width + 3*u + 2]);
			}
		}
	}

	if (depth_type == SceneFlow::DEPTH16) {
		unsigned short * depth_data = (unsigned short *)depth;
		for (int u = 0; u < width; u++) {
			for (int v = 0; v < height; v++) {
				depth_buffer[u*height + v] = float(depth_data[v*width + u])/1000.0f;
			}
		}
	} else if (depth_type == SceneFlow::DEPTH32) {
		float * depth_data = (float *)depth;
		for (int u = 0; u < width; u++) {
			for (int v = 0; v < height; v++) {
				depth_buffer[u*height + v] = depth_data[v*width + u];
			}
		}
	}
}

//void SceneFlow::prepareRGBDImagePair(const cv::Mat &rgb, const cv::Mat &depth) {
//	cv::Mat intensity_tmp, depth_tmp;
//
//	if (rgb.channels() > 1) {
//		cv::cvtColor(rgb, intensity_tmp, CV_BGR2GRAY);
//	} else {
//		rgb.copyTo(intensity_tmp);
//	}
//	if (intensity_tmp.type() == CV_8U) {
//		intensity_tmp.convertTo(intensity_tmp, CV_32F);
//	}
//	intensity_img_buffer = intensity_tmp.t();
//
//	if (depth.type() == CV_16U) {
//		depth.convertTo(depth_tmp, CV_32F, 0.001);
//	} else {
//		depth.copyTo(depth_tmp);
//	}
//	depth_img_buffer = depth_tmp.t();
//}

SceneFlow::SceneFlow() {
	is_initialized = false;
	dx = NULL;
	dy = NULL;
	dz = NULL;
	image_buffer = NULL;
	depth_buffer = NULL;

	int v_mask[5] = {1,4,6,4,1};
	for (unsigned int i = 0; i < 5; i++) {
		for (unsigned int j = 0; j < 5; j++) {
			g_mask[i+5*j] = float(v_mask[i]*v_mask[j])/256.f;
		}
	}

	setRGBDImageParameters(640, 480, 528.0, 528.0, 319.5, 239.5);
	setCoarseToFineParameters(480, 640, 6, 100);
	setOptimizationParameters(75.0, 0.04, 0.35);
	initialize();
}

SceneFlow::~SceneFlow() {
	cleanUp();
}

void SceneFlow::initialize() {
	cleanUp();

	max_iter.resize(ctf_levels);
	for (int i = ctf_levels - 1; i >= 0; i--) {
		if (i >= ctf_levels - 1)
			max_iter[i] = fine_max_iter;
		else
			max_iter[i] = (max_iter[i+1]-15 > 15) ? (max_iter[i+1]-15) : 15;
	}

	dx = (float *) malloc(rows*cols*sizeof(float));
	dy = (float *) malloc(rows*cols*sizeof(float));
	dz = (float *) malloc(rows*cols*sizeof(float));

	image_buffer = (float *) malloc(width*height*sizeof(float));
	depth_buffer = (float *) malloc(width*height*sizeof(float));

	csf_host.readParameters(width, height, fx, fy, cx, cy, rows, cols, ctf_levels, g_mask, lambda_i, lambda_d, mu);
	csf_host.allocateDevMemory();

	is_initialized = true;
}

void SceneFlow::loadRGBDFrames(const unsigned char * image1, const unsigned char * depth1, const unsigned char * image2, const unsigned char * depth2, const ImageType &image_type, const ImageType &depth_type) {
	unsigned int pyr_levels = static_cast<unsigned int>(log2(float(width/cols))) + ctf_levels;

	prepareRGBDImagePair(image1, depth1, image_type, depth_type);
	csf_host.copyNewFrames(image_buffer, depth_buffer);
	csf_device = ObjectToDevice(&csf_host);
	GaussianPyramidBridge(csf_device, pyr_levels, width, height);
	BridgeBack(&csf_host, csf_device);

	prepareRGBDImagePair(image2, depth2, image_type, depth_type);
	csf_host.copyNewFrames(image_buffer, depth_buffer);
	csf_device = ObjectToDevice(&csf_host);
	GaussianPyramidBridge(csf_device, pyr_levels, width, height);
	BridgeBack(&csf_host, csf_device);
}

//void SceneFlow::loadRGBDFrames(const cv::Mat &rgb1, const cv::Mat &depth1, const cv::Mat &rgb2, const cv::Mat &depth2) {
//	unsigned int pyr_levels = static_cast<unsigned int>(log2(float(width/cols))) + ctf_levels;
//
//	prepareRGBDImagePair(rgb1, depth1);
//	csf_host.copyNewFrames((float *)(intensity_img_buffer.data), (float *)(depth_img_buffer.data));
//	csf_device = ObjectToDevice(&csf_host);
//	GaussianPyramidBridge(csf_device, pyr_levels, width, height);
//	BridgeBack(&csf_host, csf_device);
//
//	prepareRGBDImagePair(rgb2, depth2);
//	csf_host.copyNewFrames((float *)(intensity_img_buffer.data), (float *)(depth_img_buffer.data));
//	csf_device = ObjectToDevice(&csf_host);
//	GaussianPyramidBridge(csf_device, pyr_levels, width, height);
//	BridgeBack(&csf_host, csf_device);
//}

void SceneFlow::computeFlow() {
	unsigned int s;
	unsigned int cols_i, rows_i;
	unsigned int level_image;
	unsigned int num_iter;

	//For every level (coarse-to-fine)
	for (unsigned int i = 0; i < ctf_levels; i++)
	{
		s = static_cast<unsigned int>(pow(2.f,int(ctf_levels-(i+1))));
		cols_i = cols/s;
		rows_i = rows/s;
		level_image = ctf_levels - i + static_cast<unsigned int>(log2(float(width/cols))) - 1;

		//Cuda allocate memory
		csf_host.allocateMemoryNewLevel(rows_i, cols_i, i, level_image);
		//Cuda copy object to device
		csf_device = ObjectToDevice(&csf_host);
		//Assign zeros to the corresponding variables
		AssignZerosBridge(csf_device);

		//Upsample previous solution
		if (i > 0) UpsampleBridge(csf_device);

		//Compute connectivity (Rij)
		RijBridge(csf_device);
		//Compute colour and depth derivatives
		ImageGradientsBridge(csf_device);
		WarpingBridge(csf_device);
		//Compute mu_uv and step sizes for the primal-dual algorithm
		MuAndStepSizesBridge(csf_device);

		//Primal-Dual solver
		for (num_iter = 0; num_iter < max_iter[i]; num_iter++)
		{
			GradientBridge(csf_device);
			DualVariablesBridge(csf_device);
			DivergenceBridge(csf_device);
			PrimalVariablesBridge(csf_device);
		}

		//Filter solution
		FilterBridge(csf_device);
		//Compute the motion field
		MotionFieldBridge(csf_device);
		//BridgeBack to host
		BridgeBack(&csf_host, csf_device);
		//Free memory of variables associated to this level
		csf_host.freeLevelVariables();
		//Copy motion field to CPU
		csf_host.copyMotionField(dx, dy, dz);
	}
}

void SceneFlow::getFlowField(float * x, float * y, float * z) {
	for (int v = 0; v < rows; v++) {
		for (int u = 0; u < cols; u++) {
			x[cols*v + u] = dx[u*rows + v];
			y[cols*v + u] = dy[u*rows + v];
			z[cols*v + u] = dz[u*rows + v];
		}
	}
}

void SceneFlow::getFlowField(float * xyz) {
	for (int v = 0; v < rows; v++) {
		for (int u = 0; u < cols; u++) {
			xyz[cols*v*3 + u*3 + 0] = dx[u*rows + v];
			xyz[cols*v*3 + u*3 + 1] = dy[u*rows + v];
			xyz[cols*v*3 + u*3 + 2] = dz[u*rows + v];
		}
	}
}

//void SceneFlow::getFlowImages(cv::Mat &vx, cv::Mat &vy, cv::Mat &vz) {
//	cv::Mat vx_tmp(cols, rows, CV_32F, dx);
//	cv::Mat vy_tmp(cols, rows, CV_32F, dy);
//	cv::Mat vz_tmp(cols, rows, CV_32F, dz);
//	vx = vx_tmp.t();
//	vy = vy_tmp.t();
//	vz = vz_tmp.t();
//}

//void SceneFlow::getFlowImage(cv::Mat &flow) {
//	std::vector<cv::Mat> channels(3);
//	getFlowImages(channels[0], channels[1], channels[2]);
//	cv::merge(channels, flow);
//}

//cv::Mat SceneFlow::getFlowVisualizationImage() {
//	cv::Mat vis_image(rows, cols, CV_8UC3);
////	float maxmodx = 0.f, maxmody = 0.f, maxmodz = 0.f;
////	for (unsigned int v = 0; v < rows; v++) {
////		for (unsigned int u = 0; u < cols; u++) {
////			if (std::abs(dx[v + u*rows]) > maxmodx) maxmodx = std::abs(dx[v + u*rows]);
////			if (std::abs(dy[v + u*rows]) > maxmody) maxmody = std::abs(dy[v + u*rows]);
////			if (std::abs(dz[v + u*rows]) > maxmodz) maxmodz = std::abs(dz[v + u*rows]);
////		}
////	}
////	for (unsigned int v = 0; v < rows; v++) {
////		for (unsigned int u = 0; u < cols; u++) {
////			vis_image.at<cv::Vec3b>(v,u)[0] = static_cast<unsigned char>(255.f * std::abs(dx[v + u*rows])/maxmodx);
////			vis_image.at<cv::Vec3b>(v,u)[1] = static_cast<unsigned char>(255.f * std::abs(dy[v + u*rows])/maxmody);
////			vis_image.at<cv::Vec3b>(v,u)[2] = static_cast<unsigned char>(255.f * std::abs(dz[v + u*rows])/maxmodz);
////		}
////	}
//    float maxmod = 0.f;
//    for (unsigned int v = 0; v < rows; v++) {
//        for (unsigned int u = 0; u < cols; u++) {
//            if (std::abs(dx[v + u*rows]) > maxmod) maxmod = std::abs(dx[v + u*rows]);
//            if (std::abs(dy[v + u*rows]) > maxmod) maxmod = std::abs(dy[v + u*rows]);
//            if (std::abs(dz[v + u*rows]) > maxmod) maxmod = std::abs(dz[v + u*rows]);
//        }
//    }
//    for (unsigned int v = 0; v < rows; v++) {
//        for (unsigned int u = 0; u < cols; u++) {
//            vis_image.at<cv::Vec3b>(v,u)[0] = static_cast<unsigned char>(255.f * std::abs(dx[v + u*rows])/maxmod);
//            vis_image.at<cv::Vec3b>(v,u)[1] = static_cast<unsigned char>(255.f * std::abs(dy[v + u*rows])/maxmod);
//            vis_image.at<cv::Vec3b>(v,u)[2] = static_cast<unsigned char>(255.f * std::abs(dz[v + u*rows])/maxmod);
//        }
//    }
//	return vis_image;
//}

void SceneFlow::setRGBDImageParameters(unsigned int im_width, unsigned int im_height, float intr_fx, float intr_fy, float intr_cx, float intr_cy) {
	width = im_width;
	height = im_height;
	fx = intr_fx;
	fy = intr_fy;
	cx = intr_cx;
	cy = intr_cy;
	if (is_initialized) initialize();
}

void SceneFlow::setCoarseToFineParameters(unsigned int max_rows, unsigned int max_cols, unsigned int num_levels, unsigned int num_fine_max_iter) {
	rows = max_rows;
	cols = max_cols;
	ctf_levels = num_levels;
	fine_max_iter = num_fine_max_iter;
	if (is_initialized) initialize();
}

void SceneFlow::setOptimizationParameters(float par_mu, float par_lambda_i, float par_lambda_d) {
	mu = par_mu;
	lambda_i = par_lambda_i;
	lambda_d = par_lambda_d;
	if (is_initialized) initialize();
}
