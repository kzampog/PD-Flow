#include <iostream>

#include <scene_flow/scene_flow.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/videoio/videoio_c.h>
#include "eigen3-hdf5.hpp"

class RGBDReader {
public:
    RGBDReader() : reader_type(RGBDReader::OPENNI_DEVICE) {}
    RGBDReader(const std::string &rgb_fmt, const std::string &depth_fmt)
            : reader_type(RGBDReader::FILE_READER),
              rgb_file_fmt(rgb_fmt),
              depth_file_fmt(depth_fmt)
    {}
    ~RGBDReader() { release(); }

    bool open() {
        if (reader_type == RGBDReader::OPENNI_DEVICE) {
            return cap_openni.open(CV_CAP_OPENNI_ASUS);
        } else if (reader_type == RGBDReader::FILE_READER) {
            return cap_depth.open(depth_file_fmt, cv::CAP_IMAGES) && cap_rgb.open(rgb_file_fmt, cv::CAP_IMAGES);
        }
        return false;
    }

    void release() {
        cap_depth.release();
        cap_rgb.release();
        cap_openni.release();
    }

    bool getFrames(cv::Mat &rgb, cv::Mat &depth) {
        bool success = true;
        if (reader_type == RGBDReader::OPENNI_DEVICE) {
            success = success && cap_openni.grab();
            success = success && cap_openni.retrieve(depth, CV_CAP_OPENNI_DEPTH_MAP);
            success = success && cap_openni.retrieve(rgb, CV_CAP_OPENNI_BGR_IMAGE);
        } else if (reader_type == RGBDReader::FILE_READER) {
            success = success && cap_depth.grab();
            success = success && cap_rgb.grab();
            success = success && cap_depth.retrieve(depth);
            success = success && cap_rgb.retrieve(rgb);
        } else {
            success = false;
        }
        return success;
    }

private:
    enum ReaderType {OPENNI_DEVICE, FILE_READER};
    ReaderType reader_type;

    std::string depth_file_fmt;
    std::string rgb_file_fmt;

    cv::VideoCapture cap_depth;
    cv::VideoCapture cap_rgb;
    cv::VideoCapture cap_openni;
};

int main(int argc, char** argv) {

    RGBDReader reader(argv[1], argv[2]);

    char buff[1000];

    std::string fmt = argv[3];

    SceneFlow sf;

    cv::Mat rgb1, depth1, rgb2, depth2;
    Eigen::MatrixXf x_flow(640,480), y_flow(640,480), z_flow(640,480);

    reader.open();
    reader.getFrames(rgb1, depth1);

    int i = 0;
    while (reader.getFrames(rgb2, depth2)) {
        std::cout << i << std::endl;

        sf.loadRGBDFrames(rgb1.data, depth1.data, rgb2.data, depth2.data, SceneFlow::BGR8, SceneFlow::DEPTH16);
        sf.computeFlow();
        sf.getFlowField(x_flow.data(), y_flow.data(), z_flow.data());

        sprintf(buff, fmt.c_str(), i);

        H5::H5File x_file(std::string(buff) + "_x.dat", H5F_ACC_TRUNC);
        EigenHDF5::save(x_file, "x_flow", x_flow.transpose());
        H5::H5File y_file(std::string(buff) + "_y.dat", H5F_ACC_TRUNC);
        EigenHDF5::save(y_file, "y_flow", y_flow.transpose());
        H5::H5File z_file(std::string(buff) + "_z.dat", H5F_ACC_TRUNC);
        EigenHDF5::save(z_file, "z_flow", z_flow.transpose());

//        writeEigenMatrixToFile2(std::string(buff) + "_x.dat", x_flow.transpose(), true);
//        writeEigenMatrixToFile2(std::string(buff) + "_y.dat", y_flow.transpose(), true);
//        writeEigenMatrixToFile2(std::string(buff) + "_z.dat", z_flow.transpose(), true);

        rgb2.copyTo(rgb1);
        depth2.copyTo(depth1);

        i++;
    }

    return 0;
}
