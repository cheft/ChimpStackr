#ifndef ALGORITHM_H
#define ALGORITHM_H
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <fftw3.h> 
#include "NumCpp.hpp"
#include <string>

cv::Mat get_apofield(const cv::Size &shape, int aporad);
cv::Point argmax2D(const cv::Mat &array);
cv::Mat minimum_filter(const cv::Mat &array, int size);
cv::Point2d argmax_ext(const cv::Mat& array, double exponent);
cv::Mat get_subarr(const cv::Mat& array, const cv::Point& center, int rad);
double get_success(const cv::Mat& array, const cv::Point2d& coord, int radius = 2);
cv::Point2d interpolate(const cv::Mat& array, const cv::Point2d& rough, int rad = 2);
std::pair<cv::Point2d, double> argmax_translation(cv::Mat array, int filter_pcorr, std::map<std::string, std::pair<int, int>> constraints = {});
cv::Mat resize_image(const cv::Mat &im, double division_factor);
void fftShift(const cv::Mat& input, cv::Mat& output);
std::pair<cv::Point2d, double> phase_correlation(const cv::Mat &im0, const cv::Mat &im1, const std::string &callback_name = "argmax2D", int filter_pcorr = 0, const std::map<std::string, std::pair<int, int>> &constraints = {});
cv::Point2d translation(const cv::Mat &im0, const cv::Mat &im1, int filter_pcorr = 0, double odds = 1, const std::map<std::string, std::pair<int, int>> &constraints = {});
cv::Mat read_image_from_path(const std::string &path);
cv::Mat register_image_translation(const cv::Mat &im0, const cv::Mat &im1, double scale_factor);
cv::Mat align_image_pair(const std::string &ref_im_path, const std::string &im_to_align_path);
std::vector<cv::Mat> gaussian_pyramid(const cv::Mat& img, int num_levels);
std::vector<cv::Mat> generate_laplacian_pyramid(const cv::Mat& img, int num_levels);

#endif // ALGORITHM_H
