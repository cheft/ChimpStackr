#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "NumCpp.hpp"
#include "Algorithm.h" // 包含 Algorithm.h 头文件
#include <chrono>

// using namespace nc; // Use NumCpp namespace

// main function, open cv hello world
int main(int argc, char** argv) {
    auto start = std::chrono::high_resolution_clock::now();

    auto pyramid_num_levels = 8;
    auto fusion_kernel_size = 6;

    auto image_paths = [] {
        std::vector<std::string> paths;
        for (int i = 1; i < 3; ++i) {
            paths.push_back("/Users/chenhaifeng/Documents/Study/python/ChimpStackr/samples/examples/zoom/Gibbaranea-0" + std::to_string(i) + ".jpg");
            // paths.push_back("/Users/chenhaifeng/Documents/Study/python/ChimpStackr/samples/2160/" + std::to_string(i) + ".png");
        }
        return paths;
    }();

    std::vector<cv::Mat> aligned_images;
    aligned_images.push_back(read_image_from_path(image_paths[0]));
    std::cout << "aligned_images size: " << aligned_images[0].size() << ", channels: " << aligned_images[0].channels() << ", type: " << aligned_images[0].type() << std::endl; // python 的 dtype: uint8 == CV_8UC3 == type 16

    auto fused_pyr = generate_laplacian_pyramid(aligned_images[0], pyramid_num_levels);
    std::cout << "======= fused_pyr image size " << fused_pyr.size() << "..." << std::endl;
    // // 归一化并显示
    // cv::Mat display;
    // cv::normalize(fused_pyr[0], display, 0, 1, cv::NORM_MINMAX);
    // cv::imshow("Display window", fused_pyr[0]);
    // cv::waitKey(0);

    for (size_t i = 1; i < image_paths.size(); ++i) {
        std::cout << "Processing image " << i + 1 << "/" << image_paths.size() << "..." << std::endl;

        aligned_images.push_back(align_image_pair(image_paths[0], image_paths[i]));
        auto new_pyr = generate_laplacian_pyramid(aligned_images[1], pyramid_num_levels);
        std::cout << "aligned_images[1] size: " << aligned_images[1].size() << ", channels: " << aligned_images[1].channels() << ", type: " << aligned_images[1].type() << std::endl;
        // cv::imshow("Display window", new_pyr[1]);
        // cv::waitKey(0);
        aligned_images.erase(aligned_images.begin());

        fused_pyr = focus_fuse_pyramid_pair(fused_pyr, new_pyr, fusion_kernel_size);
    }

    std::vector<cv::Mat> new_fused_pyr;
    for (auto &pyr : fused_pyr) { // TODO: channels 3, 1 1 1 1... & type are diff, 16, 21 21 21 ...
        cv::Mat temp;
        if (pyr.channels() == 1) {
            cv::cvtColor(pyr, temp, cv::COLOR_GRAY2BGR);
        } else {
            temp = pyr;
        }
        if (temp.type() != CV_8UC3) {
            temp.convertTo(temp, CV_8UC3);
        }
        new_fused_pyr.push_back(temp);
        // std::cout << "temp size: " << temp.size() << ", channels: " << temp.channels() << ", type: " << temp.type() << std::endl;
    }
    auto fused_image = reconstruct_pyramid(new_fused_pyr);

    auto end = std::chrono::high_resolution_clock::now();
    // 计算耗时，单位为毫秒
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Function execution time: " << duration.count() << " ms" << std::endl;

    cv::imshow("Display window", fused_image);
    cv::waitKey(0);
    return 0;
}
