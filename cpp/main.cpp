#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "NumCpp.hpp"
#include "Algorithm.h" // 包含 Algorithm.h 头文件

// using namespace nc; // Use NumCpp namespace

// main function, open cv hello world
int main(int argc, char** argv) {
    auto pyramid_num_levels = 8;
    auto fusion_kernel_size = 6;

    auto image_paths = [] {
        std::vector<std::string> paths;
        for (int i = 1; i < 3; ++i) {
            paths.push_back("/Users/chenhaifeng/Documents/Study/python/ChimpStackr/samples/examples/zoom/Gibbaranea-0" + std::to_string(i) + ".jpg");
        }
        return paths;
    }();

    std::vector<cv::Mat> aligned_images;
    aligned_images.push_back(read_image_from_path(image_paths[0]));
    
    auto fused_pyr = generate_laplacian_pyramid(aligned_images[0], pyramid_num_levels);
    std::cout << "======= fused_pyr image size " << fused_pyr.size() << "..." << std::endl;
    // 归一化并显示
    // cv::Mat display;
    // cv::normalize(fused_pyr[0], display, 0, 1, cv::NORM_MINMAX);
    // cv::imshow("Display window", fused_pyr[0]);
    // cv::waitKey(0);

    for (size_t i = 1; i < image_paths.size(); ++i) {
        std::cout << "Processing image " << i + 1 << "/" << image_paths.size() << "..." << std::endl;

        aligned_images.push_back(align_image_pair(image_paths[0], image_paths[i]));
        auto new_pyr = generate_laplacian_pyramid(aligned_images[1], pyramid_num_levels);
        std::cout << "======= new_pyr image size " << new_pyr.size() << "..." << std::endl;

        // cv::imshow("Display window", new_pyr[1]);
        // cv::waitKey(0);

        aligned_images.erase(aligned_images.begin());

    
        fused_pyr = focus_fuse_pyramid_pair(fused_pyr, new_pyr, fusion_kernel_size);
    }

    // auto fused_image = reconstruct_pyramid(fused_pyr);

    // cv::imshow("Display window", image);
    // cv::waitKey(0);
    return 0;
}
