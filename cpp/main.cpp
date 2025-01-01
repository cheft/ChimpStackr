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
    // Read the image file
    cv::Mat image = cv::imread("/Users/chenhaifeng/Documents/Study/python/ChimpStackr/result.jpg", cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    nc::NdArray<int> a = { {1, 2}, {3, 4}, {5, 6} };
    std::cout << a << std::endl;

    auto image_paths = [] {
        std::vector<std::string> paths;
        for (int i = 1; i < 3; ++i) {
            paths.push_back("/Users/chenhaifeng/Documents/Study/python/ChimpStackr/samples/examples/zoom/Gibbaranea-0" + std::to_string(i) + ".jpg");
        }
        return paths;
    }();

    std::vector<cv::Mat> aligned_images;
    aligned_images.push_back(align_image_pair(image_paths[0], image_paths[0]));
    // auto fused_pyr = generate_laplacian_pyramid(aligned_images[0], pyramid_num_levels);

    // for (size_t i = 1; i < image_paths.size(); ++i) {
    //     std::cout << "Processing image " << i + 1 << "/" << image_paths.size() << "..." << std::endl;

    //     aligned_images.push_back(algorithm.align_image_pair(image_paths[0], image_paths[i]));
    //     auto new_pyr = Algorithm::generate_laplacian_pyramid(aligned_images[1], pyramid_num_levels);
    //     aligned_images.erase(aligned_images.begin());

    //     fused_pyr = Algorithm::focus_fuse_pyramid_pair(fused_pyr, new_pyr, fusion_kernel_size);
    // }

    // auto fused_image = Algorithm::reconstruct_pyramid(fused_pyr);


    // Create a window
    cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
    // Show our image inside the created window
    cv::imshow("Display window", image);
    // cv::imshow("Fused Image", fused_image);

    // Wait for any keystroke in the window
    cv::waitKey(0);
    return 0;
}
