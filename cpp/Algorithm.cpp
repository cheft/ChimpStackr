#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fftw3.h> 
#include "NumCpp.hpp"
#include <cmath>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <string>

cv::Mat get_apofield(const cv::Size &shape, int aporad)
{
  if (aporad == 0)
  {
    return cv::Mat::ones(shape, CV_64F);
  }
  // 手动生成汉宁窗
  cv::Mat apos(aporad * 2, 1, CV_64F);
  for (int i = 0; i < apos.rows; ++i)
  {
      apos.at<double>(i, 0) = 0.5 * (1 - cos(2 * CV_PI * i / (aporad * 2 - 1)));
  }

  std::vector<cv::Mat> vecs;

  for (int dim : {shape.width, shape.height})
  {
    if (dim <= aporad * 2)
    {
      throw std::invalid_argument("Apodization radius too big for shape dimension.");
    }

    cv::Mat toapp = cv::Mat::ones(dim, 1, CV_64F);
    apos.rowRange(0, aporad).copyTo(toapp.rowRange(0, aporad));
    apos.rowRange(0, aporad).copyTo(toapp.rowRange(dim - aporad, dim));
    vecs.push_back(toapp);
  }
  
  cv::Mat apofield = vecs[1] * vecs[0].t(); // 生成矩阵，这里跟 python 的宽高是反的

  return apofield;
}

std::vector<int> unravel_index(int index, const std::vector<int>& shape) {
    int size = shape.size();
    std::vector<int> indices(size);
    for (int i = size - 1; i >= 0; --i) {
        indices[i] = index % shape[i];
        index /= shape[i];
    }
    return indices;
}

cv::Point argmax2D(const cv::Mat &array)
{
    // 将 cv::Mat 转换为 NumCpp 的 NdArray
    nc::NdArray<double> ncArray(array.rows, array.cols);
    for (int i = 0; i < array.rows; ++i) {
        for (int j = 0; j < array.cols; ++j) {
            ncArray(i, j) = array.at<double>(i, j);
        }
    }

    // 找到最大值的索引
    auto amax = nc::argmax(ncArray).item();

    // 将一维索引转换为二维索引
    auto ret = unravel_index(amax, {static_cast<int>(ncArray.shape().rows), static_cast<int>(ncArray.shape().cols)});

    return cv::Point(ret[1], ret[0]);
}


cv::Mat minimum_filter(const cv::Mat &array, int size)
{
  if (size % 2 == 0)
  {
    throw std::invalid_argument("Size must be an odd integer.");
  }

  int pad_width = size / 2;
  cv::Mat padded_array;
  cv::copyMakeBorder(array, padded_array, pad_width, pad_width, pad_width, pad_width, cv::BORDER_REPLICATE);

  cv::Mat filtered_array = array.clone();

  for (int i = 0; i < array.rows; ++i)
  {
    for (int j = 0; j < array.cols; ++j)
    {
      cv::Rect roi(j, i, size, size);
      cv::Mat local_window = padded_array(roi);
      double min_val;
      cv::minMaxLoc(local_window, &min_val);
      filtered_array.at<uchar>(i, j) = static_cast<uchar>(min_val);
    }
  }

  return filtered_array;
}

cv::Point2d argmax_ext(const cv::Mat& array, double exponent) {
    cv::Point2d ret;
    if (exponent == std::numeric_limits<double>::infinity()) {
        printf("exponent is infinity\n");
        ret = argmax2D(array);
    } else {
        // 使用 NumCpp 实现计算
        nc::NdArray<double> ncArray(array.rows, array.cols);
        for (int i = 0; i < array.rows; ++i) {
            for (int j = 0; j < array.cols; ++j) {
                ncArray(i, j) = array.at<double>(i, j);
                // printf("array.at<double>(i, j): %f\n", array.at<double>(i, j));
            }
            printf("\n");
        }

        // col = np.arange(array.shape[0])[:, np.newaxis]
        // row = np.arange(array.shape[1])[np.newaxis, :]

        // arr2 = array**exponent
        // arrsum = arr2.sum()
        // if arrsum == 0:
        //     # We have to return SOMETHING, so let's go for (0, 0)
        //     return np.zeros(2)
        // arrprody = np.sum(arr2 * col) / arrsum
        // arrprodx = np.sum(arr2 * row) / arrsum
        // ret = [arrprody, arrprodx]
        // print("ret 1111", ret)

        auto col = nc::arange<double>(0, array.rows).reshape(array.rows, 1);
        auto row = nc::arange<double>(0, array.cols).reshape(1, array.cols);

        auto arr2 = nc::power(ncArray, exponent);
        // std::cout << "Array using manual iteration:" << std::endl;
        // for (int row = 0; row < arr2.numRows(); ++row)
        // {
        //     for (int col = 0; col < arr2.numCols(); ++col)
        //     {
        //         std::cout << arr2(row, col) << " ";
        //     }
        //     std::cout << std::endl;
        // }

        double arrsum = nc::sum(arr2).item();
        printf("arrsum: %f\n", arrsum);
        if (arrsum == 0) {
            return cv::Point2d(0, 0);
        }
        double arrprody = nc::sum(arr2 * col).item() / arrsum;
        double arrprodx = nc::sum(arr2 * row).item() / arrsum;
        
        ret = cv::Point2d(arrprody, arrprodx);
    }
    return ret;
}

cv::Mat get_subarr(const cv::Mat& array, const cv::Point& center, int rad) {
    int dim = 1 + 2 * rad;
    nc::NdArray<double> ncArray(array.rows, array.cols);
    for (int i = 0; i < array.rows; ++i) {
        for (int j = 0; j < array.cols; ++j) {
            ncArray(i, j) = array.at<double>(i, j);
        }
    }

    nc::NdArray<double> subarr = nc::zeros<double>(dim, dim);
    nc::NdArray<int> corner = nc::NdArray<int>({center.y, center.x}) - rad;

    for (int ii = 0; ii < dim; ++ii) {
        int yidx = (corner[0] + ii + ncArray.shape().rows) % ncArray.shape().rows;
        for (int jj = 0; jj < dim; ++jj) {
            int xidx = (corner[1] + jj + ncArray.shape().cols) % ncArray.shape().cols;
            subarr(ii, jj) = ncArray(yidx, xidx);
        }
    }

    cv::Mat result(dim, dim, array.type());
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            result.at<double>(i, j) = subarr(i, j);
        }
    }

    return result;
}


double get_success(const cv::Mat& array, const cv::Point2d& coord, int radius = 2) {
    cv::Point coord_int = cv::Point(cvRound(coord.x), cvRound(coord.y));
    cv::Mat subarr = get_subarr(array, coord_int, radius);
    double theval = cv::sum(subarr)[0];
    double theval2 = array.at<double>(coord_int);
    return std::sqrt(theval * theval2);
}

cv::Point2d interpolate(const cv::Mat& array, const cv::Point2d& rough, int rad = 2) {
    cv::Point rough_int = cv::Point(cvRound(rough.x), cvRound(rough.y));
   
    cv::Mat surroundings = get_subarr(array, rough_int, rad);

    cv::Point2d com = argmax_ext(surroundings, 1);
    printf("com.x: %f\n", com.x);
    printf("com.y: %f\n", com.y);

    cv::Point2d offset = com - cv::Point2d(rad, rad);
    cv::Point2d ret = rough + offset;
    ret += cv::Point2d(0.5, 0.5);
    ret.x = std::fmod(ret.x + array.cols, array.cols) - 0.5;
    ret.y = std::fmod(ret.y + array.rows, array.rows) - 0.5;
    return ret;
}

std::pair<cv::Point2d, double> argmax_translation(cv::Mat array, int filter_pcorr, std::map<std::string, std::pair<int, int>> constraints = {}) {
    if (constraints.empty()) {
        printf("constraints is empty\n");
        constraints = {{"tx", {0, 0}}, {"ty", {0, 0}}};
    }

    if (filter_pcorr > 0) {
        printf("filter_pcorr: %d\n", filter_pcorr);
        array = minimum_filter(array, filter_pcorr);
    }

    cv::Mat array_orig = array.clone();
    cv::Size ashape = array.size();
    cv::Mat mask = cv::Mat::ones(ashape, CV_64F);
    for (int dim = 0; dim < 2; ++dim) {
        std::string key = (dim == 0) ? "ty" : "tx";
        if (constraints[key].second == 0) {
            continue;
        }
        int pos = constraints[key].first;
        int sigma = constraints[key].second;
        int alen = (dim == 0) ? ashape.height : ashape.width;
        cv::Mat dom = cv::Mat::zeros(1, alen, CV_64F);
        for (int i = 0; i < alen; ++i) dom.at<double>(0, i) = -alen / 2 + i;
        cv::Mat vals;
        if (sigma == 0) {
            // 计算 dom 和 pos 的绝对差
            cv::Mat diff = cv::abs(dom - pos);

            // 使用 cv::minMaxLoc 找到最小值的位置
            double minVal, maxVal;
            cv::Point minLoc, maxLoc;
            cv::minMaxLoc(diff, &minVal, &maxVal, &minLoc, &maxLoc);

            // 创建一个全零的 vals 矶阵
            vals = cv::Mat::zeros(1, alen, CV_64F);

            // 在最小值的位置设置为 1.0
            vals.at<double>(0, minLoc.x) = 1.0;
        } else {
            // cv::exp(-cv::pow(dom - pos, 2) / (sigma * sigma), vals);
            // 计算平方差
            cv::Mat diffSquared;
            cv::pow(dom - pos, 2, diffSquared);

            // 计算高斯函数的值
            cv::Mat vals;
            cv::exp(-diffSquared / (sigma * sigma), vals);

            // 输出结果
            std::cout << "vals: " << vals << std::endl;
        }
        if (dim == 0) {
            mask = mask.mul(vals.t());
        } else {
            mask = mask.mul(vals);
        }
    }
    std::cout << "mask size: " << mask.size() << ", channels: " << mask.channels() << std::endl;
    std::cout << "array size: " << array.size() << ", channels: " << array.channels() << std::endl;

    array = array.mul(mask);
    int aporad = std::min(ashape.width / 6, ashape.height / 6);
    cv::Mat mask2 = get_apofield(ashape, aporad);
    std::cout << "mask2 size: " << mask2.size() << ", channels: " << mask2.channels() << std::endl;

    array = array.mul(mask2);

    cv::Point2d tvec = argmax_ext(array, std::numeric_limits<double>::infinity());
    
    tvec = interpolate(array_orig, tvec);
    // printf("tvecx: %f\n", tvec.x);
    // printf("tvecy: %f\n", tvec.y);

    double success = get_success(array_orig, tvec, 2);
    std::cout << "tvec: " << tvec << std::endl;
    std::cout << "success: " << success << std::endl;
    return {tvec, success};
}

cv::Mat resize_image(const cv::Mat &im, double division_factor)
{
  int largest_axis = 0;
  if (im.cols > im.rows)
  {
    largest_axis = 1;
  }

  double multiplication_factor = static_cast<double>(im.size[largest_axis] / division_factor) / im.size[largest_axis];

  int new_width = std::floor(im.cols * multiplication_factor);
  int new_height = std::floor(im.rows * multiplication_factor);

  cv::Mat resized_image;
  cv::resize(im, resized_image, cv::Size(new_width, new_height));
  return resized_image;
}



void fftShift(const cv::Mat& input, cv::Mat& output) {
    output = input.clone();
    int cx = output.cols / 2;
    int cy = output.rows / 2;

    cv::Mat q0(output, cv::Rect(0, 0, cx, cy));   // Top-Left
    cv::Mat q1(output, cv::Rect(cx, 0, cx, cy));  // Top-Right
    cv::Mat q2(output, cv::Rect(0, cy, cx, cy));  // Bottom-Left
    cv::Mat q3(output, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

    cv::Mat tmp; // 临时存储

    // 交换象限 (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    // 交换象限 (Top-Right with Bottom-Left)
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

void printPartialMat(const cv::Mat& mat, int startRow, int endRow, int startCol, int endCol) {
    // Ensure the specified range is within the matrix bounds
    startRow = std::max(0, startRow);
    endRow = std::min(mat.rows, endRow);
    startCol = std::max(0, startCol);
    endCol = std::min(mat.cols, endCol);

    // Create a submatrix (ROI)
    cv::Mat subMat = mat(cv::Range(startRow, endRow), cv::Range(startCol, endCol));

    // Print the submatrix
    for (int i = 0; i < subMat.rows; ++i) {
        for (int j = 0; j < subMat.cols; ++j) {
            std::cout << subMat.at<double>(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

std::pair<cv::Point2d, double> phase_correlation(
    const cv::Mat &im0, 
    const cv::Mat &im1, 
    // std::function<cv::Point(const cv::Mat &)> callback = argmax2D, 
    const std::string &callback_name = "argmax2D",
    int filter_pcorr = 0, 
    const std::map<std::string, std::pair<int, int>> &constraints = {}
)
{
  // print(im0);
  // std::cout << "\n\n" << std::endl;
  // print(im1);
  // 获取图像尺寸

  int rows = im0.rows;
  int cols = im0.cols;

  cv::Mat im0_double, im1_double;
  im0.convertTo(im0_double, CV_64F);
  im1.convertTo(im1_double, CV_64F);

  // 创建 FFTW 计划
  fftw_complex *f0 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * rows * (cols / 2 + 1));
  fftw_complex *f1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * rows * (cols / 2 + 1));
  fftw_complex *cps = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * rows * (cols / 2 + 1));
  double *ifft_result = (double*) fftw_malloc(sizeof(double) * rows * cols);

  fftw_plan plan0 = fftw_plan_dft_r2c_2d(rows, cols, im0_double.ptr<double>(), f0, FFTW_ESTIMATE);
  fftw_plan plan1 = fftw_plan_dft_r2c_2d(rows, cols, im1_double.ptr<double>(), f1, FFTW_ESTIMATE);
  fftw_plan iplan = fftw_plan_dft_c2r_2d(rows, cols, cps, ifft_result, FFTW_ESTIMATE);

  // 执行 FFT
  fftw_execute(plan0);
  fftw_execute(plan1);

  // 打印 FFT 结果
  // std::cout << "FFT of im0:" << std::endl;
  // for (int i = 0; i < 4; ++i) {
  //     for (int j = 0; j < 4 / 2 + 1; ++j) {
  //         std::cout << "(" << f0[i * (cols / 2 + 1) + j][0] << ", " << f0[i * (cols / 2 + 1) + j][1] << ") ";
  //     }
  //     std::cout << std::endl;
  // }

  // std::cout << "FFT of im1:" << std::endl;
  // for (int i = 0; i < 4; ++i) {
  //     for (int j = 0; j < 4 / 2 + 1; ++j) {
  //         std::cout << "(" << f1[i * (cols / 2 + 1) + j][0] << ", " << f1[i * (cols / 2 + 1) + j][1] << ") ";
  //     }
  //     std::cout << std::endl;
  // }

  double max_abs_f1 = 0.0;
  for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols / 2 + 1; ++j) {
          double real = f1[i * (cols / 2 + 1) + j][0];
          double imag = f1[i * (cols / 2 + 1) + j][1];
          double abs_value = std::sqrt(real * real + imag * imag);
          if (abs_value > max_abs_f1) {
              max_abs_f1 = abs_value;
          }
      }
  }

  // 计算 eps
  double eps = max_abs_f1 * 1e-15;
  std::cout << "Eps: " << eps << std::endl;

  // 计算 cps = (f0 * conj(f1)) / (abs(f0) * abs(f1) + eps)
  for (int i = 0; i < rows * (cols / 2 + 1); ++i) {
      double abs_f0 = std::sqrt(f0[i][0] * f0[i][0] + f0[i][1] * f0[i][1]);
      double abs_f1 = std::sqrt(f1[i][0] * f1[i][0] + f1[i][1] * f1[i][1]);

      double conj_f1_real = f1[i][0];
      double conj_f1_imag = -f1[i][1];

      double numerator_real = f0[i][0] * conj_f1_real - f0[i][1] * conj_f1_imag;
      double numerator_imag = f0[i][0] * conj_f1_imag + f0[i][1] * conj_f1_real;

      double denominator = (abs_f0 * abs_f1) + eps;

      cps[i][0] = numerator_real / denominator;
      cps[i][1] = numerator_imag / denominator;
  }

  // 执行逆 FFT
  fftw_execute(iplan);

   // 归一化逆 FFT 结果
  cv::Mat cps_mat(rows, cols, CV_64F, ifft_result);
  cps_mat /= (rows * cols);

  // 频谱移位
  cv::Mat scps;
  fftShift(cps_mat, scps);

  // 归一化并显示
  // cv::Mat display;
  // cv::normalize(cps_mat, display, 0, 1, cv::NORM_MINMAX); 这个归一化会影响图片显示, 有时需要加上

  // cv::imshow("cps", scps);
  // cv::waitKey(0);
  // 以上是 OK 的

  std::pair<cv::Point2d, double> pd = argmax_translation(scps, filter_pcorr, constraints);
  cv::Point2d max_loc = pd.first;
  double success = pd.second;

  // 计算逆 FFT 结果的绝对值并找到最大值
  // double max_cps = 0.0;
  // cv::Point2d max_loc(0, 0);
  // for (int i = 0; i < rows; ++i) {
  //     for (int j = 0; j < cols; ++j) {
  //         double value = im0.at<double>(i, j); // 使用 im0 作为输出缓冲区
  //         if (value > max_cps) {
  //             max_cps = value;
  //             max_loc = cv::Point2d(j, i);
  //         }
  //     }
  // }
  
   // 清理
  fftw_destroy_plan(plan0);
  fftw_destroy_plan(plan1);
  fftw_destroy_plan(iplan);
  fftw_free(f0);
  fftw_free(f1);
  fftw_free(cps);
  fftw_free(ifft_result);
  fftw_cleanup();

  // 这里可以进行相位相关计算
  // 例如，计算 f0 和 f1 的共轭乘积，然后进行逆 FFT

  // 返回一个占位符结果
  return std::make_pair(cv::Point2d(0, 0), 0.0);
  
  // cv::Mat im0_double, im1_double;
  // im0.convertTo(im0_double, CV_64F);
  // im1.convertTo(im1_double, CV_64F);

  // // Get the size of the input images
  // int rows = im0.rows;
  // int cols = im0.cols;

  // // Allocate memory for FFTW
  // fftw_complex *f0 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * rows * cols);
  // fftw_complex *f1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * rows * cols);
  // fftw_complex *cps = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * rows * cols);

  // // Create FFTW plans
  // fftw_plan plan_f0 = fftw_plan_dft_r2c_2d(rows, cols, im0_double.ptr<double>(), f0, FFTW_ESTIMATE);
  // fftw_plan plan_f1 = fftw_plan_dft_r2c_2d(rows, cols, im1_double.ptr<double>(), f1, FFTW_ESTIMATE);
  // fftw_plan plan_cps = fftw_plan_dft_c2r_2d(rows, cols, cps, im0_double.ptr<double>(), FFTW_ESTIMATE);

  // // Execute FFT
  // fftw_execute(plan_f0);
  // fftw_execute(plan_f1);

  // // Calculate cross power spectrum
  // double eps = 1e-15;
  // for (int i = 0; i < rows * cols; ++i) {
  //     std::complex<double> F0(f0[i][0], f0[i][1]);
  //     std::complex<double> F1(f1[i][0], f1[i][1]);
  //     std::complex<double> conjF1 = std::conj(F1);
  //     std::complex<double> denominator = std::abs(F0) * std::abs(F1) + eps;
  //     std::complex<double> result = (F0 * conjF1) / denominator;
  //     cps[i][0] = result.real();
  //     cps[i][1] = result.imag();
  // }

  // // Execute inverse FFT

  // cv::Mat cps_abs(rows, cols, CV_64F);

  // // Execute inverse FFT
  // fftw_execute_dft_c2r(plan_cps, cps, cps_abs.ptr<double>());

  // cv::normalize(cps_abs, cps_abs, 0, 1, cv::NORM_MINMAX);

  // printPartialMat(cps_abs, 0, 5, 0, 5);

  // cv::Mat scps;
  // fftShift(cps_abs, scps);
  // // cv::fftshift(result, shifted_result);
  // cv::imshow("scps c++", cps_abs);
  // cv::waitKey(0);

  // // if (callback_name == "argmax2D") {
  // //   cv::Point max_loc = argmax2D(scps);
  // // } else if (callback_name == "argmax_translation") {

  // std::pair<cv::Point2d, double> pd = argmax_translation(scps, filter_pcorr, constraints);
  // cv::Point2d max_loc = pd.first;
  // double success = pd.second;

  // // }
  // // cv::Point max_loc = callback(scps);
  // // double success = scps.at<double>(max_loc);

  // max_loc.x -= cols / 2;
  // max_loc.y -= rows / 2;

  // fftw_destroy_plan(plan_f0);
  // fftw_destroy_plan(plan_f1);
  // fftw_destroy_plan(plan_cps);
  // fftw_free(f0);
  // fftw_free(f1);
  // fftw_free(cps);
  // printf("max_loc.x: %f\n", max_loc.x);
  // printf("max_loc.y: %f\n", max_loc.y);
  // printf("success: %f\n", success);
  // return {cv::Point2d(max_loc), success};
}

std::vector<cv::Mat> gaussian_pyramid(const cv::Mat& img, int num_levels) {
    cv::Mat lower = img.clone();
    std::vector<cv::Mat> gaussian_pyr;
    lower.convertTo(lower, CV_32F);  // 每次下采样后转换为 float32 类型
    gaussian_pyr.push_back(lower);
    for (int i = 0; i < num_levels; ++i) {
        cv::pyrDown(lower, lower);
        lower.convertTo(lower, CV_32F);  // 每次下采样后转换为 float32 类型
        gaussian_pyr.push_back(lower);
    }
    
    return gaussian_pyr;
}

std::vector<cv::Mat> generate_laplacian_pyramid(const cv::Mat& img, int num_levels) {
    std::vector<cv::Mat> gaussian_pyr = gaussian_pyramid(img, num_levels);
    
    cv::Mat laplacian_top = gaussian_pyr.back();
    std::vector<cv::Mat> laplacian_pyr;
    laplacian_pyr.push_back(laplacian_top);

    for (int i = num_levels; i > 0; --i) {
        cv::Mat gaussian_expanded;
        cv::pyrUp(gaussian_pyr[i], gaussian_expanded, gaussian_pyr[i - 1].size());
        
        // 使用 OpenCV 的逐元素操作
        cv::Mat laplacian;
        cv::subtract(gaussian_pyr[i - 1], gaussian_expanded, laplacian);
  
        // std::cout << "laplacian[1] size: " << laplacian.size() << ", channels: " << laplacian.channels() << ", type: " << laplacian.type() << std::endl;

        // cv::Mat display;
        // cv::normalize(laplacian, display, 0, 1, cv::NORM_MINMAX);
        // cv::imshow("Display window", display);
        // cv::waitKey(0);
        laplacian_pyr.push_back(laplacian);
    }

    return laplacian_pyr;
}

cv::Mat reconstruct_pyramid(const std::vector<cv::Mat>& laplacian_pyr) {
    cv::Mat laplacian_top = laplacian_pyr[0];
   
    std::vector<cv::Mat> laplacian_lst;
    laplacian_lst.push_back(laplacian_top);
    int num_levels = laplacian_pyr.size() - 1;

    for (int i = 0; i < num_levels; ++i) {
        cv::Size size(laplacian_pyr[i + 1].cols, laplacian_pyr[i + 1].rows);
        cv::Mat laplacian_expanded;
        cv::pyrUp(laplacian_top, laplacian_expanded, size);

         // // 归一化并显示
        // cv::Mat display;
        // cv::normalize(laplacian_top, display, 0, 1, cv::NORM_MINMAX);
        // cv::imshow("Display window", display);
        // cv::waitKey(0);

        laplacian_top = laplacian_pyr[i + 1] + laplacian_expanded;
        laplacian_lst.push_back(laplacian_top);
    }

   
    return laplacian_lst[num_levels];
}

cv::Point2d translation(const cv::Mat &im0, const cv::Mat &im1, int filter_pcorr = 0, double odds = 1, const std::map<std::string, std::pair<int, int>> &constraints = {})
{
  double angle = 0;
  cv::Point2d tvec, tvec2;
  double succ, succ2;

  // We estimate translation for the original image...
  std::tie(tvec, succ) = phase_correlation(im0, im1, "argmax_translation", filter_pcorr, constraints);

  // ... and for the 180-degrees rotated image (the rotation estimation doesn't distinguish rotation of x vs x + 180deg).
  cv::Mat rotated_im1;
  cv::rotate(im1, rotated_im1, cv::ROTATE_180);
  std::tie(tvec2, succ2) = phase_correlation(im0, rotated_im1, "argmax_translation", filter_pcorr, constraints);

  bool pick_rotated = false;
  if (succ2 * odds > succ || odds == -1)
  {
    pick_rotated = true;
  }

  if (pick_rotated)
  {
    tvec = tvec2;
    succ = succ2;
    angle += 180;
  }
//   printf("tvecx: %f\n", tvec.x);
//   printf("tvecy: %f\n", tvec.y);
  return tvec;
}

cv::Mat register_image_translation(const cv::Mat &im0, const cv::Mat &im1, double scale_factor)
{
  cv::Mat gray_im0, gray_im1;
  cv::cvtColor(im0, gray_im0, cv::COLOR_BGR2GRAY);
  cv::cvtColor(im1, gray_im1, cv::COLOR_BGR2GRAY);

  cv::Mat resized_im0 = resize_image(gray_im0, scale_factor);
  cv::Mat resized_im1 = resize_image(gray_im1, scale_factor);

  // 假设 translation 函数返回一个包含平移向量的结构
  cv::Point2d translation_result = translation(resized_im0, resized_im1);
  
  int height = im1.rows;
  int width = im1.cols;
  double y_shift = translation_result.y * scale_factor;
  double x_shift = translation_result.x * scale_factor;

  cv::Mat translation_matrix = (cv::Mat_<double>(2, 3) << 1, 0, x_shift, 0, 1, y_shift);
  cv::Mat result;
  cv::warpAffine(im1, result, translation_matrix, cv::Size(width, height));

//   cv::imshow("result 7777 ", result);
//   cv::waitKey(0);
  return result;
}

cv::Mat read_image_from_path(const std::string &path)
{
  return cv::imread(path, cv::IMREAD_COLOR);
}

cv::Mat align_image_pair(const std::string &ref_im_path, const std::string &im_to_align_path)
{
  cv::Mat ref_image = read_image_from_path(ref_im_path);
  cv::Mat image_to_align = read_image_from_path(im_to_align_path);
  
  std::cout << "image_to_align size: " << image_to_align.size() << ", channels: " << image_to_align.channels() << ", type: " << image_to_align.type() << std::endl; // python 的 dtype: uint8 == CV_8UC3 == type 16

  // Calculate translational shift
  return register_image_translation(ref_image, image_to_align, 10.0);
}

cv::Mat pad_array(const cv::Mat& array, int kernel_size) {
    int y_shape = array.rows;
    int x_shape = array.cols;

    // Calculate the necessary padding on each side
    int y_pad = kernel_size - y_shape;
    int x_pad = kernel_size - x_shape;

    // If no padding is needed, return the original array
    if (y_pad <= 0 && x_pad <= 0) {
        return array;
    }

    // Create a new padded array with the same type as the input
    cv::Mat padded_array = cv::Mat::zeros(y_shape + std::max(0, y_pad), x_shape + std::max(0, x_pad), array.type());

    // Copy the original array into the top-left corner of the padded array
    array.copyTo(padded_array(cv::Rect(0, 0, x_shape, y_shape)));
    cv::imshow("padded_array c++", padded_array);
    cv::waitKey(0);
    return padded_array;
}

double get_deviation(const cv::Mat& matrix) {
    // cv::Mat matrix_double;
    // matrix.convertTo(matrix_double, CV_64F);
    // matrix_double /= 255.0; // 上面的代码待验证

    double summed_deviation = 0.0;
    double average_value = cv::mean(matrix)[0];
    int kernel_area = matrix.rows * matrix.cols;

    for (int y = 0; y < matrix.rows; ++y) {
        for (int x = 0; x < matrix.cols; ++x) {
            double diff = matrix.at<double>(y, x) - average_value;
            summed_deviation += (diff * diff) / kernel_area;
        }
    }
    // std::cout << "summed_deviation: " << summed_deviation << std::endl;
    return summed_deviation;
}

cv::Mat compute_focusmap(const cv::Mat& pyr_level1, const cv::Mat& pyr_level2, int kernel_size) {
    int y_range = pyr_level1.rows;
    int x_range = pyr_level1.cols;
    // 2D focusmap (CV_8U); possible values:
    // 0 => pixel of pyr_level1
    // 1 => pixel of pyr_level2
    cv::Mat focusmap = cv::Mat::zeros(y_range, x_range, CV_8U);
    int k = kernel_size / 2;
    std::cout << "y_range: " << y_range << ", x_range: " << x_range << ", k: " << k << std::endl;

    for (int y = 0; y < y_range; y++) {
        for (int x = 0; x < x_range; x++) {
            int y_slice_start = std::max(0, y - k);
            int y_slice_end = std::min(y_range, y + k);
            int x_slice_start = std::max(0, x - k);
            int x_slice_end = std::min(x_range, x + k - 1); // 有问题

            cv::Mat patch1 = pyr_level1(cv::Range(y_slice_start, y_slice_end), cv::Range(x_slice_start, x_slice_end));
            std::cout << "patch1 size: " << patch1.size() << ", channels: " << patch1.channels() << ", type: " << patch1.type() << std::endl;
            cv::Mat padded_patch1 = pad_array(patch1, kernel_size);
            double dev1 = get_deviation(padded_patch1);

            cv::Mat patch2 = pyr_level2(cv::Range(y_slice_start, y_slice_end), cv::Range(x_slice_start, x_slice_end));
            cv::Mat padded_patch2 = pad_array(patch2, kernel_size);
            double dev2 = get_deviation(padded_patch2);

            // Determine which patch is more in focus
            uchar value_to_insert = (dev2 > dev1) ? 1 : 0;
            focusmap.at<uchar>(y, x) = value_to_insert;
        }
    }

    cv::Mat display;
    cv::normalize(focusmap, display, 0, 1, cv::NORM_MINMAX);
    cv::imshow("focusmap c++", display);
    cv::waitKey(0);
    return focusmap;
}

cv::Mat fuse_pyramid_levels_using_focusmap(cv::Mat pyr_level1, const cv::Mat& pyr_level2, const cv::Mat& focusmap) {
    for (int y = 0; y < focusmap.rows; ++y) {
        for (int x = 0; x < focusmap.cols; ++x) {
            if (focusmap.at<uchar>(y, x) != 0) {  // 如果 focusmap[y, x] 不为 0
                pyr_level1.at<cv::Vec3b>(y, x) = pyr_level2.at<cv::Vec3b>(y, x);
            }
            // 否则，pyr_level1[y, x] 保持不变
        }
    }
    return pyr_level1;
}
std::vector<cv::Mat> focus_fuse_pyramid_pair(const std::vector<cv::Mat>& pyr1, const std::vector<cv::Mat>& pyr2, int kernel_size) {
    int threshold_index = pyr1.size() - 1;
    std::vector<cv::Mat> new_pyr;
    cv::Mat current_focusmap;

    for (int pyramid_level = 0; pyramid_level < pyr1.size(); ++pyramid_level) {
        if (pyramid_level < threshold_index) {
            cv::Mat gray_pyr1, gray_pyr2;
            cv::cvtColor(pyr1[pyramid_level], gray_pyr1, cv::COLOR_BGR2GRAY);
            cv::cvtColor(pyr2[pyramid_level], gray_pyr2, cv::COLOR_BGR2GRAY);
            current_focusmap = compute_focusmap(gray_pyr1, gray_pyr2, kernel_size);
            // cv::imshow("current_focusmap c++ 1", current_focusmap);
            // cv::waitKey(0);
        } else {
            cv::Size s = pyr2[pyramid_level].size();
            cv::resize(current_focusmap, current_focusmap, s, 0, 0, cv::INTER_AREA);
            // cv::imshow("current_focusmap c++ 2", current_focusmap);
            // cv::waitKey(0);
        }

        cv::Mat new_pyr_level = fuse_pyramid_levels_using_focusmap(pyr1[pyramid_level].clone(), pyr2[pyramid_level], current_focusmap);
        // cv::imshow("new_pyr_level c++", new_pyr_level);
        // cv::waitKey(0);
        new_pyr.push_back(new_pyr_level);
    }

    return new_pyr;
}
