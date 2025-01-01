#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
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

  cv::Mat apos = cv::getGaussianKernel(aporad * 2, -1, CV_64F);
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

  cv::Mat apofield = vecs[0] * vecs[1].t();
  return apofield;
}

cv::Point argmax2D(const cv::Mat &array)
{
  cv::Point max_loc;
  cv::minMaxLoc(array, nullptr, nullptr, nullptr, &max_loc);

  return max_loc;
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
        ret = argmax2D(array);
    } else {
        cv::Mat col = cv::Mat::zeros(array.rows, 1, CV_64F);
        cv::Mat row = cv::Mat::zeros(1, array.cols, CV_64F);
        for (int i = 0; i < array.rows; ++i) col.at<double>(i, 0) = i;
        for (int j = 0; j < array.cols; ++j) row.at<double>(0, j) = j;

        cv::Mat arr2;
        cv::pow(array, exponent, arr2);
        double arrsum = cv::sum(arr2)[0];
        if (arrsum == 0) {
            return cv::Point2d(0, 0);
        }
        double arrprody = cv::sum(arr2.mul(col))[0] / arrsum;
        double arrprodx = cv::sum(arr2.mul(row))[0] / arrsum;
        ret = cv::Point2d(arrprody, arrprodx);
    }
    return ret;
}

cv::Mat get_subarr(const cv::Mat& array, const cv::Point& center, int rad) {
    int dim = 1 + 2 * rad;
    cv::Mat subarr = cv::Mat::zeros(dim, dim, array.type());
    cv::Point corner = center - cv::Point(rad, rad);
    for (int ii = 0; ii < dim; ++ii) {
        int yidx = (corner.y + ii + array.rows) % array.rows;
        for (int jj = 0; jj < dim; ++jj) {
            int xidx = (corner.x + jj + array.cols) % array.cols;
            subarr.at<double>(ii, jj) = array.at<double>(yidx, xidx);
        }
    }
    return subarr;
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
    cv::Point2d offset = com - cv::Point2d(rad, rad);
    cv::Point2d ret = rough + offset;
    ret += cv::Point2d(0.5, 0.5);
    ret.x = std::fmod(ret.x + array.cols, array.cols) - 0.5;
    ret.y = std::fmod(ret.y + array.rows, array.rows) - 0.5;
    return ret;
}

std::pair<cv::Point2d, double> argmax_translation(cv::Mat array, int filter_pcorr, std::map<std::string, std::pair<int, int>> constraints = {}) {
    if (constraints.empty()) {
        constraints = {{"tx", {0, 0}}, {"ty", {0, 0}}};
    }

    if (filter_pcorr > 0) {
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

            // 创建一个全零的 vals 矩阵
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
    // array = array.mul(mask);
    try {
        if (mask.channels() == 1 && array.channels() > 1) {
            std::vector<cv::Mat> channels(array.channels(), mask);
            cv::merge(channels, mask);
        }
        array = array.mul(mask);
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV exception1: " << e.what() << std::endl;
    }

    int aporad = std::min(ashape.width, ashape.height) / 6;
    cv::Mat mask2 = get_apofield(ashape, aporad);
    // array = array.mul(mask2);
    try {
        if (mask2.channels() == 1 && array.channels() > 1) {
            std::vector<cv::Mat> channels(array.channels(), mask2);
            cv::merge(channels, mask2);
        }
        array = array.mul(mask2);
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV exception2: " << e.what() << std::endl;
    }

    cv::Point2d tvec = argmax_ext(array, std::numeric_limits<double>::infinity());
    try {
      tvec = interpolate(array_orig, tvec);
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV exception3: " << e.what() << std::endl;
    }
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

std::pair<cv::Point2d, double> phase_correlation(
    const cv::Mat &im0, 
    const cv::Mat &im1, 
    // std::function<cv::Point(const cv::Mat &)> callback = argmax2D, 
    const std::string &callback_name = "argmax2D",
    int filter_pcorr = 0, 
    const std::map<std::string, std::pair<int, int>> &constraints = {}
)
{
  // 将图像转换为浮点型
  cv::Mat im0_float, im1_float;
  im0.convertTo(im0_float, CV_32F);
  im1.convertTo(im1_float, CV_32F);

  cv::Mat f0, f1;
  cv::dft(im0_float, f0, cv::DFT_COMPLEX_OUTPUT);
  cv::dft(im1_float, f1, cv::DFT_COMPLEX_OUTPUT);

  cv::Mat f0_conj;
  cv::mulSpectrums(f0, f1, f0_conj, 0, true);

  cv::Mat cps;
  cv::idft(f0_conj / (cv::abs(f0_conj) + 1e-15), cps, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);

  cv::Mat scps;
  // cv::fftShift(cps, scps);
  fftShift(cps, scps);

  // if (callback_name == "argmax2D") {
  //   cv::Point max_loc = argmax2D(scps);
  // } else if (callback_name == "argmax_translation") {

  std::pair<cv::Point2d, double> pd = argmax_translation(scps, filter_pcorr, constraints);
  cv::Point2d max_loc = pd.first;
  double success = pd.second;

  // }
  // cv::Point max_loc = callback(scps);
  // double success = scps.at<double>(max_loc);

  max_loc.x -= f0.cols / 2;
  max_loc.y -= f0.rows / 2;

  return {cv::Point2d(max_loc), success};
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

  // Calculate translational shift
  return register_image_translation(ref_image, image_to_align, 10.0);
}
