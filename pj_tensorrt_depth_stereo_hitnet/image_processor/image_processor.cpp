/*** Include ***/
/* for general */
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <memory>

/* for OpenCV */
#include <opencv2/opencv.hpp>

/* for My modules */
#include "common_helper.h"
#include "common_helper_cv.h"
#include "depth_stereo_engine.h"
#include "image_processor.h"

/*** Macro ***/
#define TAG "ImageProcessor"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/*** Global variable ***/
std::unique_ptr<DepthStereoEngine> s_engine;

/*** Function ***/
static void DrawFps(cv::Mat& mat, double time_inference, cv::Point pos, double font_scale, int32_t thickness, cv::Scalar color_front, cv::Scalar color_back, bool is_text_on_rect = true)
{
    char text[64];
    static auto time_previous = std::chrono::steady_clock::now();
    auto time_now = std::chrono::steady_clock::now();
    double fps = 1e9 / (time_now - time_previous).count();
    time_previous = time_now;
    snprintf(text, sizeof(text), "FPS: %.1f, Inference: %.1f [ms]", fps, time_inference);
    CommonHelper::DrawText(mat, text, cv::Point(0, 0), 0.5, 2, CommonHelper::CreateCvColor(0, 0, 0), CommonHelper::CreateCvColor(180, 180, 180), true);
}

int32_t ImageProcessor::Initialize(const InputParam& input_param)
{
    if (s_engine) {
        PRINT_E("Already initialized\n");
        return -1;
    }

    s_engine.reset(new DepthStereoEngine());
    if (s_engine->Initialize(input_param.work_dir, input_param.num_threads) != DepthStereoEngine::kRetOk) {
        s_engine->Finalize();
        s_engine.reset();
        return -1;
    }
    return 0;
}

int32_t ImageProcessor::Finalize(void)
{
    if (!s_engine) {
        PRINT_E("Not initialized\n");
        return -1;
    }

    if (s_engine->Finalize() != DepthStereoEngine::kRetOk) {
        return -1;
    }

    return 0;
}


int32_t ImageProcessor::Command(int32_t cmd)
{
    if (!s_engine) {
        PRINT_E("Not initialized\n");
        return -1;
    }

    switch (cmd) {
    case 0:
    default:
        PRINT_E("command(%d) is not supported\n", cmd);
        return -1;
    }
}

static cv::Mat ConvertDisparity2Depth(const cv::Mat& mat_disparity, float fov, float baseline, float mag = 1.0f)
{
    cv::Mat mat_depth(mat_disparity.size(), CV_8UC1);
    const float scale = mag * fov * baseline;
#pragma omp parallel for
    for (int32_t i = 0; i < mat_disparity.total(); i++) {
        if (mat_disparity.at<float>(i) > 0) {
            float Z = scale / mat_disparity.at<float>(i);   // [meter]
            if (Z <= 255.0f) {
                mat_depth.at<uint8_t>(i) = static_cast<uint8_t>(Z);
            } else {
                mat_depth.at<uint8_t>(i) = 255;
            }
        } else {
            mat_depth.at<uint8_t>(i) = 255;
        }
    }
    return mat_depth;
}

static cv::Mat NormalizeDisparity(const cv::Mat& mat_disparity, float max_disparity, float mag = 1.0f)
{
    cv::Mat mat_depth(mat_disparity.size(), CV_8UC1);
    const float scale = mag * 255.0f / max_disparity;
#pragma omp parallel for
    for (int32_t i = 0; i < mat_disparity.total(); i++) {
        mat_depth.at<uint8_t>(i) = static_cast<uint8_t>(mat_disparity.at<float>(i) * scale);
    }
    return mat_depth;
}

int32_t ImageProcessor::Process(cv::Mat& mat_left, cv::Mat& mat_right, cv::Mat& mat_result, Result& result)
{
    if (!s_engine) {
        PRINT_E("Not initialized\n");
        return -1;
    }

    DepthStereoEngine::Result ss_result;
    if (s_engine->Process(mat_left, mat_right, ss_result) != DepthStereoEngine::kRetOk) {
        return -1;
    }
    
    /* Convert to colored depth map */
    //cv::Mat mat_depth = ConvertDisparity2Depth(ss_result.image, 500.0f, 0.2f, 50);
    cv::Mat mat_depth = NormalizeDisparity(ss_result.image, s_engine->GetMaxDisparity(), 1.0f);
    cv::applyColorMap(mat_depth, mat_depth, cv::COLORMAP_MAGMA);

    /* Create result image */
    cv::Mat mat_depth_orgsize = cv::Mat::zeros(mat_left.size(), CV_8UC3);
    cv::Mat mat_depth_orgsize_cropped = mat_depth_orgsize(cv::Rect(ss_result.crop.x, ss_result.crop.y, ss_result.crop.w, ss_result.crop.h));
    cv::resize(mat_depth, mat_depth_orgsize_cropped, mat_depth_orgsize_cropped.size());

    cv::vconcat(mat_left, mat_depth_orgsize, mat_result);
    if (mat_result.rows > 1080) {// just to fit to my display
        cv::resize(mat_result, mat_result, cv::Size(), 960.0f / mat_result.rows, 960.0f / mat_result.rows);
    }

    DrawFps(mat_result, ss_result.time_inference, cv::Point(0, 0), 0.5, 2, CommonHelper::CreateCvColor(0, 0, 0), CommonHelper::CreateCvColor(180, 180, 180), true);

    /* Return the results */
    result.time_pre_process = ss_result.time_pre_process;
    result.time_inference = ss_result.time_inference;
    result.time_post_process = ss_result.time_post_process;

    return 0;
}

