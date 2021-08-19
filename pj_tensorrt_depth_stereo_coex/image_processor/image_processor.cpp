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
    cv::Mat mat_depth;
    cv::applyColorMap(ss_result.image, mat_depth, cv::COLORMAP_JET);

    /* Create result image */
    cv::Mat mat_new = cv::Mat::zeros(mat_left.size(), CV_8UC3);
    cv::Mat target = mat_new(cv::Rect(ss_result.crop.x, ss_result.crop.y, ss_result.crop.w, ss_result.crop.h));
    cv::resize(mat_depth, target, target.size());

    cv::vconcat(mat_left, mat_new, mat_result);
    // cv::resize(mat_result, mat_result, cv::Size(), 0.5, 0.5);   // just to fit to my display

    DrawFps(mat_result, ss_result.time_inference, cv::Point(0, 0), 0.5, 2, CommonHelper::CreateCvColor(0, 0, 0), CommonHelper::CreateCvColor(180, 180, 180), true);

    /* Return the results */
    result.time_pre_process = ss_result.time_pre_process;
    result.time_inference = ss_result.time_inference;
    result.time_post_process = ss_result.time_post_process;

    return 0;
}

