/* Copyright 2021 iwatake2222

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
/*** Include ***/
/* for general */
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <memory>

/* for OpenCV */
#include <opencv2/opencv.hpp>

/* for My modules */
#include "common_helper.h"
#include "common_helper_cv.h"
#include "segmentation_engine.h"
#include "image_processor.h"

/*** Macro ***/
#define TAG "ImageProcessor"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/*** Global variable ***/
std::unique_ptr<SegmentationEngine> s_engine;

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

static cv::Scalar GetColorForId(int32_t id)
{
    static constexpr int32_t kGap = 17;          // better to select a number can divide 255
    static constexpr int32_t kIncInGap = 255 / kGap;
    uint32_t index = (id % kGap) * kIncInGap + id / kGap;
    index %= 255;

    static std::vector<cv::Scalar> color_list;
    if (color_list.empty()) {
        std::vector<uint8_t> seq_num(256);
        std::iota(seq_num.begin(), seq_num.end(), 0);
        cv::Mat mat_seq = cv::Mat(256, 1, CV_8UC1, seq_num.data());
        cv::Mat mat_colormap;
        cv::applyColorMap(mat_seq, mat_colormap, cv::COLORMAP_JET);
        for (int32_t i = 0; i < 256; i++) {
            const auto& bgr = mat_colormap.at<cv::Vec3b>(i);
            color_list.push_back(CommonHelper::CreateCvColor(bgr[0], bgr[1], bgr[2]));
        }
    }
    return color_list[index];
}


int32_t ImageProcessor::Initialize(const InputParam& input_param)
{
    GetColorForId(0);   // just to initialize color map

    if (s_engine) {
        PRINT_E("Already initialized\n");
        return -1;
    }

    s_engine.reset(new SegmentationEngine());
    if (s_engine->Initialize(input_param.work_dir, input_param.num_threads) != SegmentationEngine::kRetOk) {
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

    if (s_engine->Finalize() != SegmentationEngine::kRetOk) {
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


int32_t ImageProcessor::Process(cv::Mat& mat, Result& result)
{
    if (!s_engine) {
        PRINT_E("Not initialized\n");
        return -1;
    }

    cv::resize(mat, mat, cv::Size(), 0.5, 0.5);

    SegmentationEngine::Result segmentation_result;
    if (s_engine->Process(mat, segmentation_result) != SegmentationEngine::kRetOk) {
        return -1;
    }

    /* Convert to colored depth map */
    cv::Mat mat_segmentation = cv::Mat::zeros(segmentation_result.mat_out_list[0].size(), CV_8UC3);
#pragma omp parallel for
    for (int32_t i = 0; i < segmentation_result.mat_out_list.size(); i++) {
        auto& mat_out = segmentation_result.mat_out_list[i];
        cv::cvtColor(mat_out, mat_out, cv::COLOR_GRAY2BGR); /* 1channel -> 3 channel */
        cv::multiply(mat_out, GetColorForId(i), mat_out);
        mat_out.convertTo(mat_out, CV_8UC1);
    }
    for (int32_t i = 0; i < segmentation_result.mat_out_list.size(); i++) {
        auto& mat_out = segmentation_result.mat_out_list[i];
        cv::add(mat_segmentation, mat_out, mat_segmentation);
    }

    /* Create result image */
    //mat = mat_segmentation;
    double scale = static_cast<double>(mat.cols) / mat_segmentation.cols;
    cv::resize(mat_segmentation, mat_segmentation, cv::Size(), scale, scale);
    cv::add(mat_segmentation, mat, mat_segmentation);
    cv::vconcat(mat, mat_segmentation, mat);

    DrawFps(mat, segmentation_result.time_inference, cv::Point(0, 0), 0.5, 2, CommonHelper::CreateCvColor(0, 0, 0), CommonHelper::CreateCvColor(180, 180, 180), true);

    /* Return the results */
    result.time_pre_process = segmentation_result.time_pre_process;
    result.time_inference = segmentation_result.time_inference;
    result.time_post_process = segmentation_result.time_post_process;

    return 0;
}

