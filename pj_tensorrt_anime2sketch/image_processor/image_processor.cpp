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
#include "anime_to_sketch_engine.h"
#include "image_processor.h"

/*** Macro ***/
#define TAG "ImageProcessor"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/*** Global variable ***/
std::unique_ptr<Anime2SketchEngine> s_anime2SketchEngine;

/*** Function ***/
static cv::Scalar CreateCvColor(int32_t b, int32_t g, int32_t r) {
#ifdef CV_COLOR_IS_RGB
    return cv::Scalar(r, g, b);
#else
    return cv::Scalar(b, g, r);
#endif
}


int32_t ImageProcessor::Initialize(const InputParam* input_param)
{
    if (s_anime2SketchEngine) {
        PRINT_E("Already initialized\n");
        return -1;
    }

    s_anime2SketchEngine.reset(new Anime2SketchEngine());
    if (s_anime2SketchEngine->Initialize(input_param->work_dir, input_param->num_threads) != Anime2SketchEngine::kRetOk) {
        s_anime2SketchEngine->Finalize();
        s_anime2SketchEngine.reset();
        return -1;
    }
    return 0;
}

int32_t ImageProcessor::Finalize(void)
{
    if (!s_anime2SketchEngine) {
        PRINT_E("Not initialized\n");
        return -1;
    }

    if (s_anime2SketchEngine->Finalize() != Anime2SketchEngine::kRetOk) {
        return -1;
    }

    return 0;
}


int32_t ImageProcessor::Command(int32_t cmd)
{
    if (!s_anime2SketchEngine) {
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


int32_t ImageProcessor::Process(cv::Mat* mat, OutputParam* output_param)
{
    if (!s_anime2SketchEngine) {
        PRINT_E("Not initialized\n");
        return -1;
    }

    cv::Mat& original_mat = *mat;
    Anime2SketchEngine::Result styleTransferResult;
    s_anime2SketchEngine->Process(original_mat, styleTransferResult);

    /* Return the results */
    original_mat = styleTransferResult.image;
    output_param->time_pre_process = styleTransferResult.time_pre_process;
    output_param->time_inference = styleTransferResult.time_inference;
    output_param->time_post_process = styleTransferResult.time_post_process;

    return 0;
}

