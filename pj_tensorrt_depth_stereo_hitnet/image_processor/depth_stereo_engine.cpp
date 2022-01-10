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

/* for OpenCV */
#include <opencv2/opencv.hpp>

/* for My modules */
#include "common_helper.h"
#include "common_helper_cv.h"
#include "inference_helper.h"
#include "inference_helper_tensorrt.h"      // to call SetDlaCore
#include "depth_stereo_engine.h"

/*** Macro ***/
#define TAG "DepthStereoEngine"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/* Model parameters */
#define MODEL_NAME   "hitnet_eth3d_480x640.onnx"
#define IS_GRAYSCALE
#define MAX_DISPLARITY 128
//#define MODEL_NAME   "hitnet_flyingthings_finalpass_xl_480x640.onnx"
//#undef IS_GRAYSCALE
//#define MAX_DISPLARITY 320
//#define MODEL_NAME   "hitnet_middlebury_d400_480x640.onnx"
//#undef IS_GRAYSCALE
//#define MAX_DISPLARITY 400

#ifdef IS_GRAYSCALE
#define INPUT_DIMS    { 1, 2, 480, 640 }
#else
#define INPUT_DIMS    { 1, 6, 480, 640 }
#endif
#define IS_NCHW       true
#define INPUT_NAME   "input"
#define OUTPUT_NAME  "reference_output_disparity"
#define TENSORTYPE    TensorInfo::kTensorTypeFp32

/*** Function ***/
int32_t DepthStereoEngine::Initialize(const std::string& work_dir, const int32_t num_threads)
{
    /* Set model information */
    std::string model_filename = work_dir + "/model/" + MODEL_NAME;

    /* Set input tensor info */
    input_tensor_info_list_.clear();
    InputTensorInfo input_tensor_info(INPUT_NAME, TENSORTYPE, IS_NCHW);
    input_tensor_info.tensor_dims = INPUT_DIMS;
    input_tensor_info.data_type = InputTensorInfo::kDataTypeBlobNchw;
    input_tensor_info_list_.push_back(input_tensor_info);

    /* Set output tensor info */
    output_tensor_info_list_.clear();
    output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_NAME, TENSORTYPE));

    /* Create and Initialize Inference Helper */
    inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorrt));
    if (!inference_helper_) {
        return kRetErr;
    }
    InferenceHelperTensorRt* p = dynamic_cast<InferenceHelperTensorRt*>(inference_helper_.get());
    if (p) p->SetDlaCore(-1);  /* Use GPU */
    if (inference_helper_->SetNumThreads(num_threads) != InferenceHelper::kRetOk) {
        inference_helper_.reset();
        return kRetErr;
    }
    if (inference_helper_->Initialize(model_filename, input_tensor_info_list_, output_tensor_info_list_) != InferenceHelper::kRetOk) {
        inference_helper_.reset();
        return kRetErr;
    }

    return kRetOk;
}

int32_t DepthStereoEngine::Finalize()
{
    if (!inference_helper_) {
        PRINT_E("Inference helper is not created\n");
        return kRetErr;
    }
    inference_helper_->Finalize();
    return kRetOk;
}


int32_t DepthStereoEngine::Process(const cv::Mat& image_src_l, const cv::Mat& image_src_r, Result& result)
{
    if (!inference_helper_) {
        PRINT_E("Inference helper is not created\n");
        return kRetErr;
    }

    /*** PreProcess ***/
    const auto& t_pre_process0 = std::chrono::steady_clock::now();
    
    InputTensorInfo& input_tensor_info = input_tensor_info_list_[0];
    /* Do preprocess here and set input data as nchw blob because InferenceHelper cannot handle Grayscale x 2 input */
    cv::Mat image_l;
    cv::Mat image_r;
    cv::resize(image_src_l, image_l, cv::Size(input_tensor_info.GetWidth(), input_tensor_info.GetHeight()));
    cv::resize(image_src_r, image_r, cv::Size(input_tensor_info.GetWidth(), input_tensor_info.GetHeight()));
    int32_t image_size = input_tensor_info.GetWidth() * input_tensor_info.GetHeight();
    
#ifdef IS_GRAYSCALE
    const int32_t image_channel = 1;
    cv::cvtColor(image_l, image_l, cv::COLOR_BGR2GRAY);
    cv::cvtColor(image_r, image_r, cv::COLOR_BGR2GRAY);
#else
    const int32_t image_channel = 3;
    cv::cvtColor(image_l, image_l, cv::COLOR_BGR2RGB);
    cv::cvtColor(image_r, image_r, cv::COLOR_BGR2RGB);
#endif
    
    auto data = std::make_unique<float[]>(image_size * image_channel * 2);
#pragma omp parallel for
    for (int32_t c = 0; c < image_channel; c++) {
        for (int32_t i = 0; i < image_size; i++) {
            data[i + c * image_size] = image_l.data[i * image_channel + c] / 255.0f;
        }
    }

    const int32_t offset_for_right_image = image_size * image_channel;
#pragma omp parallel for
    for (int32_t c = 0; c < image_channel; c++) {
        for (int32_t i = 0; i < image_size; i++) {
            data[i + c * image_size + offset_for_right_image] = image_r.data[i * image_channel + c] / 255.0f;
        }
    }

    input_tensor_info.data = data.get();
   
    if (inference_helper_->PreProcess(input_tensor_info_list_) != InferenceHelper::kRetOk) {
        return kRetErr;
    }
    const auto& t_pre_process1 = std::chrono::steady_clock::now();

    /*** Inference ***/
    const auto& t_inference0 = std::chrono::steady_clock::now();
    if (inference_helper_->Process(output_tensor_info_list_) != InferenceHelper::kRetOk) {
        return kRetErr;
    }
    const auto& t_inference1 = std::chrono::steady_clock::now();

    /*** PostProcess ***/
    const auto& t_post_process0 = std::chrono::steady_clock::now();
    int32_t output_height = output_tensor_info_list_[0].tensor_dims[1];
    int32_t output_width = output_tensor_info_list_[0].tensor_dims[2];
    float* values = output_tensor_info_list_[0].GetDataAsFloat();

    cv::Mat out_fp = cv::Mat(output_height, output_width, CV_32FC1, values);

    const auto& t_post_process1 = std::chrono::steady_clock::now();

    /* Return the results */
    result.image = out_fp;
    result.crop.x = 0;
    result.crop.y = 0;
    result.crop.w = image_src_l.cols;
    result.crop.h = image_src_l.rows;
    result.time_pre_process = static_cast<std::chrono::duration<double>>(t_pre_process1 - t_pre_process0).count() * 1000.0;
    result.time_inference = static_cast<std::chrono::duration<double>>(t_inference1 - t_inference0).count() * 1000.0;
    result.time_post_process = static_cast<std::chrono::duration<double>>(t_post_process1 - t_post_process0).count() * 1000.0;;

    return kRetOk;
}

float DepthStereoEngine::GetMaxDisparity(void)
{
    return MAX_DISPLARITY;
}