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
#define MODEL_NAME   "coex_480x640.onnx"
#define TENSORTYPE    TensorInfo::kTensorTypeFp32
#define INPUT_NAME_0   "imgL"
#define INPUT_NAME_1   "imgR"
#define INPUT_DIMS    { 1, 3, 480, 640 }
#define IS_NCHW       true
#define IS_RGB        false
#define OUTPUT_NAME  "1603"

/*** Function ***/
int32_t DepthStereoEngine::Initialize(const std::string& work_dir, const int32_t num_threads)
{
    /* Set model information */
    std::string model_filename = work_dir + "/model/" + MODEL_NAME;

    /* Set input tensor info */
    input_tensor_info_list_.clear();
    InputTensorInfo input_tensor_info(INPUT_NAME_0, TENSORTYPE, IS_NCHW);
    input_tensor_info.tensor_dims = INPUT_DIMS;
    input_tensor_info.data_type = InputTensorInfo::kDataTypeImage;
    input_tensor_info.normalize.mean[0] = 0.485f;
    input_tensor_info.normalize.mean[1] = 0.456f;
    input_tensor_info.normalize.mean[2] = 0.406f;
    input_tensor_info.normalize.norm[0] = 0.229f;
    input_tensor_info.normalize.norm[1] = 0.224f;
    input_tensor_info.normalize.norm[2] = 0.225f;
    input_tensor_info_list_.push_back(input_tensor_info);

    input_tensor_info.name = INPUT_NAME_1;
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


int32_t DepthStereoEngine::Process(const cv::Mat& image_l, const cv::Mat& image_r, Result& result)
{
    if (!inference_helper_) {
        PRINT_E("Inference helper is not created\n");
        return kRetErr;
    }

    /*** PreProcess ***/
    const auto& t_pre_process0 = std::chrono::steady_clock::now();
    cv::Mat img_src[2];
    int32_t crop_x;
    int32_t crop_y;
    int32_t crop_w;
    int32_t crop_h;
    for (int32_t i = 0; i < 2; i++) {
        InputTensorInfo& input_tensor_info = input_tensor_info_list_[i];
        const cv::Mat& original_mat = (i == 0) ? image_l : image_r;
        /* do resize and color conversion here because some inference engine doesn't support these operations */
        crop_x = 0;
        crop_y = 0;
        crop_w = original_mat.cols;
        crop_h = original_mat.rows;
        img_src[i] = cv::Mat::zeros(input_tensor_info.GetHeight(), input_tensor_info.GetWidth(), CV_8UC3);
        // CommonHelper::CropResizeCvt(original_mat, img_src[i], crop_x, crop_y, crop_w, crop_h, IS_RGB, CommonHelper::kCropTypeStretch);
        CommonHelper::CropResizeCvt(original_mat, img_src[i], crop_x, crop_y, crop_w, crop_h, IS_RGB, CommonHelper::kCropTypeCut);
        // CommonHelper::CropResizeCvt(original_mat, img_src[i], crop_x, crop_y, crop_w, crop_h, IS_RGB, CommonHelper::kCropTypeExpand);

        input_tensor_info.data = img_src[i].data;
        input_tensor_info.data_type = InputTensorInfo::kDataTypeImage;
        input_tensor_info.image_info.width = img_src[i].cols;
        input_tensor_info.image_info.height = img_src[i].rows;
        input_tensor_info.image_info.channel = img_src[i].channels();
        input_tensor_info.image_info.crop_x = 0;
        input_tensor_info.image_info.crop_y = 0;
        input_tensor_info.image_info.crop_width = img_src[i].cols;
        input_tensor_info.image_info.crop_height = img_src[i].rows;
        input_tensor_info.image_info.is_bgr = false;
        input_tensor_info.image_info.swap_color = false;
    }

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
    cv::Mat out_mat;
    out_fp.convertTo(out_mat, CV_8UC1);

    const auto& t_post_process1 = std::chrono::steady_clock::now();

    /* Return the results */
    result.image = out_mat;
    result.crop.x = (std::max)(0, crop_x);
    result.crop.y = (std::max)(0, crop_y);
    result.crop.w = (std::min)(crop_w, image_l.cols - result.crop.x);
    result.crop.h = (std::min)(crop_h, image_l.rows - result.crop.y);
    result.time_pre_process = static_cast<std::chrono::duration<double>>(t_pre_process1 - t_pre_process0).count() * 1000.0;
    result.time_inference = static_cast<std::chrono::duration<double>>(t_inference1 - t_inference0).count() * 1000.0;
    result.time_post_process = static_cast<std::chrono::duration<double>>(t_post_process1 - t_post_process0).count() * 1000.0;;

    return kRetOk;
}

