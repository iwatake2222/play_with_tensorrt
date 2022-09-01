/* Copyright 2022 iwatake2222

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
#include <algorithm>
#include <chrono>
#include <fstream>

/* for OpenCV */
#include <opencv2/opencv.hpp>

/* for My modules */
#include "common_helper.h"
#include "common_helper_cv.h"
#include "inference_helper.h"
#include "detection_engine.h"

/*** Macro ***/
#define TAG "DetectionEngine"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/* Model parameters */
#define MODEL_NAME  "yolopv2_384x640.onnx"
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#define INPUT_NAME  "input"
#define INPUT_DIMS  { 1, 3, 384, 640 }
#define IS_NCHW     true
#define IS_RGB      true
#define OUTPUT_NAME_0 "seg"
#define OUTPUT_NAME_1 "ll"
#define OUTPUT_NAME_2 "pred0"
#define OUTPUT_NAME_3 "pred1"
#define OUTPUT_NAME_4 "pred2"
#define OUTPUT_NAME_5 "anchor_grid0"
#define OUTPUT_NAME_6 "anchor_grid1"
#define OUTPUT_NAME_7 "anchor_grid2"

/* from model_105_anchor_grid.npy */
static constexpr float kAnchorGrid8[3][2] = { { 12, 16 }, { 19, 36 }, { 40, 28 } };
static constexpr float kAnchorGrid16[3][2] = { { 36, 75 }, { 76, 55 }, { 72, 146 } };
static constexpr float kAnchorGrid32[3][2] = { { 142, 110 }, { 192, 243 }, { 459, 401 } };

static const std::vector<std::string> kLabelListDet{ "Car" };
static const std::vector<std::string> kLabelListSeg{ "Background", "Road", "Line" };


/*** Function ***/
int32_t DetectionEngine::Initialize(const std::string& work_dir, const int32_t num_threads)
{
    /* Set model information */
    std::string model_filename = work_dir + "/model/" + MODEL_NAME;

    /* Set input tensor info */
    input_tensor_info_list_.clear();
    InputTensorInfo input_tensor_info(INPUT_NAME, TENSORTYPE, IS_NCHW);
    input_tensor_info.tensor_dims = INPUT_DIMS;
    input_tensor_info.data_type = InputTensorInfo::kDataTypeImage;
    /* [0, 255] -> [0.0, 1.0] */
    input_tensor_info.normalize.mean[0] = 0.0f;
    input_tensor_info.normalize.mean[1] = 0.0f;
    input_tensor_info.normalize.mean[2] = 0.0f;
    input_tensor_info.normalize.norm[0] = 1.0f;
    input_tensor_info.normalize.norm[1] = 1.0f;
    input_tensor_info.normalize.norm[2] = 1.0f;
    input_tensor_info_list_.push_back(input_tensor_info);

    /* Set output tensor info */
    output_tensor_info_list_.clear();
    output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_NAME_0, TENSORTYPE));
    output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_NAME_1, TENSORTYPE));
    output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_NAME_2, TENSORTYPE));
    output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_NAME_3, TENSORTYPE));
    output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_NAME_4, TENSORTYPE));
    output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_NAME_5, TENSORTYPE));
    output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_NAME_6, TENSORTYPE));
    output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_NAME_7, TENSORTYPE));

    /* Create and Initialize Inference Helper */
    //inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kOnnxRuntime));
    inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorrt));

    if (!inference_helper_) {
        return kRetErr;
    }
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

int32_t DetectionEngine::Finalize()
{
    if (!inference_helper_) {
        PRINT_E("Inference helper is not created\n");
        return kRetErr;
    }
    inference_helper_->Finalize();
    return kRetOk;
}

/* reference: https://github.com/CAIC-AD/YOLOPv2/blob/main/utils/utils.py#L170 */
/* [1,255,48,80] = [1, 3, 85, 48, 80] = [1, 3, (x, y, w, h, prob, prob x80), ny nx] */
std::vector<BoundingBox> DetectionEngine::GetBoundingBox(std::vector<float> pred, int32_t input_width, int32_t input_height, int32_t st, const float anchor_grid[3][2], float scale_w, float scale_h)
{
    std::vector<BoundingBox> bbox_list;
    size_t nx = input_width / st;
    size_t ny = input_height / st;
    for (size_t n = 0; n < 3; n++) {
        size_t offset_n = n * 85 * ny * nx;
        for (size_t y = 0; y < ny; y++) {
            for (size_t x = 0; x < nx; x++) {
                size_t offset_xy = x + y * nx;
                size_t index_prob = offset_n + 4 * ny * nx + offset_xy;
                float prob = CommonHelper::Sigmoid(pred[index_prob]);
                if (prob > threshold_class_confidence_) {
                    size_t index_x = offset_n + 0 * ny * nx + offset_xy;
                    size_t index_y = offset_n + 1 * ny * nx + offset_xy;
                    size_t index_w = offset_n + 2 * ny * nx + offset_xy;
                    size_t index_h = offset_n + 3 * ny * nx + offset_xy;
                    float cx = (CommonHelper::Sigmoid(pred[index_x]) * 2 - 0.5 + x) * st;
                    float cy = (CommonHelper::Sigmoid(pred[index_y]) * 2 - 0.5 + y) * st;
                    float w = std::pow(CommonHelper::Sigmoid(pred[index_w]) * 2, 2) * anchor_grid[n][0];
                    float h = std::pow(CommonHelper::Sigmoid(pred[index_h]) * 2, 2) * anchor_grid[n][1];

                    /* Store the detected box */
                    auto bbox = BoundingBox{
                        static_cast<int32_t>(0),
                        kLabelListDet[0],
                        prob,
                        static_cast<int32_t>((cx - w / 2.0) * scale_w),
                        static_cast<int32_t>((cy - h / 2.0) * scale_h),
                        static_cast<int32_t>(w * scale_w),
                        static_cast<int32_t>(h * scale_h)
                    };
                    bbox_list.push_back(bbox);
                }
            }
        }
    }
    return bbox_list;
}

int32_t DetectionEngine::Process(const cv::Mat& original_mat, Result& result)
{
    if (!inference_helper_) {
        PRINT_E("Inference helper is not created\n");
        return kRetErr;
    }
    /*** PreProcess ***/
    const auto& t_pre_process0 = std::chrono::steady_clock::now();
    InputTensorInfo& input_tensor_info = input_tensor_info_list_[0];
    /* do crop, resize and color conversion here because some inference engine doesn't support these operations */
    int32_t crop_x = 0;
    int32_t crop_y = 0;
    int32_t crop_w = original_mat.cols;
    int32_t crop_h = original_mat.rows;
    cv::Mat img_src = cv::Mat::zeros(input_tensor_info.GetHeight(), input_tensor_info.GetWidth(), CV_8UC3);
    CommonHelper::CropResizeCvt(original_mat, img_src, crop_x, crop_y, crop_w, crop_h, IS_RGB, CommonHelper::kCropTypeStretch);
    //CommonHelper::CropResizeCvt(original_mat, img_src, crop_x, crop_y, crop_w, crop_h, IS_RGB, CommonHelper::kCropTypeCut);
    //CommonHelper::CropResizeCvt(original_mat, img_src, crop_x, crop_y, crop_w, crop_h, IS_RGB, CommonHelper::kCropTypeExpand);

    input_tensor_info.data = img_src.data;
    input_tensor_info.data_type = InputTensorInfo::kDataTypeImage;
    input_tensor_info.image_info.width = img_src.cols;
    input_tensor_info.image_info.height = img_src.rows;
    input_tensor_info.image_info.channel = img_src.channels();
    input_tensor_info.image_info.crop_x = 0;
    input_tensor_info.image_info.crop_y = 0;
    input_tensor_info.image_info.crop_width = img_src.cols;
    input_tensor_info.image_info.crop_height = img_src.rows;
    input_tensor_info.image_info.is_bgr = false;
    input_tensor_info.image_info.swap_color = false;
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
    /* Retrieve the result */
    std::vector<float> output_seg_list(output_tensor_info_list_[0].GetDataAsFloat(), output_tensor_info_list_[0].GetDataAsFloat() + output_tensor_info_list_[0].GetElementNum());
    std::vector<float> output_ll_list(output_tensor_info_list_[1].GetDataAsFloat(), output_tensor_info_list_[1].GetDataAsFloat() + output_tensor_info_list_[1].GetElementNum());
    std::vector<float> output_pred0_list(output_tensor_info_list_[2].GetDataAsFloat(), output_tensor_info_list_[2].GetDataAsFloat() + output_tensor_info_list_[2].GetElementNum());
    std::vector<float> output_pred1_list(output_tensor_info_list_[3].GetDataAsFloat(), output_tensor_info_list_[3].GetDataAsFloat() + output_tensor_info_list_[3].GetElementNum());
    std::vector<float> output_pred2_list(output_tensor_info_list_[4].GetDataAsFloat(), output_tensor_info_list_[4].GetDataAsFloat() + output_tensor_info_list_[4].GetElementNum());

    /* Get Segmentation result. ArgMax */
    cv::Mat mat_seg_max = cv::Mat::zeros(input_tensor_info.GetHeight(), input_tensor_info.GetWidth(), CV_8UC1);
#pragma omp parallel for
    for (int32_t y = 0; y < input_tensor_info.GetHeight(); y++) {
        for (int32_t x = 0; x < input_tensor_info.GetWidth(); x++) {
            int32_t class_index_max = 0;
            float class_score_max = 0;
            for (int32_t class_index = 0; class_index < 2; class_index++) {
                float score = output_seg_list[class_index * input_tensor_info.GetHeight() * input_tensor_info.GetWidth() + input_tensor_info.GetWidth() * y + x];
                if (score > class_score_max) {
                    class_score_max = score;
                    class_index_max = class_index;
                }
            }
            /* Overwrite if ll score is high */
            float score_ll = output_ll_list[input_tensor_info.GetWidth() * y + x];
            if (score_ll > threshold_seg_ll_) {
                class_index_max = 2;    /* 2 = line */
            }
            mat_seg_max.at<uint8_t>(cv::Point(x, y)) = static_cast<uint8_t>(class_index_max);
        }
    }

    /* Get boundig box */
    std::vector<BoundingBox> bbox_list;
    float scale_w = static_cast<float>(crop_w) / input_tensor_info.GetWidth();
    float scale_h = static_cast<float>(crop_h) / input_tensor_info.GetHeight();
    auto bbox_list_8 = GetBoundingBox(output_pred0_list, input_tensor_info.GetWidth(), input_tensor_info.GetHeight(), 8, kAnchorGrid8, scale_w, scale_h);
    auto bbox_list_16 = GetBoundingBox(output_pred1_list, input_tensor_info.GetWidth(), input_tensor_info.GetHeight(), 16, kAnchorGrid16, scale_w, scale_h);
    auto bbox_list_32 = GetBoundingBox(output_pred2_list, input_tensor_info.GetWidth(), input_tensor_info.GetHeight(), 32, kAnchorGrid32, scale_w, scale_h);
    bbox_list.insert(bbox_list.end(), bbox_list_8.begin(), bbox_list_8.end());
    bbox_list.insert(bbox_list.end(), bbox_list_16.begin(), bbox_list_16.end());
    bbox_list.insert(bbox_list.end(), bbox_list_32.begin(), bbox_list_32.end());

    /* Adjust bounding box */
    for (auto& bbox : bbox_list) {
        bbox.x += crop_x;  
        bbox.y += crop_y;
    }

    /* NMS */
    std::vector<BoundingBox> bbox_nms_list;
    BoundingBoxUtils::Nms(bbox_list, bbox_nms_list, threshold_nms_iou_);

    const auto& t_post_process1 = std::chrono::steady_clock::now();

    /* Return the results */
    result.mat_seg_max = mat_seg_max;
    result.bbox_list = bbox_nms_list;
    result.crop.x = (std::max)(0, crop_x);
    result.crop.y = (std::max)(0, crop_y);
    result.crop.w = (std::min)(crop_w, original_mat.cols - result.crop.x);
    result.crop.h = (std::min)(crop_h, original_mat.rows - result.crop.y);
    result.time_pre_process = static_cast<std::chrono::duration<double>>(t_pre_process1 - t_pre_process0).count() * 1000.0;
    result.time_inference = static_cast<std::chrono::duration<double>>(t_inference1 - t_inference0).count() * 1000.0;
    result.time_post_process = static_cast<std::chrono::duration<double>>(t_post_process1 - t_post_process0).count() * 1000.0;;

    return kRetOk;
}
