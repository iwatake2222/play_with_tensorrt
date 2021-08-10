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
#include "detection_engine.h"

/*** Macro ***/
#define TAG "DetectionEngine"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/* Model parameters */
#define MODEL_NAME  "ctdet_coco_dlav0_384.onnx"
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#define INPUT_NAME  "input.1"
#define INPUT_DIMS  { 1, 3, 384, 384 }
#define IS_NCHW     true
#define IS_RGB      false   /* it looks they use BGR. https://github.com/xingyizhou/CenterNet/blob/master/src/lib/detectors/base_detector.py#L37 */
#define OUTPUT_NAME_0 "508" /* 80 x 96 x 96. heat map (score for each class) */
#define OUTPUT_NAME_1 "511" /*  2 x 96 x 96. regressor. xy */
#define OUTPUT_NAME_2 "514" /*  2 x 96 x 96. regressor. wh */
#define HM_HEIGHT  96
#define HM_WIDTH   96
#define HM_CHANNEL 80

#define LABEL_NAME   "label_coco_80.txt"


/*** Function ***/
int32_t DetectionEngine::Initialize(const std::string& work_dir, const int32_t num_threads)
{
    /* Set model information */
    std::string model_filename = work_dir + "/model/" + MODEL_NAME;
    std::string labelFilename = work_dir + "/model/" + LABEL_NAME;

    /* Set input tensor info */
    input_tensor_info_list_.clear();
    InputTensorInfo input_tensor_info(INPUT_NAME, TENSORTYPE, IS_NCHW);
    input_tensor_info.tensor_dims = INPUT_DIMS;
    input_tensor_info.data_type = InputTensorInfo::kDataTypeImage;
    input_tensor_info.normalize.mean[0] = 0.408f;
    input_tensor_info.normalize.mean[1] = 0.447f;
    input_tensor_info.normalize.mean[2] = 0.470f;
    input_tensor_info.normalize.norm[0] = 0.289f;
    input_tensor_info.normalize.norm[1] = 0.274f;
    input_tensor_info.normalize.norm[2] = 0.278f;
    input_tensor_info_list_.push_back(input_tensor_info);

    /* Set output tensor info */
    output_tensor_info_list_.clear();
    output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_NAME_0, TENSORTYPE));
    output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_NAME_1, TENSORTYPE));
    output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_NAME_2, TENSORTYPE));

    /* Create and Initialize Inference Helper */
    //inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kOpencv));
    //inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kOpencvGpu));
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

    /* read label */
    if (ReadLabel(labelFilename, label_list_) != kRetOk) {
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
    //CommonHelper::CropResizeCvt(original_mat, img_src, crop_x, crop_y, crop_w, crop_h, IS_RGB, CommonHelper::kCropTypeStretch);
    //CommonHelper::CropResizeCvt(original_mat, img_src, crop_x, crop_y, crop_w, crop_h, IS_RGB, CommonHelper::kCropTypeCut);
    CommonHelper::CropResizeCvt(original_mat, img_src, crop_x, crop_y, crop_w, crop_h, IS_RGB, CommonHelper::kCropTypeExpand);

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
    /* Get boundig box */
    float* hm_list = output_tensor_info_list_[0].GetDataAsFloat();
    float* reg_xy_list = output_tensor_info_list_[1].GetDataAsFloat();
    float* reg_wh_list = output_tensor_info_list_[2].GetDataAsFloat();
    const int32_t hm_h = output_tensor_info_list_[0].GetHeight() != -1 ? output_tensor_info_list_[0].GetHeight() : HM_HEIGHT;
    const int32_t hm_w = output_tensor_info_list_[0].GetWidth() != -1 ? output_tensor_info_list_[0].GetWidth() : HM_WIDTH;
    const int32_t hm_c = output_tensor_info_list_[0].GetChannel() > 1 ? output_tensor_info_list_[0].GetChannel() : HM_CHANNEL;
    const float threshold_score_logit = CommonHelper::Logit(threshold_class_confidence_);
    const float scale_w = static_cast<float>(crop_w) / input_tensor_info.GetWidth();
    const float scale_h = static_cast<float>(crop_h) / input_tensor_info.GetHeight();

    /* https://github.com/xingyizhou/CenterNet/blob/master/src/lib/models/decode.py#L472 */
    std::vector<BoundingBox> bbox_list;
    for (int32_t class_id = 0; class_id < hm_c; class_id++) {
        for (int32_t hm_y = 0; hm_y < hm_h; hm_y++) {
            for (int32_t hm_x = 0; hm_x < hm_w; hm_x++) {
                const float score_logit = *hm_list;
                hm_list++;
                if (score_logit > threshold_score_logit) {
                    const int32_t index_x = hm_w * hm_y + hm_x;
                    const int32_t index_y = index_x + hm_h * hm_w;
                    const float width = reg_wh_list[index_x];
                    const float height = reg_wh_list[index_y];
                    const float cx = hm_x + reg_xy_list[index_x];  /* no need to add +0.5f according to sample code */
                    const float cy = hm_y + reg_xy_list[index_y];
                    const float x0 = cx - width / 2.0f;
                    const float y0 = cy - height / 2.0f;

                    BoundingBox bbox;
                    bbox.class_id = class_id;
                    bbox.label = label_list_[class_id];
                    bbox.score = CommonHelper::Sigmoid(score_logit);
                    bbox.x = static_cast<int32_t>(x0 * 4 * scale_w);
                    bbox.y = static_cast<int32_t>(y0 * 4 * scale_h);
                    bbox.w = static_cast<int32_t>(width * 4 * scale_w);
                    bbox.h = static_cast<int32_t>(height * 4 * scale_h);
                    bbox_list.push_back(bbox);
                }
            }
        }
    }

    /* Adjust bounding box */
    for (auto& bbox : bbox_list) {
        bbox.x += crop_x;  
        bbox.y += crop_y;
        bbox.label = label_list_[bbox.class_id];
    }

    /* NMS */
    std::vector<BoundingBox> bbox_nms_list;
    BoundingBoxUtils::Nms(bbox_list, bbox_nms_list, threshold_nms_iou_);

    const auto& t_post_process1 = std::chrono::steady_clock::now();

    /* Return the results */
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


int32_t DetectionEngine::ReadLabel(const std::string& filename, std::vector<std::string>& label_list)
{
    std::ifstream ifs(filename);
    if (ifs.fail()) {
        PRINT_E("Failed to read %s\n", filename.c_str());
        return kRetErr;
    }
    label_list.clear();
    std::string str;
    while (getline(ifs, str)) {
        label_list.push_back(str);
    }
    return kRetOk;
}

